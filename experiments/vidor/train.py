# -*- coding: utf-8 -*-
'''
vidor_trainval_anomaly_2lr.py

VidOR train->val logical-consistency anomaly runner (2-level rules), patterned after
oi_trainval_anomaly_2lr.py.

- Leaves: obj:<category> and rel:<subj>-<predicate>-<obj>
- Leaf bank: CNN over frames, temporal mean-pool
- Rules:
  * Structural: rel:* -> obj:subj and rel:* -> obj:obj (if those obj leaves exist)
  * Mined direction-aware pairs A=>B on TRAIN labels
  * Compound rules (2-level):
      - (A1 AND A2) -> B   (or inverted B -> (A1 AND A2) if mined direction flips)
      - optional A -> (B1 OR B2) for alternative consequents under A
- Pseudo anomaly GT: anomaly=1 iff any rule is FALSE under GT leaves.
- Score: antecedent-gated violation s_r = gate(p_ante,tau) * (1 - p_rule).

Assumed layout:
  <vidor_root>/annotations/{train,val}/*.json
  <vidor_root>/videos/<video_path from json>  (or fallback by video_id)

Deps:
  pip install opencv-python scikit-learn dgl torchvision
'''

import os, json, argparse, hashlib, random, glob
from typing import List, Tuple, Dict, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import dgl
from tqdm import tqdm, trange

from chimera_logic.trainer import LevelwiseTrainer, SubtreeCache, CacheConfig, subtree_signature  # type: ignore
from chimera_logic.evaluator import propagate_truth_values_hard_from_leaf_labels  # type: ignore

try:
    import cv2
except Exception:
    cv2 = None


# -----------------------------
# misc
# -----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encoder_fingerprint(leaf_bank: nn.Module, manual_tag: str = "") -> str:
    if manual_tag:
        return f"manual_{manual_tag}"
    try:
        h = hashlib.sha256()
        sd = leaf_bank.state_dict()
        for k in sorted(sd.keys()):
            v = sd[k]
            if isinstance(v, torch.Tensor):
                h.update(v.detach().cpu().numpy().tobytes())
        return h.hexdigest()[:24]
    except Exception:
        return hashlib.sha256(repr(leaf_bank).encode()).hexdigest()[:24]


def _rule_sig(g: dgl.DGLGraph) -> str:
    root = int(torch.nonzero(g.out_degrees() == 0, as_tuple=False).flatten()[0].item())
    return subtree_signature(g, root)


# -----------------------------
# CLI
# -----------------------------

def build_args():
    p = argparse.ArgumentParser("VidOR train→val anomaly via logical consistency (2-level)")

    # paths / modes
    p.add_argument("--vidor_root", type=str, required=True)
    p.add_argument("--ann_subdir", type=str, default="annotations")
    p.add_argument("--videos_subdir", type=str, default="videos")
    p.add_argument("--split_train", type=str, default="train")
    p.add_argument("--split_val", type=str, default="val")

    p.add_argument("--cache_dir", type=str, default="vidor_rule_cache_trainval_2lr")
    p.add_argument("--run_dir", type=str, default="runs/vidor_trainval_2lr")
    p.add_argument("--leaf_ckpt", type=str, default="checkpoints_trainval_2lr/vidor_leafbank.pt")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--no_stats", action="store_true")
    p.add_argument(
        "--train_missing_only",
        action="store_true",
        help="When training, skip nodes already cached (if trainer supports it).",
    )

    # concepts
    p.add_argument("--topk_obj", type=int, default=40)
    p.add_argument("--topk_rel", type=int, default=200)
    p.add_argument("--min_rel_support", type=int, default=20)
    p.add_argument("--keep_predicates", type=str, default="",
                   help="Optional comma-separated whitelist of predicates (e.g. 'hold,on,near').")

    # rule budgets
    p.add_argument("--max_rules_simple", type=int, default=250)
    p.add_argument("--max_rules_compound", type=int, default=250)
    p.add_argument("--per_B_pair_limit", type=int, default=3)
    p.add_argument("--inv_overlap_min", type=float, default=0.60,
                help="For inverted compounds B->(A1 AND A2), require overlap >= this. Set <=0 to disable.")
    p.add_argument("--inv_overlap_scan", type=int, default=2000,
                help="How many TRAIN jsons to scan for inv-overlap. 0 => use all.")

    # support pruning (rule usefulness under GT labels)
    p.add_argument("--support_scan", type=int, default=2000,
                   help="Scan up to this many TRAIN annotations to estimate per-rule (true/false) support. 0 disables.")
    p.add_argument("--min_true", type=int, default=5,
                   help="Keep a rule only if it is TRUE in at least this many scanned TRAIN samples.")
    p.add_argument("--min_false", type=int, default=1,
                   help="Keep a rule only if it is FALSE in at least this many scanned TRAIN samples.")


    # mining thresholds (direction-aware)
    p.add_argument("--mine_pairs", action="store_true", default=True)
    p.add_argument("--mine_direction", choices=["auto", "forward", "invert"], default="auto")
    p.add_argument("--min_support_count", type=int, default=50)
    p.add_argument("--min_conf_fwd", type=float, default=0.60)
    p.add_argument("--min_conf_rev", type=float, default=0.60)
    p.add_argument("--min_lift", type=float, default=1.10)
    p.add_argument("--min_conf_margin", type=float, default=0.05)

    # OR-consequent compounds
    p.add_argument("--add_or_compounds", action="store_true")
    p.add_argument("--or_overlap_max", type=float, default=0.20)
    p.add_argument("--or_topk_per_A", type=int, default=3)
    # support filtering (rule pruning)
    p.add_argument("--support_filter_rules", action="store_true",
                   help="Filter rules by GT support counts before training (drops near-degenerate rules).")
    p.add_argument("--support_split", choices=["train","val"], default="val")
    p.add_argument("--support_min_true", type=int, default=5)
    p.add_argument("--support_min_false", type=int, default=1)
    p.add_argument("--support_scan_limit", type=int, default=2000)


    # training knobs
    p.add_argument("--epochs_leaf", type=int, default=3)
    p.add_argument("--epochs_level", type=int, default=2)
    p.add_argument("--lr_leaf", type=float, default=1e-3)
    p.add_argument("--lr_level", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_train", type=int, default=64)
    p.add_argument("--batch_eval", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--encoder_tag", type=str, default="")
    p.add_argument(
        "--negatives",
        type=str,
        default="ad_strict",
        choices=["legacy", "ad_strict", "chimera_plus_true", "chimeras_only"],
    )

    # video
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--resize", type=int, default=224)

    # backbone & aug
    p.add_argument("--backbone", type=str, default="resnet18", choices=["tiny", "resnet18"])
    p.add_argument("--augment", action="store_true")

    # calibration
    p.add_argument("--calib_train", action="store_true")
    p.add_argument("--calib_ckpt", type=str, default="checkpoints_trainval_2lr/vidor_temp_scaler.pt")

    # aggregator
    p.add_argument("--agg", type=str, default="min", choices=["geo", "mean", "min", "learned"])
    p.add_argument("--agg_ckpt", type=str, default="checkpoints_trainval_2lr/vidor_agg_2lr.pt")

    # implication gate
    p.add_argument("--gate_tau", type=float, default=0.0)

    # subsample
    p.add_argument("--train_frac", type=float, default=1.0)

    # seed
    p.add_argument("--seed", type=int, default=123)

    return p.parse_args()


@torch.no_grad()
def evaluate_leaf_bank(bank, loader, idx_to_name, device="cuda", threshold=0.5, max_batches=None):
    """
    Prints per-class ROC-AUC, Average Precision, and Accuracy@thr for the leaf bank.

    Assumes:
      - bank.forward_logits(clips) -> (B, K) raw logits
      - labels from loader are multi-hot (B, K) in {0,1}
    """
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
    except Exception as e:
        print(f"[LeafEval] skipped (scikit-learn not available): {type(e).__name__}: {e}")
        return

    import numpy as np

    bank.eval()
    bank.to(device)

    all_logits = []
    all_labels = []

    for b_idx, (x, y) in enumerate(loader):
        if max_batches is not None and b_idx >= int(max_batches):
            break
        x = x.to(device, non_blocking=True)
        logits = bank.forward_logits(x)          # (B, K)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    if len(all_logits) == 0:
        print("[LeafEval] no batches collected.")
        return

    logits = torch.cat(all_logits, 0)           # (N, K)
    labels = torch.cat(all_labels, 0).float()   # (N, K)

    probs = logits.sigmoid().numpy()
    y_true = labels.numpy().astype(np.int64)

    N, K = y_true.shape
    prev_cnt = y_true.sum(axis=0)

    def _name(j):
        if isinstance(idx_to_name, dict):
            return idx_to_name.get(j, f"c{j}")
        if isinstance(idx_to_name, (list, tuple)) and j < len(idx_to_name):
            return idx_to_name[j]
        return f"c{j}"

    print("\n[LeafEval] Per-class metrics on VAL")
    print(" class                              prev    ROC-AUC    AP      Acc@thr")
    print(" ----------------------------------------------------------------------")

    aucs, aps, accs = [], [], []
    for j in range(K):
        yj = y_true[:, j]
        pj = probs[:, j]
        name = str(_name(j))

        pred = (pj >= threshold).astype(np.int64)
        acc = float((pred == yj).mean())

        if yj.max() == yj.min():
            auc = float("nan")
            ap = float("nan")
        else:
            try:
                auc = float(roc_auc_score(yj, pj))
            except Exception:
                auc = float("nan")
            try:
                ap = float(average_precision_score(yj, pj))
            except Exception:
                ap = float("nan")

        print(f" {name:34s} {int(prev_cnt[j]):7d}  {auc:8.3f}  {ap:7.3f}   {acc:8.3f}")

        aucs.append(auc); aps.append(ap); accs.append(acc)

    # macro summaries (ignore NaNs for AUC/AP)
    def _nanmean(xs):
        xs = np.array(xs, dtype=np.float64)
        return float(np.nanmean(xs)) if np.isfinite(xs).any() else float("nan")

    print(" ----------------------------------------------------------------------")
    print(f" [LeafEval] macro ROC-AUC={_nanmean(aucs):.3f} | macro AP={_nanmean(aps):.3f} | macro Acc={float(np.mean(accs)):.3f}\n")


def vidor_labels_from_ann_path(ann_path: str, name_to_idx: Dict[str, int], frames: int) -> torch.Tensor:
    """
    Build the same multi-hot leaf vector as VidORDataset, but WITHOUT decoding video.
    Uses only VidOR annotation JSON (trajectories + relation_instances).
    """
    ann = _read_json(ann_path)

    frame_count = int(ann.get("frame_count", 0))
    sampled_ts = _sample_frame_indices(frame_count if frame_count > 0 else 1, frames)

    tid2cat = _tid_to_cat(ann)

    traj = ann.get("trajectories", []) or []
    obj_present: Set[str] = set()
    for t in sampled_ts:
        if t < 0 or t >= len(traj):
            continue
        for box in traj[t]:
            try:
                tid = int(box["tid"])
            except Exception:
                continue
            cat = tid2cat.get(tid)
            if cat is not None:
                obj_present.add(cat)

    rel_present: Set[str] = set()
    rels = ann.get("relation_instances", []) or []
    for r in rels:
        try:
            st = int(r["subject_tid"])
            ot = int(r["object_tid"])
            pred = str(r["predicate"])
        except Exception:
            continue
        subj = tid2cat.get(st); obj = tid2cat.get(ot)
        if subj is None or obj is None:
            continue
        if any(_relation_active_at(r, t) for t in sampled_ts):
            rel_present.add(f"{subj}-{pred}-{obj}")
            obj_present.add(subj); obj_present.add(obj)

    K = len(name_to_idx)
    y = torch.zeros(K, dtype=torch.bool)

    for cat in obj_present:
        j = name_to_idx.get(f"obj:{cat}", None)
        if j is not None:
            y[j] = True
    for rel in rel_present:
        j = name_to_idx.get(f"rel:{rel}", None)
        if j is not None:
            y[j] = True

    return y


def inv_overlap_for_pairs(train_paths: List[str],
                          name_to_idx: Dict[str, int],
                          frames: int,
                          b_idx: int,
                          pairs: List[Tuple[int, int]],
                          scan_n: int) -> Dict[Tuple[int, int], float]:
    """
    For fixed B and a list of (A1,A2) index pairs, compute overlap(B;A1,A2) on TRAIN scan:
        overlap = count(B&A1&A2) / min(count(B&A1), count(B&A2))
    Returns mapping (a1,a2)->overlap.
    """
    if scan_n <= 0 or scan_n > len(train_paths):
        scan_n = len(train_paths)

    nBA1 = {p: 0 for p in pairs}
    nBA2 = {p: 0 for p in pairs}
    nBA12 = {p: 0 for p in pairs}

    for ap in train_paths[:scan_n]:
        y = vidor_labels_from_ann_path(ap, name_to_idx, frames)
        if not bool(y[b_idx].item()):
            continue

        for (a1, a2) in pairs:
            a1_on = bool(y[a1].item())
            a2_on = bool(y[a2].item())
            if a1_on:
                nBA1[(a1, a2)] += 1
            if a2_on:
                nBA2[(a1, a2)] += 1
            if a1_on and a2_on:
                nBA12[(a1, a2)] += 1

    out = {}
    for p in pairs:
        denom = max(min(nBA1[p], nBA2[p]), 1)
        out[p] = float(nBA12[p]) / float(denom)
    return out


# -----------------------------
# transforms
# -----------------------------

def make_transforms(split: str, resize: int, use_aug: bool) -> T.Compose:
    if split == "train" and use_aug:
        return T.Compose([
            T.Resize((resize, resize)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.RandomErasing(p=0.1),
        ])
    return T.Compose([T.Resize((resize, resize)), T.ToTensor()])


# -----------------------------
# VidOR parsing
# -----------------------------

def list_ann_paths(root: str, ann_subdir: str, split: str) -> List[str]:
    d = os.path.join(root, ann_subdir, split)
    paths = sorted(glob.glob(os.path.join(d, "*.json")))
    if len(paths) == 0:
        paths = sorted(glob.glob(os.path.join(d, "**", "*.json"), recursive=True))
    return paths


def _read_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _video_path_from_ann(root: str, videos_subdir: str, ann: dict) -> str:
    rel = ann.get("video_path", None)
    if isinstance(rel, str) and rel:
        return os.path.join(root, videos_subdir, rel)
    vid = ann.get("video_id", None)
    if vid is not None:
        cand = os.path.join(root, videos_subdir, f"{vid}.mp4")
        if os.path.exists(cand):
            return cand
        hits = glob.glob(os.path.join(root, videos_subdir, "**", f"{vid}.mp4"), recursive=True)
        if hits:
            return hits[0]
    return os.path.join(root, videos_subdir, "MISSING.mp4")


def _tid_to_cat(ann: dict) -> Dict[int, str]:
    objs = ann.get("subject/objects", []) or ann.get("subjects/objects", []) or ann.get("subject_objects", [])
    out: Dict[int, str] = {}
    for o in objs:
        try:
            out[int(o["tid"])] = str(o["category"])
        except Exception:
            continue
    return out


def _sample_frame_indices(frame_count: int, T_out: int) -> List[int]:
    frame_count = max(int(frame_count), 1)
    if T_out <= 1:
        return [0]
    return [int(round(i * (frame_count - 1) / (T_out - 1))) for i in range(T_out)]


def _relation_active_at(rel_inst: dict, t: int) -> bool:
    b = int(rel_inst.get("begin_fid", 0))
    e = int(rel_inst.get("end_fid", 0))
    return (b <= t) and (t < e)


# -----------------------------
# concept mining
# -----------------------------

def mine_vidor_concepts(train_ann_paths: List[str],
                        topk_obj: int,
                        topk_rel: int,
                        keep_predicates: Optional[Set[str]],
                        min_rel_support: int) -> Tuple[List[str], List[str]]:
    from collections import Counter
    c_obj = Counter(); c_rel = Counter()

    for pth in tqdm(train_ann_paths, desc="[Mine] scan TRAIN annotations"):
        try:
            ann = _read_json(pth)
        except Exception:
            continue
        tid2cat = _tid_to_cat(ann)
        for _, cat in tid2cat.items():
            c_obj[cat] += 1
        rels = ann.get("relation_instances", []) or []
        for r in rels:
            try:
                st = int(r["subject_tid"])
                ot = int(r["object_tid"])
                pred = str(r["predicate"])
            except Exception:
                continue
            if keep_predicates is not None and pred not in keep_predicates:
                continue
            subj = tid2cat.get(st); obj = tid2cat.get(ot)
            if subj is None or obj is None:
                continue
            c_rel[f"{subj}-{pred}-{obj}"] += 1

    obj_keep = [k for k, _ in c_obj.most_common(topk_obj)]
    rel_keep = [k for k, c in c_rel.most_common(max(topk_rel * 3, topk_rel)) if c >= min_rel_support][:topk_rel]
    return obj_keep, rel_keep


# -----------------------------
# Dataset
# -----------------------------

class VidORDataset(Dataset):
    def __init__(self, ann_paths: List[str], root: str, videos_subdir: str,
                 concept_names: List[str], frames: int, transform):
        self.ann_paths = ann_paths
        self.root = root
        self.videos_subdir = videos_subdir
        self.concept_names = concept_names
        self.name_to_idx = {nm: i for i, nm in enumerate(concept_names)}
        self.frames = frames
        self.transform = transform
        self.ids = [os.path.splitext(os.path.basename(p))[0] for p in ann_paths]

    def __len__(self):
        return len(self.ann_paths)

    def _read_clip_cv2(self, video_path: str, frame_count_hint: int) -> List[Image.Image]:
        if cv2 is None:
            raise RuntimeError("OpenCV not available. Install opencv-python.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            # hard fallback: return all-black
            return [Image.new("RGB", (224, 224), (0, 0, 0)) for _ in range(self.frames)]

        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n <= 0:
            n = int(frame_count_hint or 0)
        n = max(n, 1)

        idxs = _sample_frame_indices(n, self.frames)  # length=self.frames, may contain repeats
        idx_last = max(idxs)
        idx_set = set(idxs)

        # Decode sequentially up to idx_last (much faster/robust than cap.set per frame)
        frames_by_idx: Dict[int, Optional[Image.Image]] = {}
        last_good: Optional[Image.Image] = None

        t = 0
        while t <= idx_last:
            ok = cap.grab()
            if not ok:
                break
            if t in idx_set:
                ok2, frame = cap.retrieve()
                if (not ok2) or (frame is None) or (getattr(frame, "size", 0) == 0):
                    frames_by_idx[t] = None
                else:
                    try:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        im = Image.fromarray(frame)
                        frames_by_idx[t] = im
                        last_good = im
                    except Exception:
                        frames_by_idx[t] = None
            t += 1

        cap.release()

        # Build the final list in the exact requested order, padding missing with last_good or black
        out: List[Image.Image] = []
        black = Image.new("RGB", (224, 224), (0, 0, 0))
        for idx in idxs:
            im = frames_by_idx.get(idx, None)
            if im is None:
                im = last_good.copy() if last_good is not None else black
            out.append(im)

        # Guarantee length exactly self.frames
        if len(out) < self.frames:
            pad = out[-1] if out else black
            out.extend([pad.copy() for _ in range(self.frames - len(out))])
        elif len(out) > self.frames:
            out = out[:self.frames]

        return out

    def __getitem__(self, i):
        ann = _read_json(self.ann_paths[i])
        vid_path = _video_path_from_ann(self.root, self.videos_subdir, ann)

        frame_count = int(ann.get("frame_count", 0))
        sampled_ts = _sample_frame_indices(frame_count if frame_count > 0 else 1, self.frames)

        frames_pil = self._read_clip_cv2(vid_path, frame_count)
        clip = torch.stack([self.transform(im) for im in frames_pil], dim=0)  # (T,3,H,W)

        y = torch.zeros(len(self.concept_names), dtype=torch.long)
        tid2cat = _tid_to_cat(ann)

        traj = ann.get("trajectories", []) or []
        obj_present: Set[str] = set()
        for t in sampled_ts:
            if t < 0 or t >= len(traj):
                continue
            for box in traj[t]:
                try:
                    tid = int(box["tid"])
                except Exception:
                    continue
                cat = tid2cat.get(tid)
                if cat is not None:
                    obj_present.add(cat)

        rel_present: Set[str] = set()
        rels = ann.get("relation_instances", []) or []
        for r in rels:
            try:
                st = int(r["subject_tid"])
                ot = int(r["object_tid"])
                pred = str(r["predicate"])
            except Exception:
                continue
            subj = tid2cat.get(st); obj = tid2cat.get(ot)
            if subj is None or obj is None:
                continue
            if any(_relation_active_at(r, t) for t in sampled_ts):
                rel_present.add(f"{subj}-{pred}-{obj}")
                # enforce structural consistency under GT
                obj_present.add(subj); obj_present.add(obj)

        for cat in obj_present:
            j = self.name_to_idx.get(f"obj:{cat}")
            if j is not None:
                y[j] = 1
        for rel in rel_present:
            j = self.name_to_idx.get(f"rel:{rel}")
            if j is not None:
                y[j] = 1

        return clip, y


# -----------------------------
# Label-only extraction (no video decode) for mining/support stats
# -----------------------------
def labels_from_ann_path(ann_path: str,
                         name_to_idx: Dict[str, int],
                         K: int,
                         frames: int) -> torch.Tensor:
    """Return multi-hot label vector y (K,) from a VidOR annotation JSON WITHOUT reading video frames."""
    ann = _read_json(ann_path)
    frame_count = int(ann.get("frame_count", 0))
    sampled_ts = _sample_frame_indices(frame_count if frame_count > 0 else 1, frames)

    y = torch.zeros(K, dtype=torch.long)
    tid2cat = _tid_to_cat(ann)

    traj = ann.get("trajectories", []) or []
    obj_present: Set[str] = set()
    for t in sampled_ts:
        if t < 0 or t >= len(traj):
            continue
        for box in traj[t]:
            try:
                tid = int(box["tid"])
            except Exception:
                continue
            cat = tid2cat.get(tid)
            if cat is not None:
                obj_present.add(cat)

    rel_present: Set[str] = set()
    rels = ann.get("relation_instances", []) or []
    for r in rels:
        try:
            st = int(r["subject_tid"])
            ot = int(r["object_tid"])
            pred = str(r["predicate"])
        except Exception:
            continue
        subj = tid2cat.get(st); obj = tid2cat.get(ot)
        if subj is None or obj is None:
            continue
        if any(_relation_active_at(r, t) for t in sampled_ts):
            rel_present.add(f"{subj}-{pred}-{obj}")
            obj_present.add(subj); obj_present.add(obj)

    for cat in obj_present:
        j = name_to_idx.get(f"obj:{cat}")
        if j is not None:
            y[j] = 1
    for rel in rel_present:
        j = name_to_idx.get(f"rel:{rel}")
        if j is not None:
            y[j] = 1
    return y


def prune_rules_by_support(rule_items: List[Tuple[str, dgl.DGLGraph]],
                           ann_paths: List[str],
                           concept_names: List[str],
                           frames: int,
                           support_scan: int,
                           min_true: int,
                           min_false: int,
                           max_keep: int,
                           kind: str) -> List[Tuple[str, dgl.DGLGraph]]:
    """Prune rules based on how often they are TRUE/FALSE under GT labels on TRAIN."""
    if support_scan <= 0 or (min_true <= 0 and min_false <= 0) or len(rule_items) == 0:
        return rule_items

    name_to_idx = {nm: i for i, nm in enumerate(concept_names)}
    K = len(concept_names)
    graphs = [g for _, g in rule_items]
    roots = [int(torch.nonzero(g.out_degrees() == 0, as_tuple=False).flatten()[0].item()) for g in graphs]

    true_cnt = [0] * len(rule_items)
    false_cnt = [0] * len(rule_items)
    scanned = 0

    for ap in ann_paths:
        if scanned >= support_scan:
            break
        try:
            y = labels_from_ann_path(ap, name_to_idx=name_to_idx, K=K, frames=frames)
        except Exception:
            continue

        for r_idx, (g, root) in enumerate(zip(graphs, roots)):
            tv = propagate_truth_values_hard_from_leaf_labels(g, y)
            troot = int(tv[root].item())
            true_cnt[r_idx] += troot
            false_cnt[r_idx] += (1 - troot)

        scanned += 1

    keep = [i for i in range(len(rule_items)) if (true_cnt[i] >= min_true) and (false_cnt[i] >= min_false)]
    keep.sort(key=lambda i: (false_cnt[i], true_cnt[i]), reverse=True)

    kept_all = len(keep)
    if max_keep is not None and max_keep > 0:
        keep = keep[:max_keep]

    kept = [rule_items[i] for i in keep]
    print(f"[Support] {kind}: scanned={scanned} | kept={len(kept)}/{len(rule_items)} "
          f"(passed={kept_all}) (min_true={min_true}, min_false={min_false})")
    return kept


# -----------------------------
# Leaf bank
# -----------------------------

class VidORLeafBank(nn.Module):
    def __init__(self, K: int, feat_dim: int = 256, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            import torchvision.models as M
            m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
            self.frame_backbone = nn.Sequential(*(list(m.children())[:-1]))
            in_ch = 512
        else:
            self.frame_backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64, 96, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(96, 128, 3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            in_ch = 128
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, feat_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(K)])
        self.temp_scaler: Optional[TempScaler] = None

    def encoder(self, clips: torch.Tensor) -> torch.Tensor:
        if clips.dim() == 4:
            clips = clips.unsqueeze(0)
        B, T, C, H, W = clips.shape
        x = clips.view(B * T, C, H, W)
        z = self.frame_backbone(x)
        z = self.proj(z)
        return z.view(B, T, -1).mean(dim=1)

    def forward_logits(self, clips: torch.Tensor) -> torch.Tensor:
        z = self.encoder(clips)
        return torch.cat([h(z) for h in self.heads], dim=1)

    def forward_probs(self, clips: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(clips)
        if self.temp_scaler is not None:
            logits = self.temp_scaler(logits)
        return torch.sigmoid(logits)


# -----------------------------
# Leaf training + calibration
# -----------------------------

def train_leaf_bank(bank, loader, epochs: int, lr: float, device: str,
                    weight_decay: float = 0.0, pos_weight: Optional[torch.Tensor] = None):
    bank = bank.to(device)
    opt = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bank.train()
    for ep in trange(1, epochs + 1, desc="[LeafBank] epochs", leave=True):
        total = 0.0; n = 0
        for clips, labels in tqdm(loader, desc=f"[LeafBank] epoch {ep}/{epochs}", leave=False):
            clips = clips.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            loss = crit(bank.forward_logits(clips), labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
        print(f"[LeafBank] ep={ep} avg_loss={total/max(n,1):.4f}")


class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.to(logits.device).clamp_min(1e-3)


def fit_temperature(bank, loader: DataLoader, device: str) -> TempScaler:
    bank.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for clips, y in loader:
            clips = clips.to(device, non_blocking=True)
            all_logits.append(bank.forward_logits(clips).detach().cpu())
            all_labels.append(y.float())
    logits_cpu = torch.cat(all_logits, dim=0)
    labels_cpu = torch.cat(all_labels, dim=0)
    scaler = TempScaler()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    crit = nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = crit(scaler(logits_cpu), labels_cpu)
        loss.backward()
        return loss

    opt.step(closure)
    return scaler


# -----------------------------
# Rule DSL + builders
# -----------------------------

OPC = {"IFF": 1, "IMPLIES": 2, "AND": 3, "OR": 4}

def leaf(cid: int):     return ("leaf", cid)

def NOT(expr):          return ("not", expr)

def AND(a, b):           return ("AND", a, b)

def OR(a, b):            return ("OR", a, b)

def IFF(a, b):           return ("IFF", a, b)

def IMPLIES(l, r):       return ("IMPLIES", l, r)


def _add_expr(expr, nodes, edges):
    kind = expr[0]
    if kind == "leaf":
        cid = int(expr[1])
        nid = len(nodes)
        nodes.append({"mask": 1, "y": 0, "x": cid})
        return nid
    if kind == "not":
        return _add_expr(expr[1], nodes, edges)

    op_name, a, b = expr
    left = _add_expr(a, nodes, edges)
    right = _add_expr(b, nodes, edges)
    nid = len(nodes)
    nodes.append({"mask": 0, "y": OPC[op_name], "x": 0})

    def _is_not(e):
        return isinstance(e, tuple) and len(e) > 0 and e[0] == "not"

    neg_l = -1 if _is_not(a) else +1
    neg_r = -1 if _is_not(b) else +1

    edges.append((left, nid, neg_l, 0))
    edges.append((right, nid, neg_r, 1))
    return nid


def build_graph_from_expr(expr) -> dgl.DGLGraph:
    nodes, edges = [], []
    _ = _add_expr(expr, nodes, edges)
    g = dgl.graph(([], []), num_nodes=len(nodes))
    g.ndata["mask"] = torch.tensor([n["mask"] for n in nodes], dtype=torch.long)
    g.ndata["y"] = torch.tensor([n["y"] for n in nodes], dtype=torch.long)
    g.ndata["x"] = torch.tensor([n["x"] for n in nodes], dtype=torch.long)
    if edges:
        src = torch.tensor([s for (s, _, _, _) in edges], dtype=torch.long)
        dst = torch.tensor([d for (_, d, _, _) in edges], dtype=torch.long)
        g.add_edges(src, dst)
        g.edata["neg"] = torch.tensor([neg for (_, _, neg, _) in edges], dtype=torch.long)
        g.edata["pos"] = torch.tensor([pos for (_, _, _, pos) in edges], dtype=torch.long)
    return g


def build_graph_for_simple_forward(ante_cid_1based: int, cons_cid_1based: int) -> dgl.DGLGraph:
    g = dgl.graph(([], []), num_nodes=3)
    g.ndata["mask"] = torch.tensor([1, 1, 0], dtype=torch.long)
    g.ndata["y"] = torch.tensor([0, 0, 2], dtype=torch.long)  # IMPLIES
    g.ndata["x"] = torch.tensor([ante_cid_1based, cons_cid_1based, 0], dtype=torch.long)
    g.add_edges(torch.tensor([0, 1]), torch.tensor([2, 2]))
    g.edata["neg"] = torch.tensor([+1, +1], dtype=torch.long)
    g.edata["pos"] = torch.tensor([0, 1], dtype=torch.long)
    return g


def build_graph_for_simple_inverted(ante_cid_1based: int, cons_cid_1based: int) -> dgl.DGLGraph:
    return build_graph_for_simple_forward(cons_cid_1based, ante_cid_1based)


# -----------------------------
# Mining (direction-aware)
# -----------------------------

def tally_label_counts(loader: DataLoader, K: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    cnt_A = torch.zeros(K, dtype=torch.long)
    cnt_AB = torch.zeros(K, K, dtype=torch.long)
    N = 0
    for _x, y in tqdm(loader, desc="[Mine] tally (TRAIN)", leave=False):
        y = y.to(torch.long)
        B = y.size(0)
        N += B
        cnt_A += y.sum(dim=0).cpu()
        yy = y.to(torch.int64)
        cnt_AB += (yy.t() @ yy).cpu()
    return cnt_A, cnt_AB, N


def mine_pairs_auto(cnt_A: torch.Tensor,
                    cnt_AB: torch.Tensor,
                    N: int,
                    min_support_count: int,
                    min_conf_fwd: float,
                    min_conf_rev: float,
                    min_lift: float,
                    mine_direction: str,
                    min_conf_margin: float) -> List[dict]:
    N = max(int(N), 1)
    pA = cnt_A.float() / float(N)
    K = cnt_A.numel()
    out: List[dict] = []

    for i in range(K):
        nA = int(cnt_A[i].item())
        if nA <= 0:
            continue
        for j in range(K):
            if i == j:
                continue
            nAB = int(cnt_AB[i, j].item())
            if nAB < min_support_count:
                continue

            conf_fwd = nAB / max(nA, 1)
            lift_fwd = conf_fwd / max(float(pA[j].item()), 1e-9)

            nB = int(cnt_A[j].item())
            conf_rev = nAB / max(nB, 1)
            lift_rev = conf_rev / max(float(pA[i].item()), 1e-9)

            choose = None
            if mine_direction == "forward":
                if conf_fwd >= min_conf_fwd and lift_fwd >= min_lift:
                    choose = ("fwd", i, j, conf_fwd, lift_fwd)
            elif mine_direction == "invert":
                if conf_rev >= min_conf_rev and lift_rev >= min_lift:
                    choose = ("rev", j, i, conf_rev, lift_rev)
            else:
                ok_fwd = (conf_fwd >= min_conf_fwd and lift_fwd >= min_lift)
                ok_rev = (conf_rev >= min_conf_rev and lift_rev >= min_lift)
                if ok_fwd and (conf_fwd >= conf_rev + min_conf_margin):
                    choose = ("fwd", i, j, conf_fwd, lift_fwd)
                elif ok_rev and (conf_rev >= conf_fwd + min_conf_margin):
                    choose = ("rev", j, i, conf_rev, lift_rev)

            if choose is None:
                continue

            dir_tag, a, b, conf, lift = choose
            out.append({
                "ante": int(a),
                "cons": int(b),
                "conf": float(conf),
                "lift": float(lift),
                "support": int(nAB),
                "dir": dir_tag,
            })

    out.sort(key=lambda r: (r["conf"], r["lift"], r["support"]), reverse=True)
    return out


def count_triples(loader: DataLoader, a: int, b1: int, b2: int) -> int:
    n = 0
    for _x, y in loader:
        y = y.to(torch.long)
        n += int(((y[:, a] == 1) & (y[:, b1] == 1) & (y[:, b2] == 1)).sum().item())
    return n


# -----------------------------
# Truths + anomaly scoring
# -----------------------------

class LearnedAggregator(nn.Module):
    def __init__(self, R: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(R))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        x = torch.logit(torch.clamp(probs, 1e-6, 1 - 1e-6))
        return torch.sigmoid(x @ self.w + self.b)


@torch.no_grad()
def compute_rule_truths_batch_from_graphs(labels: torch.Tensor, rule_graphs: List[dgl.DGLGraph]) -> torch.Tensor:
    B = labels.size(0); R = len(rule_graphs)
    out = torch.zeros(B, R, dtype=torch.long)
    for r_idx, g in enumerate(rule_graphs):
        root = int(torch.nonzero(g.out_degrees() == 0, as_tuple=False).flatten()[0].item())
        for b in range(B):
            tv = propagate_truth_values_hard_from_leaf_labels(g, labels[b])
            out[b, r_idx] = int(tv[root].item())
    return out



def compute_rule_support_counts(ds: Dataset, rule_graphs: List[dgl.DGLGraph],
                                scan_limit: int, batch_size: int, num_workers: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """
    Return (true_counts, false_counts, scanned_N) under GT labels, scanning up to scan_limit samples.
    """
    R = len(rule_graphs)
    tcnt = torch.zeros(R, dtype=torch.long)
    fcnt = torch.zeros(R, dtype=torch.long)
    seen = 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    for _x, y in tqdm(loader, desc="[Support] scan", leave=False):
        truths = compute_rule_truths_batch_from_graphs(y, rule_graphs)
        tcnt += truths.sum(dim=0).cpu()
        fcnt += (1 - truths).sum(dim=0).cpu()
        seen += int(y.size(0))
        if scan_limit is not None and scan_limit > 0 and seen >= scan_limit:
            break
    return tcnt, fcnt, seen

def anomaly_score_batch(images: torch.Tensor,
                        trainers,
                        bank: VidORLeafBank,
                        device: str,
                        rule_graphs: List[dgl.DGLGraph],
                        agg: str = "min",
                        learned: Optional[LearnedAggregator] = None,
                        gate_tau: float = 0.0) -> torch.Tensor:
    images = images.to(device, non_blocking=True)
    leaf_probs = bank.forward_probs(images)  # (B,K)

    per_rule = []
    for r_idx, (_, trainer) in enumerate(trainers):
        if getattr(trainer, "_enc_fprint", None) is None:
            trainer._enc_fprint = encoder_fingerprint(bank)

        p_rule = torch.stack(
            [torch.as_tensor(trainer.predict_root(images[b], bank), device=device) for b in range(images.size(0))],
            dim=0,
        )
        p_rule = torch.clamp(p_rule, 1e-6, 1.0)

        g = rule_graphs[r_idx]
        root = int(torch.nonzero(g.out_degrees() == 0, as_tuple=False).flatten()[0].item())
        src, dst, eids = g.in_edges(root, form="all")
        left_node = int(src[0].item())
        cid = int(g.ndata["x"][left_node].item())  # 1-based leaf id or 0

        if cid > 0:
            p_ante_raw = torch.clamp(leaf_probs[:, cid - 1], 0.0, 1.0)
            if gate_tau > 0:
                p_ante = torch.clamp(p_ante_raw - gate_tau, min=0.0) / max(1e-6, (1.0 - gate_tau))
            else:
                p_ante = p_ante_raw
        else:
            p_ante = torch.ones_like(p_rule)

        s_r = p_ante * (1.0 - p_rule)
        per_rule.append(s_r)

    S = torch.stack(per_rule, dim=1)  # (B,R)
    eps = 1e-9
    if agg == "min":
        return S.max(dim=1).values
    if agg == "mean":
        return S.mean(dim=1)
    if agg == "geo":
        return torch.exp(torch.log(S + eps).mean(dim=1))
    if agg == "learned" and learned is not None:
        return 1.0 - learned(1.0 - S)
    return S.max(dim=1).values


# -----------------------------
# Structural rules
# -----------------------------

def build_structural_rules(concept_names: List[str]) -> List[Tuple[str, dgl.DGLGraph]]:
    name_to_cid = {nm: i + 1 for i, nm in enumerate(concept_names)}
    out: List[Tuple[str, dgl.DGLGraph]] = []
    for nm in concept_names:
        if not nm.startswith("rel:"):
            continue
        trip = nm[len("rel:"):]
        parts = trip.split("-")
        if len(parts) < 3:
            continue
        subj = parts[0]
        obj = "-".join(parts[2:])
        rel_cid = name_to_cid[nm]
        subj_nm = f"obj:{subj}"
        obj_nm = f"obj:{obj}"
        if subj_nm in name_to_cid:
            out.append((f"{nm} -> {subj_nm} [struct]", build_graph_for_simple_forward(rel_cid, name_to_cid[subj_nm])))
        if obj_nm in name_to_cid:
            out.append((f"{nm} -> {obj_nm} [struct]", build_graph_for_simple_forward(rel_cid, name_to_cid[obj_nm])))
    return out


# -----------------------------
# Main
# -----------------------------

def main():
    args = build_args()
    set_seed(args.seed)

    os.makedirs(os.path.dirname(args.leaf_ckpt), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)
    if args.agg == "learned":
        os.makedirs(os.path.dirname(args.agg_ckpt), exist_ok=True)
    if args.calib_train or (args.eval_only and os.path.exists(args.calib_ckpt)):
        os.makedirs(os.path.dirname(args.calib_ckpt), exist_ok=True)

    device = (
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else
        ("cuda" if args.device == "cuda" else "cpu")
    )
    print(f"[Setup] Device: {device}")

    train_paths = list_ann_paths(args.vidor_root, args.ann_subdir, args.split_train)
    val_paths = list_ann_paths(args.vidor_root, args.ann_subdir, args.split_val)
    if len(train_paths) == 0 or len(val_paths) == 0:
        raise FileNotFoundError("Could not find annotations under annotations/{train,val}.")

    max_simple_raw = max(args.max_rules_simple * 10, args.max_rules_simple)   # e.g. 200 if keep=20

    if args.train_frac < 1.0:
        n_keep = max(1, int(len(train_paths) * args.train_frac))
        train_paths = train_paths[:n_keep]
        print(f"[Data] TRAIN subsample: {n_keep}")

    keep_preds = None
    if args.keep_predicates.strip():
        keep_preds = set([p.strip() for p in args.keep_predicates.split(",") if p.strip()])
        print(f"[Concepts] predicate whitelist size={len(keep_preds)}")

    obj_keep, rel_keep = mine_vidor_concepts(
        train_paths,
        topk_obj=args.topk_obj,
        topk_rel=args.topk_rel,
        keep_predicates=keep_preds,
        min_rel_support=args.min_rel_support,
    )
    concept_names = [f"obj:{c}" for c in obj_keep] + [f"rel:{r}" for r in rel_keep]
    name_to_idx = {nm: i for i, nm in enumerate(concept_names)}
    K = len(concept_names)
    idx_to_name = {i: nm for i, nm in enumerate(concept_names)}
    print(f"[Concepts] K={K} (obj={len(obj_keep)}, rel={len(rel_keep)})")

    tf_train = make_transforms("train", args.resize, use_aug=args.augment)
    tf_eval = make_transforms("val", args.resize, use_aug=False)

    train_ds = VidORDataset(train_paths, args.vidor_root, args.videos_subdir, concept_names, args.frames, tf_train)
    val_ds = VidORDataset(val_paths, args.vidor_root, args.videos_subdir, concept_names, args.frames, tf_eval)

    train_loader = None
    if not args.eval_only:
        train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # leaf
    bank = VidORLeafBank(K=K, feat_dim=args.feat_dim, backbone=args.backbone).to(device)

    if args.eval_only:
        if not os.path.exists(args.leaf_ckpt):
            raise FileNotFoundError(f"--eval_only but missing leaf ckpt: {args.leaf_ckpt}")
        bank.load_state_dict(torch.load(args.leaf_ckpt, map_location="cpu"), strict=True)
        bank.eval()
        print(f"[LeafBank] Loaded {args.leaf_ckpt}")
        try:
            evaluate_leaf_bank(bank, val_loader, idx_to_name, device=device, threshold=0.5)
        except Exception as e:
            print(f"[LeafEval] failed: {type(e).__name__}: {e}")
    else:
        # prevalence pos_weight
        pos_weight = None
        if not args.no_stats:
            cnt = torch.zeros(K, dtype=torch.long)
            for _x, y in tqdm(DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=args.num_workers),
                              desc="[Data] tally (TRAIN)", leave=False):
                cnt += y.sum(dim=0)
            total = len(train_ds)
            pos_weight = torch.tensor(
                [max(total - int(cnt[i].item()), 1) / max(int(cnt[i].item()), 1) for i in range(K)],
                dtype=torch.float32,
                device=device,
            )

        if args.epochs_leaf > 0:
            print("[LeafBank] Training …")
            train_leaf_bank(bank, train_loader, epochs=args.epochs_leaf, lr=args.lr_leaf,
                            device=device, weight_decay=args.weight_decay, pos_weight=pos_weight)
            torch.save(bank.state_dict(), args.leaf_ckpt)
            print(f"[LeafBank] Saved → {args.leaf_ckpt}")
            try:
                evaluate_leaf_bank(bank, val_loader, idx_to_name, device=device, threshold=0.5)
            except Exception as e:
                print(f"[LeafEval] failed: {type(e).__name__}: {e}")

    # calibration
    if args.calib_train and (not args.eval_only):
        scaler = fit_temperature(bank, DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers), device)
        torch.save(scaler.state_dict(), args.calib_ckpt)
        print(f"[Calib] Saved scaler → {args.calib_ckpt}")
    if os.path.exists(args.calib_ckpt):
        scaler = TempScaler()
        scaler.load_state_dict(torch.load(args.calib_ckpt, map_location="cpu"))
        bank.temp_scaler = scaler.to(device)
        print(f"[Calib] Loaded scaler {args.calib_ckpt}")

    # ---------------- Rules ----------------
    rules_simple: List[Tuple[str, dgl.DGLGraph]] = []
    compound_items: List[Tuple[str, dgl.DGLGraph]] = []
    seen_sigs: Set[str] = set()

    # structural
    # for name, g in build_structural_rules(concept_names):
    #     if len(rules_simple) >= max_simple_raw:
    #         break
    #     sig = _rule_sig(g)
    #     if sig in seen_sigs:
    #         continue
    #     seen_sigs.add(sig)
    #     rules_simple.append((name, g))

    # mine pairs
    mined: List[dict] = []
    cnt_A = cnt_AB = None
    if args.mine_pairs:
        mine_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)
        cnt_A, cnt_AB, N = tally_label_counts(mine_loader, K)
        mined = mine_pairs_auto(cnt_A, cnt_AB, N,
                                min_support_count=args.min_support_count,
                                min_conf_fwd=args.min_conf_fwd,
                                min_conf_rev=args.min_conf_rev,
                                min_lift=args.min_lift,
                                mine_direction=args.mine_direction,
                                min_conf_margin=args.min_conf_margin)
        print(f"[Mining] mined pairs: {len(mined)}")

        # simple mined
        for r in mined:
            if len(rules_simple) >= max_simple_raw:
                break
            a_i, c_i = r["ante"], r["cons"]
            nmA, nmC = idx_to_name[a_i], idx_to_name[c_i]
            # Drop trivial cross-type implications (often tautological given label construction)
            if nmA.startswith("rel:") and nmC.startswith("obj:"):
                continue
            if nmA.startswith("obj:") and nmC.startswith("rel:"):
                continue
            tag = "[mined rev]" if r["dir"] == "rev" else "[mined]"   # 'rev' = selected via reverse-confidence, NOT graph inversion
            g = build_graph_for_simple_forward(a_i + 1, c_i + 1)      # ALWAYS encode ante -> cons
            sig = _rule_sig(g)
            if sig in seen_sigs:
                continue

            seen_sigs.add(sig)
            nm = f"{nmA} -> {nmC} {tag} (sup={r['support']}, conf={r['conf']:.3f}, lift={r['lift']:.2f})"
            rules_simple.append((nm, g))

        # compound (A1 AND A2)->B from mined (as in OI)
        from collections import defaultdict
        ants_for_cons: Dict[int, List[dict]] = defaultdict(list)
        for r in mined:
            ants_for_cons[r["cons"]].append(r)

        count_comp = 0
        for cons_i, lst in ants_for_cons.items():
            if count_comp >= args.max_rules_compound:
                break
            best_for_ante: Dict[int, dict] = {}
            for r in lst:
                a_i = r["ante"]
                if a_i not in best_for_ante or r["conf"] > best_for_ante[a_i]["conf"]:
                    best_for_ante[a_i] = r
            ants = sorted(best_for_ante.values(), key=lambda x: (x["conf"], x["lift"], x["support"]), reverse=True)

            pairs = []
            used = set()
            for i0 in range(len(ants)):
                a1 = ants[i0]["ante"]
                if a1 in used:
                    continue
                for j0 in range(i0 + 1, len(ants)):
                    a2 = ants[j0]["ante"]
                    if a2 in used or a2 == a1:
                        continue
                    pairs.append((a1, a2))
                    used.add(a1); used.add(a2)
                    if len(pairs) >= args.per_B_pair_limit:
                        break
                if len(pairs) >= args.per_B_pair_limit:
                    break


            use_rev = any(r["dir"] == "rev" for r in lst)
            nmC = idx_to_name[cons_i]

            # Compute overlap only if we are going to build inverted compounds
            ov_map = {}
            if use_rev and args.inv_overlap_min > 0 and len(pairs) > 0:
                scan_n = args.inv_overlap_scan
                if scan_n <= 0:
                    scan_n = len(train_paths)
                scan_n = min(scan_n, len(train_paths))
                ov_map = inv_overlap_for_pairs(train_paths, name_to_idx, args.frames, cons_i, pairs, scan_n)

            for a1, a2 in pairs:
                nmA1, nmA2 = idx_to_name[a1], idx_to_name[a2]

                if use_rev:
                    ov = ov_map.get((a1, a2), 0.0)
                    if args.inv_overlap_min > 0 and ov < args.inv_overlap_min:
                        continue

                    expr = IMPLIES(leaf(cons_i + 1), AND(leaf(a1 + 1), leaf(a2 + 1)))
                    nm = f"{nmC} -> ({nmA1} AND {nmA2}) [mined inv] (ov={ov:.2f})"
                else:
                    expr = IMPLIES(AND(leaf(a1 + 1), leaf(a2 + 1)), leaf(cons_i + 1))
                    nm = f"({nmA1} AND {nmA2}) -> {nmC} [mined]"

                g = build_graph_from_expr(expr)
                sig = _rule_sig(g)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                compound_items.append((nm, g))
                count_comp += 1
                if count_comp >= args.max_rules_compound:
                    break

        # OR consequents
        if args.add_or_compounds and count_comp < args.max_rules_compound:
            cons_for_ante: Dict[int, List[dict]] = defaultdict(list)
            for r in mined:
                cons_for_ante[r["ante"]].append(r)
            mine_loader2 = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=args.num_workers)

            for a_i, lst in cons_for_ante.items():
                if count_comp >= args.max_rules_compound:
                    break
                lst_sorted = sorted(lst, key=lambda x: (x["conf"], x["lift"], x["support"]), reverse=True)
                top = lst_sorted[:max(2, args.or_topk_per_A)]
                if len(top) < 2:
                    continue
                b1 = top[0]["cons"]
                for k in range(1, len(top)):
                    b2 = top[k]["cons"]
                    nAB1 = int(cnt_AB[a_i, b1].item())
                    nAB2 = int(cnt_AB[a_i, b2].item())
                    if min(nAB1, nAB2) < args.min_support_count:
                        continue
                    nAB1B2 = count_triples(mine_loader2, a_i, b1, b2)
                    overlap = nAB1B2 / max(min(nAB1, nAB2), 1)
                    if overlap > args.or_overlap_max:
                        continue
                    nmA, nmB1, nmB2 = idx_to_name[a_i], idx_to_name[b1], idx_to_name[b2]
                    expr = IMPLIES(leaf(a_i + 1), OR(leaf(b1 + 1), leaf(b2 + 1)))
                    nm = f"{nmA} -> ({nmB1} OR {nmB2}) [mined OR alt] (ov={overlap:.2f})"
                    g = build_graph_from_expr(expr)
                    sig = _rule_sig(g)
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    compound_items.append((nm, g))
                    count_comp += 1
                    break

    print(f"[Rules] Simple:   {len(rules_simple)}")
    print(f"[Rules] Compound: {len(compound_items)}")

    # ---------------- Support pruning (like OpenImages script) ----------------
    # Drop tautological / useless rules (e.g. never violated) and cap by max_rules_* budgets.
    if args.support_scan > 0:
        rules_simple = prune_rules_by_support(
            rules_simple, train_paths, concept_names,
            frames=args.frames,
            support_scan=args.support_scan,
            min_true=args.min_true,
            min_false=args.min_false,
            max_keep=args.max_rules_simple,
            kind="simple"
        )
        compound_items = prune_rules_by_support(
            compound_items, train_paths, concept_names,
            frames=args.frames,
            support_scan=args.support_scan,
            min_true=args.min_true,
            min_false=args.min_false,
            max_keep=args.max_rules_compound,
            kind="compound"
        )
        print(f"[Rules] After support pruning: Simple={len(rules_simple)} | Compound={len(compound_items)}")

    # ---------------- Trainers ----------------
    trainers = []
    rule_graphs: List[dgl.DGLGraph] = []

    def _add_trainer(name: str, g: dgl.DGLGraph, cache_tag: str):
        rule_cache = os.path.join(args.cache_dir, cache_tag)
        cache = SubtreeCache(CacheConfig(root_dir=rule_cache))
        trn = LevelwiseTrainer(g, N_concepts=K, cache=cache, device=device,
                               cache_leafroot_only=False, lineage_aware=True)
        if not args.eval_only:
            kwargs = dict(
                dataset=train_loader,
                leaf_bank=bank,
                epochs_per_level=args.epochs_level,
                lr=args.lr_level,
                use_soft_leaves=True,
                verbose=True,
                use_tqdm=True,
                negatives=args.negatives,
            )
            if args.train_missing_only:
                try:
                    trn.train(**kwargs, train_missing_only=True)  # type: ignore[arg-type]
                except TypeError:
                    print("[Trainer] train_missing_only not supported by your trainer; training all nodes.")
                    trn.train(**kwargs)
            else:
                trn.train(**kwargs)
        trainers.append((name, trn))
        rule_graphs.append(g)

    for j, (name, g) in enumerate(rules_simple, start=1):
        _add_trainer(name, g, f"simple_{j:03d}")
    for j, (name, g) in enumerate(compound_items, start=1):
        _add_trainer(name, g, f"compound_{j:03d}")

    # manifest
    manifest = {
        "vidor_root": os.path.abspath(args.vidor_root),
        "splits": {"train": args.split_train, "val": args.split_val},
        "concepts": concept_names,
        "n_rules_simple": len(rules_simple),
        "n_rules_compound": len(compound_items),
        "negatives": args.negatives,
        "frames": args.frames,
        "resize": args.resize,
        "agg": args.agg,
        "gate_tau": args.gate_tau,
        "add_or_compounds": args.add_or_compounds,
    }
    with open(os.path.join(args.run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # learned aggregator
    learned_agg: Optional[LearnedAggregator] = None
    if args.agg == "learned":
        if args.eval_only and os.path.exists(args.agg_ckpt):
            learned_agg = LearnedAggregator(len(trainers))
            learned_agg.load_state_dict(torch.load(args.agg_ckpt, map_location="cpu"))
            learned_agg.to(device).eval()
            print(f"[Agg] Loaded {args.agg_ckpt}")
        elif not args.eval_only:
            X_list, Y_list = [], []
            fit_loader = DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
            for clips, labels in tqdm(fit_loader, desc="[Agg] collect (TRAIN)", leave=False):
                clips = clips.to(device, non_blocking=True)
                probs = []
                for _, trn in trainers:
                    p = torch.stack([torch.as_tensor(trn.predict_root(clips[b], bank), device=device)
                                     for b in range(clips.size(0))], dim=0)
                    probs.append(torch.clamp(p, 1e-6, 1.0))
                Pp = torch.stack(probs, dim=1).detach().cpu()  # (B,R)
                truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)
                is_normal = truths.min(dim=1).values
                y = 1 - is_normal.float()
                X_list.append(Pp); Y_list.append(y.detach().cpu())

            X = torch.cat(X_list, dim=0)
            Y = torch.cat(Y_list, dim=0)
            learned_agg = LearnedAggregator(len(trainers)).to(device)
            opt = torch.optim.LBFGS(learned_agg.parameters(), lr=0.5, max_iter=100)

            def closure():
                opt.zero_grad()
                out = learned_agg(torch.tensor(X, device=device))
                loss = nn.BCELoss()(out, torch.tensor(Y, device=device))
                loss.backward()
                return loss

            opt.step(closure)
            torch.save(learned_agg.state_dict(), args.agg_ckpt)
            print(f"[Agg] Saved {args.agg_ckpt}")

    # ---------------- Eval on VAL ----------------
    print("[Eval] Scoring anomaly on VAL …")
    all_scores, all_truths = [], []
    for clips, labels in tqdm(val_loader, desc="[Eval] batches"):
        scores = anomaly_score_batch(clips, trainers, bank, device,
                                     rule_graphs=rule_graphs, agg=args.agg,
                                     learned=learned_agg, gate_tau=args.gate_tau)
        truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)
        is_normal = truths.min(dim=1).values
        all_scores.extend(scores.detach().cpu().tolist())
        all_truths.extend((1 - is_normal).tolist())

    import numpy as np
    all_scores = np.array(all_scores, dtype=np.float64)
    all_truths = np.array(all_truths, dtype=np.int64)
    print(f"[Eval] Scores: mean={all_scores.mean():.3f} | std={all_scores.std():.3f}")

    # AUROC
    try:
        from sklearn.metrics import roc_auc_score
        if all_truths.sum() > 0 and all_truths.sum() < len(all_truths):
            auc = roc_auc_score(all_truths, all_scores)
            print(f"[Eval] AUROC (pseudo anomaly=1): {auc:.3f}")
        else:
            print("[Eval] AUROC undefined (all-normal or all-anomaly under pseudo-GT).")
    except Exception as e:
        print(f"[Eval] AUROC skipped ({type(e).__name__}: {e})")

    # per-rule AUROC

    try:
        from sklearn.metrics import roc_auc_score
        import numpy as np

        print("[Diag] Per-rule AUROC (higher is better):")
        R = len(trainers)
        if R == 0:
            print("  (no rules)")
            probs_total = None
            y_total = None
            leaf_total = None
        else:
            per_rule = [[] for _ in range(R)]
            y_rule   = [[] for _ in range(R)]
            probs_buf = []   # store P(rule holds) (B,R)
            truths_buf = []  # store truths (B,R) with 1=holds
            leaf_buf = []    # store leaf labels (B,K) to print labels later

            with torch.no_grad():
                for clips, labels in tqdm(
                    DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers),
                    desc="[Diag] collect", leave=False
                ):
                    clips = clips.to(device, non_blocking=True)

                    # predictions: P(rule holds)
                    probs = []
                    for _, trainer in trainers:
                        if hasattr(trainer, "predict_root_batch"):
                            p = trainer.predict_root_batch(clips, bank)
                        else:
                            p = torch.as_tensor(
                                [trainer.predict_root(clips[b], bank) for b in range(clips.size(0))],
                                device=device
                            )
                        p = p.view(-1).detach().cpu()      # (B,)
                        probs.append(p)

                    P = torch.stack(probs, dim=1)          # (B,R) on CPU
                    probs_buf.append(P)

                    # ground-truth rule truths (1 = holds)
                    labels = labels.to(torch.long)
                    truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)  # (B,R)
                    truths_buf.append(truths.cpu())
                    leaf_buf.append(labels.cpu())

                    # per-rule AUROC lists: anomaly label per rule = 1 - truth; score per rule = 1 - P(rule holds)
                    vr = (1.0 - P).tolist()
                    yr = (1 - truths.cpu()).tolist()
                    for r_idx in range(R):
                        per_rule[r_idx].extend([row[r_idx] for row in vr])
                        y_rule[r_idx].extend([row[r_idx] for row in yr])

            # cache full matrices (OI-style)
            probs_total = 1.0 - torch.cat(probs_buf, dim=0)      # (N,R) = per-rule violation score
            y_total     = 1   - torch.cat(truths_buf, dim=0)     # (N,R) = per-rule anomaly label (violated=1)
            leaf_total  = torch.cat(leaf_buf, dim=0)             # (N,K) = leaf labels

            np.savez_compressed(os.path.join(args.run_dir, "A_score_VidOR_TrainFull.npz"),
                                A_score=probs_total.numpy())
            np.savez_compressed(os.path.join(args.run_dir, "Test_Y_VidOR_TrainFull.npz"),
                                Test_Y=y_total.numpy())

            for r_idx, (rname, _) in enumerate(trainers):
                if len(y_rule[r_idx]) == 0 or len(set(y_rule[r_idx])) <= 1:
                    auc_r = float("nan")
                else:
                    auc_r = roc_auc_score(y_rule[r_idx], per_rule[r_idx])
                print(f"  - {r_idx+1:03d} {rname:60s} : AUROC={auc_r:.3f}")

    except Exception as e:
        import traceback
        print(f"[Diag] Per-rule AUROC block failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        probs_total = None
        y_total = None
        leaf_total = None


    # ------------------------------
    # Samples (OI-style detail)
    # ------------------------------
    print("[Samples] first 200")
    S = 200
    anom_log = os.path.join(args.run_dir, "anom_samples.txt")
    norm_log = os.path.join(args.run_dir, "normal_samples.txt")
    os.makedirs(args.run_dir, exist_ok=True)
    open(anom_log, "w").close()
    open(norm_log, "w").close()

    torch.set_printoptions(precision=2, sci_mode=False)

    # helper to map label indices -> names
    def _label_name(j: int) -> str:
        # y_vec is 0-based indexing
        if 0 <= j < len(concept_names):
            return concept_names[j]
        return f"leaf_{j}"

    for i in range(min(S, len(val_ds))):
        # robust id
        if hasattr(val_ds, "ids"):
            sample_id = val_ds.ids[i]
        else:
            sample_id = f"idx_{i:06d}"

        # per-rule scores vector (OI uses probs_total[i,:])
        if probs_total is None:
            sc = torch.tensor([])
            y_vec = val_ds[i][1]   # fallback: forces decode (only if needed)
        else:
            sc = probs_total[i, :]          # (R,)
            y_vec = leaf_total[i, :]        # (K,)

        # rule truths from graphs (1 = holds)
        truths = compute_rule_truths_batch_from_graphs(y_vec.unsqueeze(0), rule_graphs)  # (1,R)
        status = "NORMAL" if truths.min(dim=1).values.item() == 1 else "ANOMALY"

        # names of ALL leaf labels true for this sample
        on_idx = (y_vec == 1).nonzero(as_tuple=False).flatten().tolist()
        labels_str = ", ".join(_label_name(j) for j in on_idx) if on_idx else "(none)"

        line = f"{sample_id} | status={status:7s} | score={sc} | labels=[{labels_str}]"

        if status == "ANOMALY":
            violated_idx = (truths[0] == 0).nonzero(as_tuple=False).flatten().tolist()
            if violated_idx:
                violated_names = [f"{r+1:03d} {trainers[r][0]}" for r in violated_idx]
                shown = "; ".join(violated_names[:5]) + ("" if len(violated_names) <= 5 else "; …")
                line += f" | violated=[{shown}]"

        with open(anom_log if status == "ANOMALY" else norm_log, "a") as f:
            f.write(f"{sample_id}\n")

        print(line)


if __name__ == "__main__":
    main()
