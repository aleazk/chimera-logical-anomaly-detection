# -*- coding: utf-8 -*-
"""
oi_trainval_anomaly_2lr.py

Train on Open Images *train*; evaluate on *validation*. Two-level logical rules.

Adds only the split refactor vs your val-only script:
  * Top-K computed from TRAIN CSV
  * Train loaders from images/train + train bbox CSV
  * Eval loader from images/validation + validation bbox CSV
  * Validation labels filtered to the train Top-K

All other behaviors are unchanged:
  * taxonomy + optional part→whole rules
  * mined high-confidence pairs (A⇒B) on TRAIN
  * compound (2-level) rules from siblings and mined pairs
  * levelwise training with negatives modes (legacy/ad_strict/…)
  * caching, calibration, aggregator

Run (example):
  python oi_trainval_anomaly_2lr.py --oi_root /scratch3/$USER/oi_challenge \
    --top_k 50 --epochs_leaf 1 --epochs_level 1 \
    --augment --calib_train --negatives ad_strict \
    --agg min --backbone resnet18 \
    --batch_train 256 --batch_eval 128 --num_workers 8 \
    --leaf_ckpt checkpoints/oi_leafbank.pt \
    --cache_dir oi_rule_cache_2lr --run_dir runs/oi_trainval_2lr
"""
import os, json, argparse, csv, hashlib, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Set

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import dgl
from tqdm import tqdm, trange
# + at top of file with the other imports
import io, glob, tarfile
from functools import lru_cache


# your stack
from chimera_logic.trainer import LevelwiseTrainer, SubtreeCache, CacheConfig, subtree_signature  # type: ignore
from chimera_logic.evaluator import propagate_truth_values_hard_from_leaf_labels  # type: ignore

def _rule_sig(g: dgl.DGLGraph) -> str:
    root = int(torch.nonzero(g.out_degrees()==0, as_tuple=False).flatten()[0].item())
    return subtree_signature(g, root)


# -----------------------------
# CLI
# -----------------------------
def build_args():
    p = argparse.ArgumentParser("Open Images train→val anomaly via logical consistency (2-level)")
    # paths / modes
    p.add_argument("--oi_root", type=str, required=True, help="root with csv/, images/train, images/validation")
    p.add_argument("--cache_dir", type=str, default="oi_rule_cache_trainval_2lr")
    p.add_argument("--run_dir", type=str, default="runs/oi_trainval_trainval_2lr")
    p.add_argument("--leaf_ckpt", type=str, default="checkpoints_trainval_2lr/oi_leafbank.pt")
    p.add_argument("--eval_only", action="store_true")
    p.add_argument("--no_stats", action="store_true")
    # class/rule selection
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--max_rules_simple", type=int, default=250, help="max taxonomy/part simple rules")
    p.add_argument("--max_rules_compound", type=int, default=250, help="max compound rules")
    p.add_argument("--per_parent_pair_limit", type=int, default=4, help="limit (sibling_A∧sibling_B)→parent per parent")
    # parts & mining
    p.add_argument("--use_parts", action="store_true", default=True, help="include part→whole rules from hierarchy")
    p.add_argument("--mine_pairs", action="store_true", default=True, help="mine A⇒B from TRAIN CSV")
    p.add_argument("--min_support", type=int, default=200, help="min #images with A (and A∧B) in TRAIN to keep mined rule")
    p.add_argument("--min_conf", type=float, default=0.99, help="min P(B|A) for mined A⇒B")
    p.add_argument("--per_B_pair_limit", type=int, default=3, help="compound (A1∧A2)→B per B from mined rules")
    p.add_argument("--skip_parents", type=str, default="Person,Human face,Human body",
                   help="comma-separated parent display names to skip for simple rules")
    # training knobs
    p.add_argument("--epochs_leaf", type=int, default=3)
    p.add_argument("--epochs_level", type=int, default=2)
    p.add_argument("--lr_leaf", type=float, default=1e-3)
    p.add_argument("--lr_level", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--batch_eval", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--encoder_tag", type=str, default="")
    p.add_argument(
        "--negatives",
        type=str,
        default="ad_strict",
        choices=["legacy","ad_strict","chimera_plus_true","chimeras_only"],
        help=(
            "Batch construction for level training:\n"
            "  legacy             = same-image positives/negatives, no chimeras\n"
            "  ad_strict          = TRUE positives from normal images + chimera NEGATIVES only\n"
            "  chimera_plus_true  = legacy plus chimera POSITIVES and NEGATIVES\n"
            "  chimeras_only      = chimera POSITIVES and NEGATIVES only"
        ),
    )
    # backbone & aug
    p.add_argument("--backbone", type=str, default="resnet18", choices=["tiny","resnet18"])
    p.add_argument("--augment", action="store_true")
    # calibration
    p.add_argument("--calib_train", action="store_true")
    p.add_argument("--calib_ckpt", type=str, default="checkpoints_trainval_2lr/oi_temp_scaler.pt")
    # aggregator
    p.add_argument("--agg", type=str, default="min", choices=["geo","mean","min","learned"])
    p.add_argument("--agg_ckpt", type=str, default="checkpoints_trainval_2lr/oi_agg_2lr.pt")
    # optional subsampling of TRAIN for speed (1.0 = use all)
    p.add_argument("--train_frac", type=float, default=1.0)
    # shards
    p.add_argument("--use_shards", action="store_true",
                help="Read images from tar shards instead of loose JPEG files.")
    p.add_argument("--shards_train", type=str, default="",
                help="Path to shards/train (defaults to <oi_root>/shards/train).")
    p.add_argument("--shards_val", type=str, default="",
                help="Path to shards/validation (defaults to <oi_root>/shards/validation).")
    p.add_argument("--shard_glob", type=str, default="*.tar",
                help="Glob for shard files inside shard dirs.")
    # reproducibility
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--closure_for_truths", action="store_true",
                help="Propagate taxonomy parents in labels when computing rule truths (both rule training and eval).")
    p.add_argument("--gate_tau", type=float, default=0.0,
                help="Margin for antecedent gate; set 0.6–0.7 to suppress weak parent activations.")
    p.add_argument("--mine_direction", choices=["auto","forward","invert"], default="auto",
                help="Choose rule direction per pair. 'invert' = use B→A, 'forward' = A→B, 'auto' = pick by confidence.")
    p.add_argument("--min_conf_fwd", type=float, default=0.60, help="Min P(B|A) if using A→B.")
    p.add_argument("--min_conf_rev", type=float, default=0.60, help="Min P(A|B) if using B→A.")
    p.add_argument("--min_support_count", type=int, default=20,
                help="Min co-occurrence count A∧B; overrides fractional --min_support if >0.")
    p.add_argument("--min_conf_margin", type=float, default=0.05,
                help="Required margin between chosen direction’s conf and the other direction in --mine_direction auto.")
    p.add_argument("--min_taxo_conf", type=float, default=0.0,
                help="Keep inverted taxonomy Parent→Child only if P(Child|Parent) on TRAIN >= this.")
    p.add_argument("--min_part_conf", type=float, default=0.0,
                help="Keep inverted part Whole→Part only if P(Part|Whole) on TRAIN >= this.")


    return p.parse_args()

# -----------------------------
# Paths (TRAIN + VAL)
# -----------------------------
@torch.no_grad()
def evaluate_leaf_bank(bank, loader, idx_to_name, device="cuda", threshold=0.5):
    """
    Prints per-class ROC-AUC, Average Precision, and Accuracy@thr for the leaf bank.
    Assumes:
      - bank.forward_logits(images) -> (B, K) raw logits
      - labels from loader are multi-hot (B, K) in {0,1}
    """
    from sklearn.metrics import roc_auc_score, average_precision_score

    bank.eval(); bank.to(device)

    all_logits = []
    all_labels = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = bank.forward_logits(x)             # (B, K)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.cpu())

    import torch
    logits = torch.cat(all_logits, 0)              # (N, K)
    labels = torch.cat(all_labels, 0).float()      # (N, K)
    probs  = logits.sigmoid().numpy()
    y_true = labels.numpy()

    K = y_true.shape[1]
    # prevalence per class
    prev = labels.sum(0).numpy()

    print("\n[LeafEval] Per-class metrics on VAL")
    print(" class                          prev    ROC-AUC   AP       Acc@0.5")
    print(" ------------------------------------------------------------------")
    for j in range(K):
        yj = y_true[:, j]
        pj = probs[:, j]
        name = idx_to_name[j] if isinstance(idx_to_name, dict) else f"c{j}"
        # guard degenerate
        if yj.max() == yj.min():
            auc = float("nan"); ap = float("nan"); acc = float("nan")
        else:
            try:    auc = roc_auc_score(yj, pj)
            except: auc = float("nan")
            try:    ap  = average_precision_score(yj, pj)
            except: ap  = float("nan")
            pred = (pj >= threshold).astype("float32")
            acc  = (pred == yj).mean() if yj.size > 0 else float("nan")

        print(f" {name:30s} {int(prev[j]):7d}   {auc:7.3f}  {ap:7.3f}   {acc:8.3f}")

    # macro summaries (ignore NaNs)
    import numpy as np
    def _nanmean(a): 
        a = np.array(a, dtype=np.float64)
        return np.nan if a.size==0 else float(np.nanmean(a))
    aucs = []; aps = []; accs = []
    for j in range(K):
        yj = y_true[:, j]
        pj = probs[:, j]
        if yj.max() == yj.min(): 
            continue
        try: aucs.append(roc_auc_score(yj, pj))
        except: pass
        try: aps.append(average_precision_score(yj, pj))
        except: pass
        accs.append(((pj >= threshold).astype("float32") == yj).mean())
    print("\n[LeafEval] Macro ROC-AUC:", _nanmean(aucs))
    print("[LeafEval] Macro AP     :", _nanmean(aps))
    print("[LeafEval] Macro Acc@0.5:", _nanmean(accs))
    print()

class Paths:
    def __init__(self, root: str):
        self.root = root
        self.csv_dir = os.path.join(root, "csv")
        self.img_train = os.path.join(root, "images", "train")
        self.img_val   = os.path.join(root, "images", "validation")

        # bbox CSVs (header-aware elsewhere)
        cand_bbox_train = [
            "challenge-2019-train-detection-bbox.csv",
            "train-annotations-bbox.csv",
        ]
        cand_bbox_val = [
            "challenge-2019-validation-detection-bbox.csv",
            "validation-annotations-bbox.csv",
        ]
        self.bbox_train = self._pick_first_existing(cand_bbox_train)
        self.bbox_val   = self._pick_first_existing(cand_bbox_val)
        if self.bbox_train is None:
            raise FileNotFoundError(f"No TRAIN bbox CSV in {self.csv_dir}")
        if self.bbox_val is None:
            raise FileNotFoundError(f"No VAL bbox CSV in {self.csv_dir}")

        # class descriptions (500-class or boxable), either is fine
        cand_classes = [
            "challenge-2019-classes-description-500.csv",
            "class-descriptions-boxable.csv",
        ]
        self.classes_csv = self._pick_first_existing(cand_classes)
        if self.classes_csv is None:
            raise FileNotFoundError(f"No classes description CSV in {self.csv_dir}")

        # hierarchy
        self.hier_json = os.path.join(self.csv_dir, "bbox_labels_600_hierarchy.json")
        self.hier_json_alt = os.path.join(self.csv_dir, "challenge-2019-label500-hierarchy.json")

    def _pick_first_existing(self, names: List[str]) -> Optional[str]:
        for name in names:
            p = os.path.join(self.csv_dir, name)
            if os.path.exists(p) and os.path.getsize(p) > 0:
                return p
        return None

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_hierarchy(paths: Paths):
    if os.path.exists(paths.hier_json) or os.path.exists(paths.hier_json_alt):
        return
    os.makedirs(paths.csv_dir, exist_ok=True)
    url = "https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json"
    try:
        import urllib.request
        print("[Setup] Downloading hierarchy JSON …")
        urllib.request.urlretrieve(url, paths.hier_json)
        print(f"[Setup] Saved hierarchy → {paths.hier_json}")
    except Exception as e:
        print(f"[WARN] Could not download hierarchy: {e}.")

def read_classes_any(path_csv: str) -> Dict[str, str]:
    mid_to_name = {}
    with open(path_csv, "r", newline="") as f:
        r = csv.reader(f)
        for row in r:
            if not row or row[0] in ("LabelName", ""):
                continue
            mid, name = row[0], row[1]
            mid_to_name[mid] = name
    return mid_to_name

def read_hierarchy(path_json: str) -> dict:
    with open(path_json, "r") as f:
        return json.load(f)

# collect taxonomy and part edges
def collect_edges_with_parts(hier: dict) -> Tuple[List[Tuple[str,str]], List[Tuple[str,str]]]:
    taxo, parts = [], []
    def dfs(node, parent=None):
        mid = node.get("LabelName")
        if parent is not None and mid is not None:
            taxo.append((mid, parent))
        for ch in node.get("Subcategory", []) or []:
            dfs(ch, mid)
        for ch in node.get("Part", []) or []:
            pmid = ch.get("LabelName")
            if pmid and mid:
                parts.append((pmid, mid))  # part -> whole
            dfs(ch, mid)
    dfs(hier)
    return taxo, parts

# -----------------------------
# CSV readers (header-aware)
# -----------------------------
def _find_col_index(header, name: str) -> int:
    try:
        return header.index(name)
    except ValueError:
        raise RuntimeError(f"CSV header missing '{name}': {header}")

def top_k_from_counts(bbox_csv: str, K: int) -> List[str]:
    counts: Dict[str, int] = {}
    with open(bbox_csv, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None: raise RuntimeError(f"Empty CSV: {bbox_csv}")
        idx_label = _find_col_index(header, "LabelName")
        for row in r:
            if len(row) <= idx_label: continue
            mid = row[idx_label].strip()
            counts[mid] = counts.get(mid, 0) + 1
    mids_sorted = sorted(counts.keys(), key=lambda m: counts[m], reverse=True)
    return mids_sorted[:K]

def build_image_labels(bbox_csv: str, class_filter: Set[str]) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    per_image: Dict[str, Set[str]] = {}
    counts: Dict[str, int] = {}
    with open(bbox_csv, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None: raise RuntimeError(f"Empty CSV: {bbox_csv}")
        idx_image = _find_col_index(header, "ImageID")
        idx_label = _find_col_index(header, "LabelName")
        for row in r:
            if len(row) <= max(idx_image, idx_label): continue
            img_id = row[idx_image].strip()
            mid = row[idx_label].strip()
            if mid not in class_filter: continue
            s = per_image.setdefault(img_id, set())
            if mid not in s:
                s.add(mid)
                counts[mid] = counts.get(mid, 0) + 1
    return {k: sorted(list(v)) for k, v in per_image.items()}, counts

# -----------------------------
# Dataset
# -----------------------------
class OIDataset(Dataset):
    def __init__(self, img_dir: str, image_labels: Dict[str, List[str]],
                 mid_to_idx: Dict[str, int], transform=None):
        self.img_dir = img_dir
        all_ids = list(image_labels.keys())
        self.ids = [iid for iid in all_ids if os.path.exists(os.path.join(img_dir, f"{iid}.jpg"))]
        self.labels = image_labels
        self.mid_to_idx = mid_to_idx
        self.K = len(mid_to_idx)
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor()])
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        iid = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, f"{iid}.jpg")).convert("RGB")
        x = self.transform(img)
        y = torch.zeros(self.K, dtype=torch.long)
        for mid in self.labels[iid]:
            j = self.mid_to_idx.get(mid, None)
            if j is not None: y[j] = 1
        return x, y

def _basename_noext(p: str) -> str:
    b = os.path.basename(p)
    return os.path.splitext(b)[0]

def scan_shards(shard_dir: str, pattern: str, allowed_ids: Set[str]) -> List[Tuple[str, str, str]]:
    """
    Return a flat index of (tar_path, member_name, image_id) for members named '<id>.jpg'
    that are present in `allowed_ids`. This scans only headers (fast).
    """
    index: List[Tuple[str, str, str]] = []
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, pattern)))
    for tar_path in shard_paths:
        try:
            with tarfile.open(tar_path, mode="r") as tf:
                for m in tf:
                    if not m.isreg():
                        continue
                    if not m.name.lower().endswith(".jpg"):
                        continue
                    iid = _basename_noext(m.name)
                    if iid in allowed_ids:
                        index.append((tar_path, m.name, iid))
        except Exception as e:
            print(f"[WARN] Could not read shard {tar_path}: {e}")
    return index

@lru_cache(maxsize=256)
def _open_tar_cached(tar_path: str) -> tarfile.TarFile:
    return tarfile.open(tar_path, mode="r")

class OIShardDataset(Dataset):
    """
    Map-style dataset that reads images from .tar shards.
    `index` is a list of (tar_path, member_name, image_id).
    Labels come from `image_labels[image_id]`.
    """
    def __init__(self, index: List[Tuple[str,str,str]],
                 image_labels: Dict[str, List[str]],
                 mid_to_idx: Dict[str, int],
                 transform=None):
        self.index = index
        self.labels = image_labels
        self.mid_to_idx = mid_to_idx
        self.K = len(mid_to_idx)
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor()])
        # expose ids for your sample printout
        self.ids = [iid for _,_,iid in index]

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        tar_path, member_name, iid = self.index[idx]
        tf = _open_tar_cached(tar_path)
        # read bytes -> PIL
        ext = tf.extractfile(member_name)
        if ext is None:
            # rare tar corruption; skip gracefully
            # return a black image + zeros
            x = torch.zeros(3,224,224); y = torch.zeros(self.K, dtype=torch.long)
            return x, y
        data = ext.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        x = self.transform(img)
        y = torch.zeros(self.K, dtype=torch.long)
        for mid in self.labels.get(iid, []):
            j = self.mid_to_idx.get(mid, None)
            if j is not None:
                y[j] = 1
        return x, y


# -----------------------------
# Leaf bank (same as your script)
# -----------------------------
class OILeafBank(nn.Module):
    def __init__(self, K: int, feat_dim: int = 256, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            import torchvision.models as M
            m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))
            in_ch = 512
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32,64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64,96, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(96,128,3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
            ); in_ch = 128
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, feat_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(K)])
        self.temp_scaler: Optional[TempScaler] = None
    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        z = self.backbone(images); return self.proj(z)
    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        z = self.encoder(images); return torch.cat([h(z) for h in self.heads], dim=1)
    def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(images)
        if self.temp_scaler is not None: logits = self.temp_scaler(logits)
        return torch.sigmoid(logits)

class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # keep grads & device; do NOT .cpu()
        if self.T.device != logits.device:
            self.T.data = self.T.data.to(logits.device)
        return logits / self.T.clamp_min(1e-3)


# -----------------------------
# Fingerprint (unchanged)
# -----------------------------
def _hash_state_dict(sd) -> str:
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:24]

def encoder_fingerprint(leaf_bank: nn.Module, manual_tag: Optional[str] = None) -> str:
    if manual_tag: return f"manual_{manual_tag}"
    parts = []
    for attr in ["encoder", "backbone", "proj", "cnn", "trunk", "feature_extractor", "body"]:
        m = getattr(leaf_bank, attr, None)
        if isinstance(m, nn.Module): parts.append(m.state_dict())
    if parts:
        h = hashlib.sha256()
        for sd in parts: h.update(_hash_state_dict(sd).encode())
        return h.hexdigest()[:24]
    try: return _hash_state_dict(leaf_bank.state_dict())
    except Exception: return hashlib.sha256(repr(leaf_bank).encode()).hexdigest()[:24]

# -----------------------------
# Trainer helpers (unchanged)
# -----------------------------
def make_transforms(split: str, use_aug: bool) -> T.Compose:
    if split == "train" and use_aug:
        aug = [
            T.Resize((224,224)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            T.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.9,1.1)),
            T.ToTensor(),
            T.RandomErasing(p=0.1),
        ]
    else:
        aug = [T.Resize((224,224)), T.ToTensor()]
    return T.Compose(aug)

def train_leaf_bank(bank: OILeafBank, loader: DataLoader, epochs: int, lr: float, device: str,
                    weight_decay: float = 0.0, pos_weight: Optional[torch.Tensor] = None):
    bank = bank.to(device)
    opt = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bank.train()
    prev_scaler = getattr(bank, "scaler", None)
    bank.scaler = None
    pbar_epochs = trange(1, epochs+1, desc="[LeafBank] epochs", leave=True)
    for ep in pbar_epochs:
        total = 0.0; n=0
        pbar = tqdm(loader, desc=f"[LeafBank] epoch {ep}/{epochs}", leave=False)
        for j, (imgs,labels) in enumerate(pbar):
            #if j > 2:                               #-------------------------------DEBUG-------------------------
                #break
            imgs = imgs.to(device); labels = labels.to(device).float()
            logits = bank.forward_logits(imgs)
            labels = labels.to(logits.device).float() 
            loss = crit(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
            pbar.set_postfix(avg_loss=f"{total/max(n,1):.4f}")
        pbar_epochs.set_postfix(avg_loss=f"{total/max(n,1):.4f}")
    bank.scaler = prev_scaler

def fit_temperature(bank: OILeafBank, loader: DataLoader, device: str, max_samples: Optional[int]=None) -> TempScaler:
    bank.eval()
    all_logits, all_labels, seen = [], [], 0
    with torch.inference_mode():
        for imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = bank.forward_logits(imgs).detach().cpu()
            all_logits.append(logits); all_labels.append(y.float())
            seen += imgs.size(0)
            if max_samples is not None and seen >= max_samples: break
    logits_cpu = torch.cat(all_logits, dim=0)
    labels_cpu = torch.cat(all_labels, dim=0)
    scaler = TempScaler()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    crit = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = crit(scaler(logits_cpu), labels_cpu); loss.backward(); return loss
    opt.step(closure)
    return scaler

# -----------------------------
# Rule DSL (unchanged)
# -----------------------------
@dataclass
class RuleSpecSimple:
    kind: str      # 'IMPLIES'
    head_cid: int  # 1-based
    body_cid: int  # 1-based
    name: str

OPC = {"IFF":1, "IMPLIES":2, "AND":3, "OR":4}
def leaf(cid: int):     return ("leaf", cid)
def NOT(expr):          return ("not", expr)
def AND(a, b):          return ("AND", a, b)
def OR(a, b):           return ("OR", a, b)
def IFF(a, b):          return ("IFF", a, b)
def IMPLIES(l, r):      return ("IMPLIES", l, r)

def _add_expr(expr, nodes, edges):
    kind = expr[0]
    if kind == "leaf":
        cid = int(expr[1])
        nid = len(nodes)
        nodes.append({"mask":1, "y":0, "x":cid})
        return nid
    if kind == "not":
        return _add_expr(expr[1], nodes, edges)  # NOT carried by incoming edge at parent
    op_name, a, b = expr
    left  = _add_expr(a, nodes, edges)
    right = _add_expr(b, nodes, edges)
    nid = len(nodes)
    nodes.append({"mask":0, "y":OPC[op_name], "x":0})
    def _is_not(e): return isinstance(e, tuple) and len(e)>0 and e[0]=="not"
    neg_l = -1 if _is_not(a) else +1
    neg_r = -1 if _is_not(b) else +1
    edges.append((left,  nid, neg_l, 0))
    edges.append((right, nid, neg_r, 1))
    return nid

def build_graph_from_expr(expr) -> dgl.DGLGraph:
    nodes, edges = [], []
    _ = _add_expr(expr, nodes, edges)

    # If the root is IMPLIES, flip pos at the root and ensure antecedent-first edge order
    if edges:
        root_id = len(nodes) - 1
        if nodes[root_id]["y"] == OPC["IMPLIES"]:
            # flip pos only for edges going into the root
            flipped = []
            root_edges, non_root_edges = [], []
            for (s, d, neg, pos) in edges:
                if d == root_id:
                    root_edges.append((s, d, neg, 1 - pos))
                else:
                    non_root_edges.append((s, d, neg, pos))
            # antecedent (pos==0) first
            root_edges.sort(key=lambda e: e[3])  # 0, then 1
            edges = non_root_edges + root_edges

    g = dgl.graph(([], []), num_nodes=len(nodes))
    g.ndata["mask"] = torch.tensor([n["mask"] for n in nodes], dtype=torch.long)
    g.ndata["y"]    = torch.tensor([n["y"]    for n in nodes], dtype=torch.long)
    g.ndata["x"]    = torch.tensor([n["x"]    for n in nodes], dtype=torch.long)
    if edges:
        src = torch.tensor([s for (s,_,_,_) in edges], dtype=torch.long)
        dst = torch.tensor([d for (_,d,_,_) in edges], dtype=torch.long)
        g.add_edges(src, dst)
        g.edata["neg"] = torch.tensor([neg for (_,_,neg,_) in edges], dtype=torch.long)
        g.edata["pos"] = torch.tensor([pos for (_,_,_,pos) in edges], dtype=torch.long)
    return g



def build_graph_for_simple(rule: RuleSpecSimple) -> dgl.DGLGraph:
    g = dgl.graph(([], []), num_nodes=3)
    mask = torch.tensor([1, 1, 0], dtype=torch.long)
    y    = torch.tensor([0, 0, 2], dtype=torch.long)  # IMPLIES
    x    = torch.tensor([rule.head_cid, rule.body_cid, 0], dtype=torch.long)

    # ANTECEDENT FIRST (body/parent=1, then head/child=0)
    g.add_edges(torch.tensor([1, 0]), torch.tensor([2, 2]))

    g.ndata["mask"] = mask
    g.ndata["y"]    = y
    g.ndata["x"]    = x
    g.edata["neg"]  = torch.tensor([+1, +1], dtype=torch.long)
    g.edata["pos"]  = torch.tensor([0, 1], dtype=torch.long)  # left,right
    return g

# --- helpers to build simple rules with explicit direction ---
def build_graph_for_simple_forward(ante_cid: int, cons_cid: int) -> dgl.DGLGraph:
    """
    Encodes (Antecedent -> Consequent). Left child = antecedent.
    """
    g = dgl.graph(([], []), num_nodes=3)
    mask = torch.tensor([1, 1, 0], dtype=torch.long)
    y    = torch.tensor([0, 0, 2], dtype=torch.long)      # IMPLIES at node 2
    # x[0] = consequent, x[1] = antecedent; left edge (pos=0) must be antecedent
    x    = torch.tensor([cons_cid, ante_cid, 0], dtype=torch.long)
    # add edges so left edge (pos=0) comes from node 1 (antecedent), right from node 0 (consequent)
    g.add_edges(torch.tensor([1, 0]), torch.tensor([2, 2]))
    g.ndata["mask"] = mask; g.ndata["y"] = y; g.ndata["x"] = x
    g.edata["neg"]  = torch.tensor([+1, +1], dtype=torch.long)
    g.edata["pos"]  = torch.tensor([0, 1], dtype=torch.long)  # left=antecedent
    return g

def build_graph_for_simple_inverted(ante_cid: int, cons_cid: int) -> dgl.DGLGraph:
    """
    Encodes the inverted direction (Consequent -> Antecedent).
    Equivalent to build_graph_for_simple_forward(cons, ante).
    """
    return build_graph_for_simple_forward(cons_cid, ante_cid)


def apply_taxonomy_closure(labels: torch.Tensor,
                           ancestors_for_idx: Dict[int, List[int]]) -> torch.Tensor:
    """
    labels: (B, K) 0/1 long/byte tensor.
    For every class i with label=1, set all its ancestors to 1.
    """
    y = labels.clone()
    B, K = y.shape
    for i in range(K):
        anc = ancestors_for_idx.get(i, [])
        if not anc: 
            continue
        on = (y[:, i] == 1).nonzero(as_tuple=False).flatten()
        if on.numel():
            y[on.unsqueeze(1), torch.tensor(anc, dtype=torch.long).unsqueeze(0)] = 1
    return y





# -----------------------------
# Truths from graphs (unchanged)
# -----------------------------
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

# -----------------------------
# Aggregation (unchanged)
# -----------------------------
class LearnedAggregator(nn.Module):
    def __init__(self, R: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(R))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        x = torch.logit(torch.clamp(probs, 1e-6, 1-1e-6))
        return torch.sigmoid(x @ self.w + self.b)

@torch.no_grad()
def anomaly_score_batch(images: torch.Tensor,
                        trainers,                      # list of (name, trainer)
                        bank: OILeafBank,
                        device: str,
                        rule_graphs: List[dgl.DGLGraph],
                        agg: str = "min",
                        learned: Optional[LearnedAggregator] = None,
                        gate_tau: float = 0.0) -> torch.Tensor:
    """
    Antecedent-gated anomaly:
      per-rule violation s_r = gate(p_ante, tau) * (1 - p_rule)
      where gate(p, tau) = ReLU(p - tau)/(1 - tau) if tau>0 else p
    Aggregate:
      min  -> max(s_r)
      mean -> mean(s_r)
      geo  -> exp(mean(log(s_r + eps)))
      learned -> 1 - Learned(1 - s_r)
    """
    images = images.to(device, non_blocking=True)
    # P(antecedent) from leaves
    leaf_probs = bank.forward_probs(images)  # (B, K)

    per_rule = []
    for r_idx, (_, trainer) in enumerate(trainers):
        # P(rule holds)
        if getattr(trainer, "_enc_fprint", None) is None:
            trainer._enc_fprint = encoder_fingerprint(bank)
        if hasattr(trainer, "predict_root_batch"):
            p_rule = trainer.predict_root_batch(images, bank)  # (B,)
        else:
            p_rule = torch.stack(
                [torch.as_tensor(trainer.predict_root(images[b], bank), device=device)
                 for b in range(images.size(0))], dim=0)
        p_rule = torch.clamp(p_rule, 1e-6, 1.0)

        # antecedent concept id = left child of root (we enforced antecedent-first in builders)
        g = rule_graphs[r_idx]
        root = int(torch.nonzero(g.out_degrees() == 0, as_tuple=False).flatten()[0].item())
        src, dst, eid = g.in_edges(root, form='all')
        left_node = int(src[0].item())  # first edge is antecedent
        cid = int(g.ndata["x"][left_node].item())  # 1-based; 0 if composite
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

    S = torch.stack(per_rule, dim=1)  # (B, R)
    eps = 1e-9
    if agg == "min":
        out = S.max(dim=1).values
    elif agg == "mean":
        out = S.mean(dim=1)
    elif agg == "geo":
        out = torch.exp(torch.log(S + eps).mean(dim=1))
    elif agg == "learned" and learned is not None:
        out = 1.0 - learned(1.0 - S)
    else:
        out = S.max(dim=1).values
    return out



# -----------------------------
# Mining co-occurrence rules (on TRAIN)
# -----------------------------
# -----------------------------
# Mining co-occurrence rules (on TRAIN) — auto direction
# -----------------------------
def mine_rules_from_csv_auto(
    bbox_csv: str,
    K_mids: List[str],
    mid_to_idx: Dict[str, int],
    min_support_count: int,
    min_conf_fwd: float,
    min_conf_rev: float,
    min_lift: float,
    mine_direction: str = "auto",
    min_conf_margin: float = 0.05,
    plausible_pair = None,
    ) -> List[dict]:
    """
    Returns list of dicts:
      {'ante': i_idx, 'cons': j_idx, 'conf': float, 'lift': float, 'support': int, 'dir': 'fwd'|'rev',
       'A_mid': A_mid, 'B_mid': B_mid, 'conf_fwd':..., 'conf_rev':...}
    Direction selection:
      forward: keep A->B if conf_fwd >= min_conf_fwd and lift_fwd >= min_lift
      invert:  keep B->A if conf_rev >= min_conf_rev and lift_rev >= min_lift
      auto:    choose the side whose conf passes threshold and beats the other by margin
    """
    from collections import defaultdict

    # collect image sets per class (TRAIN)
    imgs_for: Dict[str, Set[str]] = defaultdict(set)
    with open(bbox_csv, "r", newline="") as f:
        r = csv.reader(f)
        header = next(r, None)
        if header is None:
            raise RuntimeError(f"Empty CSV: {bbox_csv}")
        idx_img = _find_col_index(header, "ImageID")
        idx_lab = _find_col_index(header, "LabelName")
        for row in r:
            mid = row[idx_lab].strip()
            if mid in K_mids:
                imgs_for[mid].add(row[idx_img].strip())

    mids = [m for m in K_mids if m in imgs_for]
    # total images that have any of these labels (for base-rate p(B))
    all_imgs = set()
    for m in mids: all_imgs |= imgs_for[m]
    tot = max(len(all_imgs), 1)

    results: List[dict] = []
    seen_pairs = set()  # (ante_idx, cons_idx)

    for A_mid in mids:
        Aset = imgs_for[A_mid]
        nA = len(Aset)
        if nA < min_support_count:
            continue
        for B_mid in mids:
            if B_mid == A_mid:
                continue
            Bset = imgs_for[B_mid]
            nB = len(Bset)
            if nB < min_support_count:
                continue
            nAB = len(Aset & Bset)
            if nAB < min_support_count:
                continue

            # probabilities / confidences / lifts
            pA  = nA / tot
            pB  = nB / tot
            conf_fwd = nAB / nA if nA else 0.0  # P(B|A)
            conf_rev = nAB / nB if nB else 0.0  # P(A|B)
            lift_fwd = (conf_fwd / max(pB, 1e-12)) if pB > 0 else 0.0
            lift_rev = (conf_rev / max(pA, 1e-12)) if pA > 0 else 0.0

            Ai = mid_to_idx[A_mid]; Bi = mid_to_idx[B_mid]

            chosen = None
            if mine_direction == "forward":
                if conf_fwd >= min_conf_fwd and lift_fwd >= min_lift:
                    chosen = ("fwd", Ai, Bi, conf_fwd, lift_fwd)
            elif mine_direction == "invert":
                if conf_rev >= min_conf_rev and lift_rev >= min_lift:
                    chosen = ("rev", Bi, Ai, conf_rev, lift_rev)  # note: ante=Bi, cons=Ai
            else:  # auto
                f_ok = (conf_fwd >= min_conf_fwd and lift_fwd >= min_lift)
                r_ok = (conf_rev >= min_conf_rev and lift_rev >= min_lift)
                if f_ok and (not r_ok or (conf_fwd >= conf_rev + min_conf_margin)):
                    chosen = ("fwd", Ai, Bi, conf_fwd, lift_fwd)
                elif r_ok and (not f_ok or (conf_rev >= conf_fwd + min_conf_margin)):
                    chosen = ("rev", Bi, Ai, conf_rev, lift_rev)

            if chosen is None:
                continue

            dir_tag, ante_i, cons_i, conf, lift = chosen

            # plausibility check in idx space (optional)
            if plausible_pair is not None and not plausible_pair(ante_i, cons_i):
                continue

            key = (ante_i, cons_i)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            results.append({
                'ante': ante_i, 'cons': cons_i, 'conf': float(conf), 'lift': float(lift),
                'support': int(nAB), 'dir': dir_tag,
                'A_mid': A_mid, 'B_mid': B_mid,
                'conf_fwd': float(conf_fwd), 'conf_rev': float(conf_rev),
            })

    # sort by confidence, then lift, then support
    results.sort(key=lambda r: (r['conf'], r['lift'], r['support']), reverse=True)
    return results


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

    device = ("cuda" if (args.device == "auto" and torch.cuda.is_available()) else
              ("cuda" if args.device == "cuda" else "cpu"))
    print(f"[Setup] Device: {device}")

    P = Paths(args.oi_root)
    for path in [P.bbox_train, P.bbox_val, P.classes_csv, P.img_train, P.img_val]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Expected at: {path}")
    ensure_hierarchy(P)
    hier_path = P.hier_json if os.path.exists(P.hier_json) else P.hier_json_alt
    if not os.path.exists(hier_path):
        print("[WARN] No hierarchy JSON found; cannot build rules."); return

    mid_to_name = read_classes_any(P.classes_csv)
    print(f"[IO] Using TRAIN bbox CSV: {P.bbox_train}")
    print(f"[IO] Using VAL   bbox CSV: {P.bbox_val}")
    print(f"[IO] Using classes CSV:   {P.classes_csv}")

    # ---- Top-K from TRAIN ----
    mids_topK = top_k_from_counts(P.bbox_train, args.top_k)
    if len(mids_topK) == 0:
        raise RuntimeError("No classes found in TRAIN CSV.")
    mid_to_idx = {mid: i for i, mid in enumerate(mids_topK)}
    idx_to_mid = {i: mid for mid, i in mid_to_idx.items()}
    K = len(mids_topK)
    print(f"[Classes] K (from TRAIN) = {K}")
    # ---- Hierarchy edges ----
    hier = read_hierarchy(hier_path)
    taxo, parts = collect_edges_with_parts(hier)
    # --- transitive ancestors among Top-K (for closure_for_truths) ---
    mid_is_top = set(mids_topK)
    parent_of_mid = {}
    for c_mid, p_mid in taxo:
        parent_of_mid.setdefault(c_mid, set()).add(p_mid)

    def _all_ancestors(mid: str):
        out, stack = set(), [mid]
        while stack:
            m = stack.pop()
            for pm in parent_of_mid.get(m, ()):
                if pm not in out:
                    out.add(pm); stack.append(pm)
        return [a for a in out if a in mid_is_top]

    ancestors_for_idx = {i: [] for i in range(K)}
    for mid, i in mid_to_idx.items():
        if mid in mid_is_top:
            anc = _all_ancestors(mid)
            ancestors_for_idx[i] = [mid_to_idx[a] for a in anc]

    # --- plausibility filter for mined pairs (taxonomy or parts related) ---
    parts_idx: Set[Tuple[int,int]] = set()
    for pmid, wmid in parts:
        if pmid in mid_to_idx and wmid in mid_to_idx:
            parts_idx.add((mid_to_idx[pmid], mid_to_idx[wmid]))  # (part, whole) in idx space

    def plausible_pair_idx(i: int, j: int) -> bool:
        # taxonomy ancestor either way
        if j in set(ancestors_for_idx.get(i, [])) or i in set(ancestors_for_idx.get(j, [])):
            return True
        # parts either way
        if (i, j) in parts_idx or (j, i) in parts_idx:
            return True
        return False


    # ---- Build TRAIN and VAL image-level labels, filtered to K ----
    train_image_labels, _ = build_image_labels(P.bbox_train, set(mids_topK))
    val_image_labels,   _ = build_image_labels(P.bbox_val,   set(mids_topK))

    # Build fast TRAIN co-occurrence sets for Top-K
    imgs_for_train: Dict[str, Set[str]] = {mid: set() for mid in mids_topK}
    for iid, mids in train_image_labels.items():
        s = set(mids)
        for mid in s:
            if mid in imgs_for_train:
                imgs_for_train[mid].add(iid)

    def _conf_B_given_A(A_mid: str, B_mid: str) -> float:
        A = imgs_for_train.get(A_mid, set())
        B = imgs_for_train.get(B_mid, set())
        return (len(A & B) / max(len(A), 1)) if A else 0.0


    n_train_imgs = sum(os.path.exists(os.path.join(P.img_train, f"{iid}.jpg")) for iid in train_image_labels.keys())
    n_val_imgs   = sum(os.path.exists(os.path.join(P.img_val,   f"{iid}.jpg")) for iid in val_image_labels.keys())
    print(f"[Data] TRAIN images with ≥1 of top-{K}: {len(train_image_labels)} | on disk: {n_train_imgs}")
    print(f"[Data]   VAL images with ≥1 of top-{K}: {len(val_image_labels)} | on disk: {n_val_imgs}")


    # --- transitive ancestors among Top-K ---
    mid_is_top = set(mids_topK)
    parent_of_mid = {}
    for c_mid, p_mid in taxo:
        parent_of_mid.setdefault(c_mid, set()).add(p_mid)


    mids_in_hier = {m for (c,p) in taxo for m in (c,p)}
    if args.use_parts:
        mids_in_hier |= {m for (c,p) in parts for m in (c,p)}
    keep = set(mids_topK) & mids_in_hier

    # skip trivial parents by display name
    skip_names = set([s.strip() for s in args.skip_parents.split(",") if s.strip()])
    def ok_parent(mid: str) -> bool:
        return mid_to_name.get(mid, mid) not in skip_names

    # taxonomy child→parent  (encoded as parent→child due to inverted simple builder)
    rules_simple: List[Tuple[str, dgl.DGLGraph]] = []
    count_simple = 0
    seen_sigs: Set[str] = set()           # <— add this set once, reuse also for compound

    # taxonomy child→parent (encoded as parent→child due to inverted simple builder)
    for c, p in taxo:
        if c in keep and p in keep and ok_parent(p):
            # empirical plausibility gate (optional)
            if args.min_taxo_conf > 0.0:
                if _conf_B_given_A(p, c) < args.min_taxo_conf:
                    continue
            i = mid_to_idx[c] + 1
            j = mid_to_idx[p] + 1
            name = f"{mid_to_name.get(p, p)} -> {mid_to_name.get(c, c)}  [inv]"
            g = build_graph_for_simple(RuleSpecSimple("IMPLIES", i, j, name))
            sig = _rule_sig(g)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            rules_simple.append((name, g))
            count_simple += 1
            if count_simple >= args.max_rules_simple:
                break

    # part→whole (encoded as whole→part due to inverted simple builder)
    if args.use_parts and count_simple < args.max_rules_simple:
        for part, whole in parts:
            if part in keep and whole in keep and ok_parent(whole):
                # empirical plausibility gate (optional)
                if args.min_part_conf > 0.0:
                    if _conf_B_given_A(whole, part) < args.min_part_conf:
                        continue
                i = mid_to_idx[part] + 1
                j = mid_to_idx[whole] + 1
                name = f"{mid_to_name.get(whole, whole)} -> {mid_to_name.get(part, part)}  [inv]"
                g = build_graph_for_simple(RuleSpecSimple("IMPLIES", i, j, name))

                sig = _rule_sig(g)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)

                rules_simple.append((name, g))
                count_simple += 1
                if count_simple >= args.max_rules_simple:
                    break



    print(f"[Rules] Simple (taxonomy + parts): {len(rules_simple)}")


    # --------- Build compound rules (siblings and mined pairs) ---------
    compound_items: List[Tuple[str, dgl.DGLGraph]] = []
    count_comp = 0

    # prepare maps for siblings by parent
    by_parent: Dict[str, List[str]] = {}
    parent_of: Dict[str, str] = {}
    for c, p in taxo:
        if c in keep and p in keep:
            by_parent.setdefault(p, []).append(c)
            parent_of[c] = p

    # (sibling_A ∧ sibling_B) → parent  (root-flip encodes parent → (A ∧ B))
    for p, childs in by_parent.items():
        if len(childs) < 2 or not ok_parent(p): 
            continue
        pairs = []
        for k in range(len(childs) - 1):
            pairs.append((childs[k], childs[k+1]))
            if len(pairs) >= args.per_parent_pair_limit: 
                break
        for a, b in pairs:
            expr = IMPLIES(AND(leaf(mid_to_idx[a] + 1), leaf(mid_to_idx[b] + 1)),
                        leaf(mid_to_idx[p] + 1))
            name = f"{mid_to_name.get(p,p)} → ({mid_to_name.get(a,a)} ∧ {mid_to_name.get(b,b)}) [inv]"
            g = build_graph_from_expr(expr)
            sig = _rule_sig(g)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            compound_items.append((name, g))
            count_comp += 1
            if count_comp >= args.max_rules_compound:
                break
        if count_comp >= args.max_rules_compound:
            break



    # mined pairs A⇒B
    mined_pairs: List[Tuple[str, str]] = []
    # mined pairs with auto direction
    if args.mine_pairs and count_comp < args.max_rules_compound:
        mined = mine_rules_from_csv_auto(
            P.bbox_train, mids_topK, mid_to_idx,
            min_support_count=max(args.min_support, args.min_support_count),
            min_conf_fwd=args.min_conf_fwd, min_conf_rev=args.min_conf_rev,
            min_lift=1.10,  # gentle lift default; adjust via CLI if you like
            mine_direction=args.mine_direction,
            min_conf_margin=args.min_conf_margin,
            plausible_pair=plausible_pair_idx
        )

        # --- simple mined (direction-aware, dedup already ensured) ---
        for r in mined:
            if count_simple >= args.max_rules_simple:
                break
            a_i, c_i = r['ante'], r['cons']
            nmA, nmC = mid_to_name.get(idx_to_mid[a_i], idx_to_mid[a_i]), mid_to_name.get(idx_to_mid[c_i], idx_to_mid[c_i])
            tag = "[mined inv]" if r['dir'] == "rev" else "[mined]"
            # build graph with correct direction; antecedent must be left child
            g = build_graph_for_simple_forward(a_i + 1, c_i + 1) if r['dir'] == "fwd" \
                else build_graph_for_simple_inverted(a_i + 1, c_i + 1)
            name = f"{nmA} -> {nmC} {tag} (sup={r['support']}, conf={r['conf']:.3f}, lift={r['lift']:.2f})"
            rules_simple.append((name, g))
            count_simple += 1

        # --- compounds from mined: need ≥2 distinct antecedents per consequent ---
        from collections import defaultdict
        ants_for_cons: Dict[int, List[dict]] = defaultdict(list)
        for r in mined:
            ants_for_cons[r['cons']].append(r)

        for cons_i, lst in ants_for_cons.items():
            # keep best antecedent per class (by conf), then make disjoint pairs
            best_for_ante: Dict[int, dict] = {}
            for r in lst:
                a_i = r['ante']
                if a_i not in best_for_ante or r['conf'] > best_for_ante[a_i]['conf']:
                    best_for_ante[a_i] = r
            ants = sorted(best_for_ante.values(), key=lambda x: (x['conf'], x['lift'], x['support']), reverse=True)

            pairs = []
            used = set()
            for i in range(len(ants)):
                a1 = ants[i]['ante']
                if a1 in used: 
                    continue
                for j in range(i+1, len(ants)):
                    a2 = ants[j]['ante']
                    if a2 in used or a2 == a1: 
                        continue
                    pairs.append((a1, a2))
                    used.add(a1); used.add(a2)
                    if len(pairs) >= args.per_B_pair_limit:
                        break
                if len(pairs) >= args.per_B_pair_limit:
                    break

            # Build in the same direction used for the simple mined rule(s).
            # If any mined record for this consequent used 'rev', prefer the inverted style.
            use_rev = any(r['dir'] == 'rev' for r in lst)
            for a1, a2 in pairs:
                nmC = mid_to_name.get(idx_to_mid[cons_i], idx_to_mid[cons_i])
                nmA1 = mid_to_name.get(idx_to_mid[a1], idx_to_mid[a1])
                nmA2 = mid_to_name.get(idx_to_mid[a2], idx_to_mid[a2])
                if use_rev:
                    # cons -> (a1 ∧ a2)
                    expr = IMPLIES(leaf(cons_i + 1), AND(leaf(a1 + 1), leaf(a2 + 1)))
                    nm = f"{nmC} → ({nmA1} ∧ {nmA2}) [mined inv]"
                else:
                    # (a1 ∧ a2) -> cons
                    expr = IMPLIES(AND(leaf(a1 + 1), leaf(a2 + 1)), leaf(cons_i + 1))
                    nm = f"({nmA1} ∧ {nmA2}) → {nmC} [mined]"
                g = build_graph_from_expr(expr)
                sig = _rule_sig(g)
                if sig in seen_sigs:
                    continue
                seen_sigs.add(sig)
                compound_items.append((nm, g))
                count_comp += 1

                if count_comp >= args.max_rules_compound:
                    break
            if count_comp >= args.max_rules_compound:
                break



    print(f"[Rules] Compound (2-level): {len(compound_items)}")
    if (len(rules_simple) + len(compound_items)) == 0:
        print("[WARN] No rules built; try increasing --top_k or enabling --use_parts/--mine_pairs.")
        return

    # ---------------- Train/Eval datasets ----------------
    tf_train = make_transforms("train", use_aug=args.augment)
    tf_eval  = make_transforms("val",   use_aug=False)

    # optional subsample TRAIN ids (by label dict presence)
    train_ids_all = [iid for iid in train_image_labels.keys()]
    if args.train_frac < 1.0:
        g = torch.Generator().manual_seed(args.seed)
        n_keep = max(1, int(len(train_ids_all) * args.train_frac))
        perm = torch.randperm(len(train_ids_all), generator=g).tolist()[:n_keep]
        train_ids = [train_ids_all[i] for i in perm]
    else:
        train_ids = train_ids_all
    val_ids = list(val_image_labels.keys())

    # Decide shards vs files
    use_shards = args.use_shards
    shards_train = args.shards_train or os.path.join(args.oi_root, "shards", "train")
    shards_val   = args.shards_val   or os.path.join(args.oi_root, "shards", "validation")

    if use_shards and os.path.isdir(shards_train) and os.path.isdir(shards_val):
        print(f"[Data] Using shards (train): {shards_train}")
        print(f"[Data] Using shards (val)  : {shards_val}")
        # Build indices filtered to ids that have labels
        train_allowed = set(train_ids)
        val_allowed   = set(val_ids)

        idx_train = scan_shards(shards_train, args.shard_glob, train_allowed)
        idx_val   = scan_shards(shards_val,   args.shard_glob, val_allowed)

        print(f"[Data] TRAIN shard index: {len(idx_train)} images (labelled & present in shards)")
        print(f"[Data]   VAL shard index: {len(idx_val)} images (labelled & present in shards)")

        # Filter label dicts down to what actually exists in shards
        train_labels = {iid: train_image_labels[iid] for _,_,iid in idx_train}
        val_labels   = {iid: val_image_labels[iid]   for _,_,iid in idx_val}

        train_ds = OIShardDataset(idx_train, train_labels, mid_to_idx, transform=tf_train)
        val_ds   = OIShardDataset(idx_val,   val_labels,   mid_to_idx, transform=tf_eval)

    else:
        # Fallback to loose JPEGs (your original path)
        n_train_imgs = sum(os.path.exists(os.path.join(P.img_train, f"{iid}.jpg")) for iid in train_ids)
        n_val_imgs   = sum(os.path.exists(os.path.join(P.img_val,   f"{iid}.jpg")) for iid in val_ids)
        print(f"[Data] TRAIN images with ≥1 of top-{K}: {len(train_image_labels)} | on disk: {n_train_imgs}")
        print(f"[Data]   VAL images with ≥1 of top-{K}: {len(val_image_labels)} | on disk: {n_val_imgs}")

        train_labels = {iid: train_image_labels[iid] for iid in train_ids
                        if os.path.exists(os.path.join(P.img_train, f"{iid}.jpg"))}
        val_labels   = {iid: val_image_labels[iid]   for iid in val_ids
                        if os.path.exists(os.path.join(P.img_val, f"{iid}.jpg"))}

        train_ds = OIDataset(P.img_train, train_labels, mid_to_idx, transform=tf_train)
        val_ds   = OIDataset(P.img_val,   val_labels,   mid_to_idx, transform=tf_eval)


    train_loader = None
    if not args.eval_only:
        train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True,
                                  num_workers=args.num_workers, pin_memory=(device == "cuda"))
        if not args.no_stats:
            # tally prevalence on TRAIN
            cnt = torch.zeros(K, dtype=torch.long)
            for _, y in tqdm(DataLoader(train_ds, batch_size=256, shuffle=False,
                                        num_workers=args.num_workers, pin_memory=(device == "cuda")),
                             desc="[Data] tally (TRAIN)", leave=False):
                cnt += y.sum(dim=0)
            print("[Data] Class counts (TRAIN):")
            for i in range(K):
                mid = idx_to_mid[i]; nm = mid_to_name.get(mid, mid)
                print(f"  c{i+1:02d} {nm:35s} : {cnt[i].item()}")

    val_loader = DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    # ---------------- Leaf bank ----------------
    bank = OILeafBank(K=K, feat_dim=args.feat_dim, backbone=args.backbone).to(device)

    if args.eval_only:
        if not os.path.exists(args.leaf_ckpt):
            raise FileNotFoundError(f"--eval_only set but leaf checkpoint not found: {args.leaf_ckpt}")
        bank.load_state_dict(torch.load(args.leaf_ckpt, map_location="cpu"), strict=True)
        bank.eval()
        print(f"[LeafBank] Loaded checkpoint: {args.leaf_ckpt}")
    else:
        if os.path.exists(args.leaf_ckpt):
            bank.load_state_dict(torch.load(args.leaf_ckpt, map_location="cpu"), strict=True)
            print(f"[LeafBank] Warm-start from {args.leaf_ckpt}")
        pos_weight = None
        if not args.no_stats:
            try:
                cnt = torch.zeros(K, dtype=torch.long)
                for _, y in DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=args.num_workers):
                    cnt += y.sum(dim=0)
                total = len(train_ds)
                pos_w = []
                for i in range(K):
                    p = max(cnt[i].item(), 1); n = max(total - p, 1)
                    pos_w.append(n / p)
                pos_weight = torch.tensor(pos_w, dtype=torch.float32, device=device)
            except Exception:
                pos_weight = None
        if args.epochs_leaf > 0:
            print("[LeafBank] Training …")
            assert train_loader is not None
            train_leaf_bank(bank, train_loader, epochs=args.epochs_leaf, lr=args.lr_leaf,
                            device=device, weight_decay=args.weight_decay, pos_weight=pos_weight)
            torch.save(bank.state_dict(), args.leaf_ckpt)
            print(f"[LeafBank] Saved checkpoint → {args.leaf_ckpt}")
            idx_to_name = {idx: mid_to_name[mid] for idx, mid in idx_to_mid.items()}
            evaluate_leaf_bank(bank, val_loader, idx_to_name, device=device, threshold=0.5)
        else:
            bank.eval()

    # Calibration on TRAIN
    if args.calib_train and not args.eval_only and args.epochs_leaf > 0:
        calib_loader = DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
        if device == "cuda": torch.cuda.empty_cache()
        scaler = fit_temperature(bank, calib_loader, device=device)
        torch.save(scaler.state_dict(), args.calib_ckpt)
        print(f"[Calib] Saved temperature scaler → {args.calib_ckpt}")
    if os.path.exists(args.calib_ckpt):
        scaler = TempScaler(); scaler.load_state_dict(torch.load(args.calib_ckpt, map_location="cpu"))
        bank.temp_scaler = scaler.to(device)
        print(f"[Calib] Loaded temperature scaler: {args.calib_ckpt}")

    # ---------------- Build trainers (simple + compound) ----------------
    trainers = []
    rule_graphs = []
    # Make a closed-labels loader for rule training if requested
    train_loader_closed = train_loader
    if (not args.eval_only) and args.closure_for_truths:
        def collate_with_closure(batch):
            xs, ys = zip(*batch)
            xs = torch.stack(xs, dim=0)
            ys = torch.stack(ys, dim=0)
            ys = apply_taxonomy_closure(ys, ancestors_for_idx)
            return xs, ys
        train_loader_closed = DataLoader(
            train_ds, batch_size=args.batch_train, shuffle=True,
            num_workers=args.num_workers, pin_memory=(device == "cuda"),
            collate_fn=collate_with_closure
        )

    # simple
    for i, (name, g) in enumerate(rules_simple, start=1):
        rule_cache = os.path.join(args.cache_dir, f"rule_simple_{i:03d}")
        cache = SubtreeCache(CacheConfig(root_dir=rule_cache))
        trainer = LevelwiseTrainer(g, N_concepts=K, cache=cache, device=device,
                                   cache_leafroot_only=False, lineage_aware=True)
        trainer._enc_fprint = encoder_fingerprint(bank, manual_tag=(args.encoder_tag or None))
        root = int(torch.nonzero(g.out_degrees()==0, as_tuple=False).flatten()[0].item())
        print(f"[Trainer] Simple {i:03d}: {name} | topo_sig={subtree_signature(g, root)}")
        if not args.eval_only:
            kwargs = dict(dataset=train_loader_closed, leaf_bank=bank, epochs_per_level=args.epochs_level,
                        lr=args.lr_level, negatives=args.negatives, use_soft_leaves=True, verbose=True, use_tqdm=True)

            trainer.train(**kwargs)
        trainers.append((name, trainer))
        rule_graphs.append(g)

    # compound
    for j, (name, g) in enumerate(compound_items, start=1):
        rule_cache = os.path.join(args.cache_dir, f"rule_comp_{j:03d}")
        cache = SubtreeCache(CacheConfig(root_dir=rule_cache))
        trainer = LevelwiseTrainer(g, N_concepts=K, cache=cache, device=device,
                                   cache_leafroot_only=False, lineage_aware=True)
        trainer._enc_fprint = encoder_fingerprint(bank, manual_tag=(args.encoder_tag or None))
        root = int(torch.nonzero(g.out_degrees()==0, as_tuple=False).flatten()[0].item())
        print(f"[Trainer] Compound {j:03d}: {name} | topo_sig={subtree_signature(g, root)}")
        if not args.eval_only:
            kwargs = dict(dataset=train_loader_closed, leaf_bank=bank, epochs_per_level=args.epochs_level,
                        lr=args.lr_level, negatives=args.negatives, use_soft_leaves=True, verbose=True, use_tqdm=True)

            trainer.train(**kwargs)
        trainers.append((name, trainer))
        rule_graphs.append(g)

    # Manifest
    manifest = {
        "leaf_ckpt": os.path.abspath(args.leaf_ckpt),
        "cache_dir": os.path.abspath(args.cache_dir),
        "encoder_fingerprint": encoder_fingerprint(bank, manual_tag=(args.encoder_tag or None)),
        "feat_dim": args.feat_dim,
        "K": K,
        "topK_mids": [idx_to_mid[i] for i in range(K)],
        "topK_names": [read_classes_any(P.classes_csv).get(idx_to_mid[i], idx_to_mid[i]) for i in range(K)],
        "n_rules_simple": len(rules_simple),
        "n_rules_compound": len(compound_items),
        "agg": args.agg,
        "backbone": args.backbone,
        "use_parts": args.use_parts,
        "mine_pairs": args.mine_pairs,
        "min_support": args.min_support,
        "min_conf": args.min_conf,
        "train_frac": args.train_frac,
    }
    with open(os.path.join(args.run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Run] Manifest saved → {os.path.join(args.run_dir, 'manifest.json')}")

    # ---------------- Optional learned aggregator ----------------
    learned_agg: Optional[LearnedAggregator] = None
    if args.agg == "learned":
        if args.eval_only and os.path.exists(args.agg_ckpt):
            learned_agg = LearnedAggregator(len(trainers))
            learned_agg.load_state_dict(torch.load(args.agg_ckpt, map_location="cpu"))
            learned_agg.to(device).eval()
            print(f"[Agg] Loaded learned aggregator: {args.agg_ckpt}")
        elif not args.eval_only:
            X_list, Y_list = [], []
            for imgs, labels in tqdm(DataLoader(train_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers),
                                     desc="[Agg] collect (TRAIN)", leave=False):
                imgs = imgs.to(device)
                probs = []
                for _, trainer in trainers:
                    if hasattr(trainer, "predict_root_batch"):
                        p = trainer.predict_root_batch(imgs, bank)
                    else:
                        p = torch.tensor([trainer.predict_root(imgs[b], bank) for b in range(imgs.size(0))], device=device)
                    probs.append(torch.clamp(p, 1e-6, 1.0))
                Pp = torch.stack(probs, dim=1)
                truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)  # (B,R)
                is_normal = truths.min(dim=1).values
                y = 1 - is_normal.float()
                X_list.append(Pp.detach().cpu()); Y_list.append(y.detach().cpu())
            X = torch.cat(X_list, dim=0); Y = torch.cat(Y_list, dim=0)
            learned_agg = LearnedAggregator(len(trainers)).to(device)
            opt = torch.optim.LBFGS(learned_agg.parameters(), lr=0.5, max_iter=100)
            def closure():
                opt.zero_grad()
                out = learned_agg(torch.tensor(X, device=device))
                loss = nn.BCELoss()(out, torch.tensor(Y, device=device))
                loss.backward(); return loss
            opt.step(closure)
            torch.save(learned_agg.state_dict(), args.agg_ckpt)
            print(f"[Agg] Saved learned aggregator → {args.agg_ckpt}")

    # ---------------- Eval on VAL ----------------
    print("[Eval] Scoring anomaly on Open Images *validation* …")
    all_scores, all_truths = [], []
    for imgs, labels in tqdm(val_loader, desc="[Eval] batches"):
        scores = anomaly_score_batch(
            imgs, trainers, bank, device=device,
            rule_graphs=rule_graphs, agg=args.agg, learned=learned_agg, gate_tau=args.gate_tau
        )

        labels_for_truths = apply_taxonomy_closure(labels, ancestors_for_idx) if args.closure_for_truths else labels
        truths = compute_rule_truths_batch_from_graphs(labels_for_truths, rule_graphs)

        is_normal = truths.min(dim=1).values
        y = (1 - is_normal).tolist()
        all_scores.extend(scores.detach().cpu().tolist())
        all_truths.extend(y)

    import numpy as np
    all_scores = np.array(all_scores); all_truths = np.array(all_truths)
    print(f"[Eval] Scores: mean={all_scores.mean():.3f} | std={all_scores.std():.3f}")
    if all_truths.sum() > 0 and all_truths.sum() < len(all_truths):
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_truths, all_scores)
            print(f"[Eval] AUROC (anomaly=1): {auc:.3f}")
        except Exception:
            print("[Eval] Install scikit-learn for AUROC.")

    # Per-rule AUROC (optional, on VAL)
    # Per-rule AUROC (optional)
    try:
        from sklearn.metrics import roc_auc_score
        print("[Diag] Per-rule AUROC (higher is better):")
        R = len(trainers)
        if R == 0:
            print("  (no rules)"); 
        else:
            per_rule = [[] for _ in range(R)]
            y_rule   = [[] for _ in range(R)]
            probs_total = []
            y_total = []
            with torch.no_grad():
                for imgs, labels in tqdm(
                    DataLoader(val_ds, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers),
                    desc="[Diag] collect", leave=False
                ):
                    imgs = imgs.to(device, non_blocking=True)
                    # predictions: list of shape-(B,) tensors
                    probs = []
                    for _, trainer in trainers:
                        if hasattr(trainer, "predict_root_batch"):
                            p = trainer.predict_root_batch(imgs, bank)           # (B,) or (B,1)
                        else:
                            p = torch.as_tensor(
                                [trainer.predict_root(imgs[b], bank) for b in range(imgs.size(0))],
                                device=device
                            )
                        p = p.view(-1).detach().cpu()                             # enforce (B,)
                        probs.append(p)
                    P = torch.stack(probs, dim=1)                                  # (B, R)
                    probs_total.append(P)

                    # ground-truth RULE truths for this batch (with optional hierarchy closure)
                    labels = labels.to(torch.long)
                    if args.closure_for_truths:
                        labels_for_truths = apply_taxonomy_closure(labels, ancestors_for_idx)
                    else:
                        labels_for_truths = labels
                    truths = compute_rule_truths_batch_from_graphs(labels_for_truths, rule_graphs)  # (B, R), 1=holds
                    y_total.append(truths)

                    # collect per-rule lists
                    # anomaly label per rule = 1 - truth; score per rule = 1 - P(rule holds)
                    vr = (1.0 - P).tolist()
                    yr = (1 - truths).tolist()
                    for r_idx in range(R):
                        # extend by column r_idx
                        per_rule[r_idx].extend([row[r_idx] for row in vr])
                        y_rule[r_idx].extend([row[r_idx] for row in yr])

            probs_total = 1 - torch.cat(probs_total, dim=0)
            y_total = 1 - torch.cat(y_total, dim=0)
            np.savez_compressed("./A_score_OI_TrainFull.npz", A_score=probs_total.cpu().numpy())
            np.savez_compressed("./Test_Y_OI_TrainFull.npz", Test_Y=y_total.cpu().numpy())

            for r_idx, (rname, _) in enumerate(trainers):
                if len(y_rule[r_idx]) == 0 or len(set(y_rule[r_idx])) <= 1:
                    auc_r = float('nan')  # only one class present in y_true
                else:
                    auc_r = roc_auc_score(y_rule[r_idx], per_rule[r_idx])
                print(f"  - {r_idx+1:03d} {rname:50s} : AUROC={auc_r:.3f}")

    except Exception as e:
        # Don't swallow silently—print a concise hint so you can fix fast next time.
        import traceback
        print(f"[Diag] Per-rule AUROC block failed: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)


    # Samples
    print("[Samples]")
    S = 100
    for i in range(min(S, len(val_ds))):
        x, y = val_ds[i]

        # robust image id (matches filename in images/validation)
        img_id = val_ds.ids[i] if hasattr(val_ds, "ids") else (val_ids[i] if "val_ids" in locals() else str(i))

        # sc = anomaly_score_batch(
        #     x.unsqueeze(0), trainers, bank, device=device,
        #     rule_graphs=rule_graphs, agg=args.agg, learned=learned_agg, gate_tau=args.gate_tau
        # )[0].item()
        sc = probs_total[i,:]


        y_closed = apply_taxonomy_closure(y.unsqueeze(0), ancestors_for_idx)[0] if args.closure_for_truths else y
        truths = compute_rule_truths_batch_from_graphs(y_closed.unsqueeze(0), rule_graphs)

        status = "NORMAL" if truths.min(dim=1).values.item() == 1 else "ANOMALY"

        # names of ALL labels that are true (y==1) for this image
        on_idx = (y_closed == 1).nonzero(as_tuple=False).flatten().tolist()
        label_names = []
        for j in on_idx:
            try:
                mid = idx_to_mid[j]
                nm = mid_to_name.get(mid, mid)
            except Exception:
                nm = f"c{j:02d}"
            label_names.append(nm)
        labels_str = ", ".join(label_names) if label_names else "(none)"

        torch.set_printoptions(precision=2, sci_mode=False)
        line = f"{img_id} | status={status:7s} | score={sc} | labels=[{labels_str}]"

        if status == "ANOMALY":
            # list all rules violated by ground-truth on this sample
            violated_idx = (truths[0] == 0).nonzero(as_tuple=False).flatten().tolist()
            if violated_idx:
                violated_names = [f"{r+1:03d} {trainers[r][0]}" for r in violated_idx]
                shown = "; ".join(violated_names[:5]) + ("" if len(violated_names) <= 5 else "; …")
                line += f" | violated=[{shown}]"
        if status == "ANOMALY":
            filename = "/scratch3/asc007/Evaluator/open_images/anom_samples.txt"
            with open(filename, 'a') as f:          # append instead of overwrite
                f.write(f"{img_id}\n")
        else:
            filename = "/scratch3/asc007/Evaluator/open_images/normal_samples.txt"
            with open(filename, 'a') as f:
                f.write(f"{img_id}\n")

        print(line)


if __name__ == "__main__":
    main()
