# -*- coding: utf-8 -*-
"""
clevr_anomaly.py  (adds 2-level rules)

- Keeps your existing simple (1-layer) rules.
- Adds compound 2-level rules built from nested expressions.
- Computes per-rule ground-truth using the actual rule graphs.
"""

# from __future__ import annotations  # keep commented if your env complains

import os, json, argparse, hashlib
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional, Any, Sequence

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import dgl
from tqdm import tqdm, trange

from chimera_logic.trainer import LevelwiseTrainer, SubtreeCache, CacheConfig, subtree_signature
# NEW: use your hard-logic propagator to evaluate any rule graph (incl. compound)
from chimera_logic.evaluator import propagate_truth_values_hard_from_leaf_labels
# --- NEW (videos) ---
import re, glob, math, random
try:
    import cv2  # for video decoding
except Exception:
    cv2 = None


# -----------------------------
# CLI
# -----------------------------
DEF_ROOT = "./CLEVR_v1.0"

def build_args():
    p = argparse.ArgumentParser("CLEVR anomaly with persistence + upgrades (now with 2-level rules)")
    # paths / modes
    p.add_argument("--clevr_root", type=str, default=DEF_ROOT)
    p.add_argument("--cache_dir", type=str, default="ckpt_cl/clevr_rule_cache_2lr", help="directory for lineage-aware gates")
    p.add_argument("--run_dir", type=str, default="ckpt_cl/runs/clevr_anomaly_2lr", help="where to save manifest/logs")
    p.add_argument("--leaf_ckpt", type=str, default="ckpt_cl/checkpoints_2lr/clevr_leafbank.pt")
    p.add_argument("--eval_only", action="store_true", help="skip all training and just evaluate using saved artifacts")
    p.add_argument("--no_stats", action="store_true", help="skip concept prevalence tally print")
    # training knobs
    p.add_argument("--epochs_leaf", type=int, default=3)
    p.add_argument("--epochs_level", type=int, default=2)
    p.add_argument("--lr_leaf", type=float, default=1e-3)
    p.add_argument("--lr_level", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_train", type=int, default=256)
    p.add_argument("--batch_eval", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--device", type=str, default="auto", choices=["auto","cpu","cuda"])
    p.add_argument("--encoder_tag", type=str, default="", help="manual encoder lineage tag (overrides hashing)")
    p.add_argument("--train_missing_only", action="store_true", help="when training, skip nodes already cached (if trainer supports it)")
    p.add_argument(
        "--negatives",
        type=str,
        default="true_only",
        choices=["true_only", "ad_strict", "ad_chimera_pos", "chimera_plus_true", "chimeras_only"],
        help=(
            "Batch construction for level training:\n"
            "  true_only             = original same-image positives/negatives, no chimeras\n"
            "  ad_strict          = same-image TRUE positives from normal images + chimera NEGATIVES only\n"
            "  chimera_plus_true  = true_only plus chimera POSITIVES and chimera NEGATIVES\n"
            "  chimeras_only      = chimera POSITIVES and NEGATIVES only (no same-image)"
        ),
    )
    # backbone & aug
    p.add_argument("--backbone", type=str, default="tiny", choices=["tiny","resnet18"], help="leaf-bank backbone")
    p.add_argument("--augment", action="store_true", help="use basic data augmentation for leaf-bank training")
    # calibration
    p.add_argument("--calib_train", action="store_true", help="fit temperature scaler after leaf training and save to --calib_ckpt")
    p.add_argument("--calib_ckpt", type=str, default="checkpoints_2lr/temp_scaler.pt")
    # aggregator
    p.add_argument("--agg", type=str, default="geo", choices=["geo","mean","min","learned"], help="rule aggregation strategy")
    p.add_argument("--agg_ckpt", type=str, default="checkpoints_2lr/agg.pt", help="save/load learned aggregator weights")
    # rule mining (still available for simple rules; compound are hand-crafted)
    p.add_argument("--auto_mine", action="store_true", help="mine A⇒B / A⇒¬B rules from train labels")
    p.add_argument("--support_thresh", type=float, default=0.05)
    p.add_argument("--confidence_pos", type=float, default=0.995)
    p.add_argument("--confidence_neg", type=float, default=0.005)
    p.add_argument("--max_rules", type=int, default=25)
    # compound rule mining (CLEVRER)
    p.add_argument("--no_static_rules", action="store_true",
                   help="Do not include static CLEVR-style rules (recommended for CLEVRER).")
    p.add_argument("--no_mine_compound_rules", action="store_true",
                   help="Disable mining of enter -> (collide & collide) compound rules.")
    p.add_argument("--max_compound_rules", type=int, default=64,
                   help="Cap for mined compound rules (enter -> (collide & collide)).")
    p.add_argument("--compound_scan_split", type=str, default="val", choices=["train","val"],
                   help="Which split annotations to scan when mining compound rules.")
    p.add_argument("--compound_scan_limit", type=int, default=2000,
                   help="How many videos to scan for mining compound rules (0 = all).")
    p.add_argument("--compound_min_hits", type=int, default=5,
                   help="Minimum number of videos where enter and BOTH collides happen (for mined rules).")
    p.add_argument("--compound_min_conf", type=float, default=0.20,
                   help="Minimum confidence P(both_collides | enter) for mined rules.")
    p.add_argument("--compound_min_viol", type=int, default=1,
                   help="Minimum number of violations (enter true, but both_collides false) for mined rules.")
    # gates hyperparams placeholders (trainer owns details)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--gate_dropout", type=float, default=0.1)
    p.add_argument("--gate_layernorm", action="store_true")
    # imbalance
    p.add_argument("--pos_weight", type=str, default="auto", help="'auto' for prevalence-based per-concept weights, or a float")

    # --- NEW (videos) in build_args() ---
    p.add_argument("--dataset", type=str, default="clevr", choices=["clevr", "clevrer"],
                help="Use CLEVR (images) or CLEVRER (videos).")
    p.add_argument("--clevrer_root", type=str, default="./clevrer",
                help="Root containing train_video/, validation_video/, annotations/, questions/")
    p.add_argument("--frames", type=int, default=16, help="Uniformly sampled frames per clip.")
    p.add_argument("--resize", type=int, default=224, help="Frame resize (square H=W).")

    # optional concept expansion
    p.add_argument("--add_event_concepts", action="store_true",
                help="Add event concepts (any collision / any enter / any exit).")
    p.add_argument("--add_question_concepts", action="store_true",
                help="Mine frequent attributes from questions and add as extra concepts.")
    p.add_argument("--q_train_json", type=str, default=None, help="Override questions train.json path")
    p.add_argument("--q_val_json", type=str, default=None, help="Override questions val.json path")
    # === Event concept options ===
    p.add_argument("--event_rich", action="store_true",
                    help="Add object-conditioned, temporal, attribute-bound event concepts for CLEVRER.")
    p.add_argument("--event_filters", type=str, default="shapes",
                    choices=["shapes", "shapes+colors", "shapes+colors+materials"],
                    help="Which attribute filters to use to define event atoms.")
    p.add_argument("--max_event_pairs", type=int, default=8,
                    help="Max number of attribute-pair combinations for pairwise event concepts.")
    p.add_argument("--event_half", type=int, default=64,
                    help="Frame threshold for 'collision_before_half'. If unknown, defaults to 64.")

    return p.parse_args()
    

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
# -----------------------------
# Paths bundle
# -----------------------------
# --- REPLACE your path pickers with these ---

def _pick_video_dir(root: str, split: str) -> str:
    """
    Return a directory that actually contains *.mp4 (recursively) for the split.
    Handles:
      root/videos/{train,validation}/video_00000-01000/*.mp4
      root/{train_video,validation_video}/video_*/*.mp4
      etc.
    """
    cand = [
        os.path.join(root, "videos", "train" if split == "train" else "validation"),
        os.path.join(root, f"{'train' if split=='train' else 'validation'}_video"),
    ]
    best = None
    for c in cand:
        if not os.path.isdir(c):
            continue
        n = (len(glob.glob(os.path.join(c, "**", "video_*.mp4"), recursive=True)) +
             len(glob.glob(os.path.join(c, "**", "video_*.avi"), recursive=True)))
        if n > 0:
            return c
        if best is None:
            best = c
    return best or cand[0]

def _pick_ann_path(root: str, split: str) -> str:
    """
    Return a directory containing annotation_*.json (recursively), or a monolithic file.
    Prefers the directory form if it actually has files.
    """
    dirp = os.path.join(root, "annotations", "train" if split == "train" else "validation")
    if os.path.isdir(dirp):
        n = len(glob.glob(os.path.join(dirp, "**", "annotation_*.json"), recursive=True))
        if n > 0:
            return dirp
    filep = os.path.join(root, f"annotation_{'train' if split=='train' else 'validation'}.json")
    if os.path.isfile(filep):
        return filep
    alt = os.path.join(root, "annotations", f"{'train' if split=='train' else 'validation'}.json")
    if os.path.isfile(alt):
        return alt
    return dirp  # last resort


class Paths:
    def __init__(self, root: str):
        self.root = root
        self.train_scenes = os.path.join(root, "scenes", "CLEVR_train_scenes.json")
        self.val_scenes   = os.path.join(root, "scenes", "CLEVR_val_scenes.json")
        self.img_train    = os.path.join(root, "images", "train")
        self.img_val      = os.path.join(root, "images", "val")

# -----------------------------
# Concepts & seed rules (simple)
# -----------------------------

EVENT_CONCEPTS: List[Tuple[str, Dict[str,str]]] = [
    ("any_collision", {"event":"collision_any"}),
    ("any_enter",     {"event":"enter_any"}),
    ("any_exit",      {"event":"exit_any"}),
]

@dataclass
class RuleSpec:
    kind: str    # 'IMPLIES' | 'IMPLIES_NOT'
    head_cid: int
    body_cid: int
    name: str

SEED_RULES: List[RuleSpec] = [
    RuleSpec("IMPLIES_NOT", 1, 2, "blue_sphere -> not red_sphere"),
    RuleSpec("IMPLIES",     3, 4, "green_cube -> metal_any"),
    RuleSpec("IMPLIES_NOT", 5, 6, "yellow_cyl -> not gray_cyl"),
]

# ======================
# Event concept helpers
# ======================
from typing import Dict, List, Tuple, Optional

def _obj_attrs(o: Dict) -> Dict[str, str]:
    """
    Robustly extract and normalize CLEVRER object attributes.
    Handles ints (e.g., color=2), strings, and multiple nests.
    """
    c = _normalize_attr(_get_attr_any(o, "color"),    "color")
    s = _normalize_attr(_get_attr_any(o, "shape"),    "shape")
    m = _normalize_attr(_get_attr_any(o, "material"), "material")
    return {"color": c, "shape": s, "material": m}

def _match_filter(attrs: Dict[str,str], flt: Dict[str,str]) -> bool:
    # flt is subset e.g. {"shape":"cube"} or {"color":"red","shape":"sphere"}
    for k,v in flt.items():
        if k.startswith("_"):  # skip meta keys
            continue
        if attrs.get(k,"") != str(v).lower():
            return False
    return True

# --- helpers for robust obj/timing extraction --------------------------------
def _obj_id(o: dict) -> int:
    for k in ("object_id", "id", "idx", "instance_id"):
        if k in o:
            try: return int(o[k])
            except: pass
    raise KeyError("no object id key in object dict")

def _get_frames(ann: dict) -> list:
    # CLEVRER has frames as a list; sometimes nested under 'video' or 'annotations'
    frames = ann.get("frames")
    if isinstance(frames, list): return frames
    # fallback: some dumps pack under ann['annotations']['frames']
    frames = ann.get("annotations", {}).get("frames")
    return frames if isinstance(frames, list) else []

def _first_last_frames(meta: dict):
    """
    Return:
      first: {obj_id -> first_frame_index}
      last : {obj_id -> last_frame_index}
      total_frames: int (best effort)
      total_seconds: float|None
    We handle three sources:
      (A) top-level frames list with per-frame object lists
      (B) object_property / objects with 'trajectory' entries
      (C) metadata with num frames / duration
    """
    # ---- Try to get frame count & duration from metadata ----
    md = meta.get("metadata") or meta.get("video_metadata") or {}
    total_seconds = None
    for k in ("video_duration", "duration", "total_time"):
        if isinstance(md.get(k), (int, float)) and md[k] > 0:
            total_seconds = float(md[k]); break

    total_frames = None
    for k in ("num_frames", "frame_count", "nframes", "video_num_frames"):
        v = md.get(k)
        if isinstance(v, (int, float)) and v > 0:
            total_frames = int(v); break

    # (A) top-level frames list
    first, last = {}, {}
    frames = meta.get("frames")
    if isinstance(frames, list) and frames:
        for t, fr in enumerate(frames):
            objs = fr.get("objects") or fr.get("object_list") or []
            if not isinstance(objs, list): 
                continue
            for o in objs:
                oid = None
                for key in ("object_id", "id", "idx", "instance_id"):
                    if key in o:
                        try: oid = int(o[key]); break
                        except: pass
                if oid is None: 
                    continue
                if oid not in first: first[oid] = t
                last[oid] = t
        if total_frames is None:
            total_frames = len(frames)

    # (B) object_property / objects with trajectory
    #    structure typically: object_property: {"1": {"id":1, "trajectory":[{"frame":..}|{"frame_id":..}|{"time":..}]}, ...}
    def _iter_object_dicts(r):
        if isinstance(r, dict):
            # likely containers
            if "object_property" in r and isinstance(r["object_property"], dict):
                for v in r["object_property"].values():
                    if isinstance(v, dict): 
                        yield v
            if "objects" in r and isinstance(r["objects"], list):
                for v in r["objects"]:
                    if isinstance(v, dict):
                        yield v
            # generic walking
            for v in r.values():
                yield from _iter_object_dicts(v)
        elif isinstance(r, list):
            for v in r:
                yield from _iter_object_dicts(v)

    saw_any_traj = False
    max_frame_seen = -1
    for od in _iter_object_dicts(meta):
        # id
        oid = None
        for k in ("object_id","id","idx","instance_id"):
            if k in od:
                try: oid = int(od[k]); break
                except: pass
        if oid is None: 
            continue

        # trajectories might be under several keys
        traj = (od.get("trajectory") or od.get("motion_trajectory") or od.get("track") or od.get("traj"))
        if not isinstance(traj, list):
            continue

        fmin, fmax = None, None
        for step in traj:
            if not isinstance(step, dict): 
                continue
            t_frame = step.get("frame")
            if t_frame is None: t_frame = step.get("frame_id")
            # If only 'time' (seconds) exists, keep it but we need frames too; we’ll derive later
            try:
                if t_frame is not None:
                    t_frame = int(t_frame)
                    fmin = t_frame if fmin is None else min(fmin, t_frame)
                    fmax = t_frame if fmax is None else max(fmax, t_frame)
            except Exception:
                pass

        if fmin is not None and fmax is not None:
            saw_any_traj = True
            if oid not in first or fmin < first.get(oid, 1 << 30):
                first[oid] = fmin
            if oid not in last or fmax > last.get(oid, -1):
                last[oid] = fmax
            max_frame_seen = max(max_frame_seen, fmax)

    if total_frames is None:
        if saw_any_traj and max_frame_seen >= 0:
            total_frames = max_frame_seen + 1
        else:
            total_frames = 128  # safest fallback

    # fill missing ids with defaults (present everywhere)
    for oid in set(list(first.keys()) + list(last.keys())):
        first.setdefault(oid, 0)
        last.setdefault(oid, total_frames - 1)

    return first, last, int(total_frames), total_seconds


def _collisions(meta: dict):
    """
    Return list of dicts:
      { 'a':int, 'b':int, 't_frame': Optional[int], 't_time': Optional[float] }
    Accept keys: 'collision' or 'collisions' (list), or any nested 'events'.
    """
    out = []

    def _push(a, b, t_frame, t_time):
        out.append({'a': a, 'b': b, 't_frame': t_frame, 't_time': t_time})

    def _as_int(x):
        try: return int(x)
        except: 
            try: return int(float(x))
            except: return None

    def _as_float(x):
        try: return float(x)
        except: return None

    # Walk JSON looking for lists of collisions / events
    def _walk(d):
        if isinstance(d, dict):
            for k in ("collision","collisions","events"):
                lst = d.get(k)
                if isinstance(lst, list):
                    for e in lst:
                        if not isinstance(e, dict): 
                            continue
                        typ = (e.get("type") or e.get("label") or "").lower()
                        if k in ("collision","collisions") and typ == "":
                            typ = "collision"
                        if typ != "collision":
                            continue

                        ids = (e.get("objects") or e.get("object_ids") or e.get("pair") or e.get("object_pair") or [])
                        if not isinstance(ids, (list, tuple)) or len(ids) < 2:
                            continue
                        try:
                            a, b = int(ids[0]), int(ids[1])
                        except Exception:
                            continue

                        t_frame = None
                        for tfk in ("frame", "frame_id", "contact_frame"):
                            if tfk in e:
                                t_frame = _as_int(e[tfk]); break
                        t_time = None
                        for ttk in ("time", "contact_time", "t"):
                            if ttk in e:
                                t_time = _as_float(e[ttk]); break

                        _push(a, b, t_frame, t_time)
            for v in d.values():
                _walk(v)
        elif isinstance(d, list):
            for v in d:
                _walk(v)

    _walk(meta)
    return out



def _objects_table(meta: dict) -> Dict[int, Dict]:
    """
    obj_id -> {
        'attrs': {'color':..,'shape':..,'material':..},
        'first': first_frame_idx,
        'last' : last_frame_idx
    }
    """
    first, last, total_frames, _ = _first_last_frames(meta)

    # collect every plausible object dict anywhere
    objs = []
    for d in _walk_dicts(meta):
        # direct attr dicts
        if any(k in d for k in ("color","shape","material")):
            objs.append(d); continue
        # objects containers
        for k in ("objects","objs"):
            if isinstance(d.get(k), list):
                for o in d[k]:
                    if isinstance(o, dict):
                        objs.append(o)

    tbl: Dict[int, Dict] = {}
    for o in objs:
        # id
        oid = None
        for k in ("object_id","id","idx","instance_id"):
            if k in o:
                try: oid = int(o[k]); break
                except: pass
        if oid is None:
            continue

        attrs = {
            "color":    _normalize_attr(_get_attr_any(o,"color"),    "color"),
            "shape":    _normalize_attr(_get_attr_any(o,"shape"),    "shape"),
            "material": _normalize_attr(_get_attr_any(o,"material"), "material"),
        }
        if oid not in tbl:
            tbl[oid] = {"attrs": attrs,
                        "first": first.get(oid, 0),
                        "last":  last.get(oid, max(total_frames - 1, 0))}
        else:
            for k,v in attrs.items():
                if not tbl[oid]["attrs"].get(k):
                    tbl[oid]["attrs"][k] = v
            tbl[oid]["first"] = min(tbl[oid]["first"], first.get(oid, tbl[oid]["first"]))
            tbl[oid]["last"]  = max(tbl[oid]["last"],  last.get(oid,  tbl[oid]["last"]))
    return tbl



def _collisions(ann: dict):
    """
    Extract collisions as a list of (obj_a, obj_b, frame_int).
    Accepts 'collision' (singular) or 'collisions' (plural) arrays in CLEVRER JSON.
    Reads any of: frame, frame_id, time, t, contact_time. Falls back to a large value.
    Accepts object id pairs from: 'objects', 'object_ids', 'pair', 'object_pair'.
    """
    out = []
    cols = ann.get("collision") or ann.get("collisions") or []
    if not isinstance(cols, list):
        return out

    for c in cols:
        if not isinstance(c, dict):
            continue
        ids = (c.get("objects") or c.get("object_ids") or c.get("pair") or c.get("object_pair") or [])
        if not isinstance(ids, (list, tuple)) or len(ids) < 2:
            continue
        a, b = int(ids[0]), int(ids[1])

        # robust time read
        t = c.get("frame")
        if t is None: t = c.get("frame_id")
        if t is None: t = c.get("time")
        if t is None: t = c.get("t")
        if t is None: t = c.get("contact_time")
        if t is None: t = 10**9

        try:
            t = int(t)
        except Exception:
            # best effort: non-integer? convert or fallback
            try:
                t = int(float(t))
            except Exception:
                t = 10**9

        out.append((a, b, t))
    return out


def _video_duration_seconds(ann: dict, default_fps: int = 25):
    """
    Try to get duration in seconds; fallback to len(frames)/fps.
    """
    meta = ann.get("metadata") or ann.get("video_metadata") or {}
    dur = meta.get("video_duration") or meta.get("duration") or None
    if isinstance(dur, (int, float)) and dur > 0: return float(dur)
    frames = _get_frames(ann)
    if isinstance(frames, list) and len(frames) > 0:
        return len(frames) / float(default_fps)
    return None  # unknown


def _any_enter(tbl: Dict[int,Dict], flt: Dict[str,str], total: int) -> bool:
    # "enter" = first_seen > 0 (appears after first frame)
    for oid, info in tbl.items():
        if _match_filter(info["attrs"], flt) and info["first"] > 0:
            return True
    return False

def _any_exit(tbl: Dict[int,Dict], flt: Dict[str,str], total: int) -> bool:
    # "exit" = last_seen < last frame
    for oid, info in tbl.items():
        if _match_filter(info["attrs"], flt) and info["last"] < (total - 1):
            return True
    return False

def _any_collision_pair(tbl: Dict[int,Dict], collisions: List[Tuple[int,int,int]],
                        A: Dict[str,str], B: Dict[str,str]) -> Optional[int]:
    # returns the earliest collision frame if any match; else None
    tmin = None
    for (i,j,t) in collisions:
        ai = tbl.get(i); aj = tbl.get(j)
        if ai is None or aj is None: continue
        ok = (_match_filter(ai["attrs"], A) and _match_filter(aj["attrs"], B)) \
             or (_match_filter(ai["attrs"], B) and _match_filter(aj["attrs"], A))
        if ok:
            tmin = t if tmin is None else min(tmin, t)
    return tmin

def _any_entered_then_collided(tbl: Dict[int,Dict], collisions: List[Tuple[int,int,int]],
                               A: Dict[str,str], B: Dict[str,str]) -> bool:
    # ∃ i∈A, j∈B s.t. first(i) < t_collision(i,j)
    for (i,j,t) in collisions:
        ai = tbl.get(i); aj = tbl.get(j)
        if ai is None or aj is None: continue
        if _match_filter(ai["attrs"], A) and _match_filter(aj["attrs"], B):
            if ai["first"] < t: return True
        if _match_filter(ai["attrs"], B) and _match_filter(aj["attrs"], A):
            if aj["first"] < t: return True
    return False

# --- CLEVRER helpers: properties, visibility, and collisions -----------------
import re
from typing import Dict, List, Tuple, Iterable

def _id_maps_from_ann(ann) -> Tuple[Dict[int, str], Dict[int, str], Dict[int, str]]:
    """
    Returns three maps: id2shape, id2color, id2material from CLEVRER
    'object_property' list.
    """
    id2shape, id2color, id2mat = {}, {}, {}
    for p in ann.get("object_property", []):
        oid = p["object_id"]
        id2shape[oid] = p.get("shape")
        id2color[oid] = p.get("color")
        id2mat[oid]   = p.get("material")
    return id2shape, id2color, id2mat

def _first_last_visible_from_ann(ann) -> Tuple[Dict[int, int], Dict[int, int], int]:
    """
    Walks 'motion_trajectory' (list of frames). Each frame has 'objects':
      [{'object_id': int, ..., 'inside_camera_view': bool}, ...]
    Returns (first_visible, last_visible, T_frames).
    If an object never visible: it won't be in the dicts.
    """
    first_visible, last_visible = {}, {}
    frames = ann.get("motion_trajectory", [])
    T = len(frames)
    for t, fr in enumerate(frames):
        for obj in fr.get("objects", []):
            oid = obj["object_id"]
            vis = bool(obj.get("inside_camera_view", True))  # default True for safety
            if vis:
                if oid not in first_visible:
                    first_visible[oid] = t
                last_visible[oid] = t
    return first_visible, last_visible, T

def _iter_collisions(ann) -> Iterable[Tuple[int, int, int]]:
    """
    Yields (a_id, b_id, frame_id) for each collision in 'collision' list.
    Each entry has {'object_ids':[i,j], 'frame_id': int, ...}.
    """
    for ev in ann.get("collision", []):
        obj_ids = ev.get("object_ids", [])
        if len(obj_ids) == 2:
            yield int(obj_ids[0]), int(obj_ids[1]), int(ev.get("frame_id", 0))

# --- tiny parsers for concept names like:
#     "enter(shape=sphere)", "exit(color=red)",
#     "collide(shape=sphere,shape=cube)",
#     "collide_before_half(shape=sphere,color=red)",
#     "entered_then_collided(shape=sphere,shape=cylinder)"
# ------------------------------------------------------------------------------

_filter_re = re.compile(r"(shape|color|material)=(\w+)")
_twoterm_re = re.compile(
    r"(?P<kind>collide|collide_before_half|entered_then_collided)"
    r"\((?P<f1>(?:shape|color|material)=\w+),(?P<f2>(?:shape|color|material)=\w+)\)"
)

def _parse_unary(concept: str) -> Tuple[str, Tuple[str, str]]:
    # e.g. "enter(shape=cube)" -> ("enter", ("shape","cube"))
    m = re.match(r"(enter|exit)\(([^)]+)\)", concept)
    if not m: 
        return "", ("","")
    kind = m.group(1)
    mf = _filter_re.search(m.group(2))
    if not mf:
        return "", ("","")
    return kind, (mf.group(1), mf.group(2))

def _parse_binary(concept: str) -> Tuple[str, Tuple[Tuple[str,str], Tuple[str,str]]]:
    # e.g. "collide(shape=sphere,shape=cube)" -> ("collide", (("shape","sphere"),("shape","cube")))
    m = _twoterm_re.match(concept)
    if not m:
        return "", (("",""),("",""))
    kind = m.group("kind")
    f1m = _filter_re.match(m.group("f1"))
    f2m = _filter_re.match(m.group("f2"))
    return kind, ((f1m.group(1), f1m.group(2)), (f2m.group(1), f2m.group(2)))

def _matches(obj_id: int,
             flt: Tuple[str,str],
             id2shape: Dict[int,str],
             id2color: Dict[int,str],
             id2mat: Dict[int,str]) -> bool:
    key, val = flt
    if key == "shape":    return id2shape.get(obj_id) == val
    if key == "color":    return id2color.get(obj_id) == val
    if key == "material": return id2mat.get(obj_id) == val
    return False



# -----------------------------
# DSL for compound expressions
# -----------------------------
# op codes per your convention: 1=IFF, 2=IMPLIES, 3=AND, 4=OR
OPC = {"IFF":1, "IMPLIES":2, "AND":3, "OR":4}

def leaf(cid: int):     return ("leaf", cid)
def NOT(expr):          return ("not", expr)
def AND(a, b):          return ("AND", a, b)
def OR(a, b):           return ("OR", a, b)
def IFF(a, b):          return ("IFF", a, b)
def IMPLIES(l, r):      return ("IMPLIES", l, r)

# --- NEW (mine attribute concepts from questions) ---
def mine_attribute_concepts_from_questions(q_json_path: str,
                                           max_extra:int=6) -> List[Tuple[str,Dict[str,str]]]:
    """
    Very light miner: look for tokens that match known attribute vocab and add EXISTS(attribute=value) concepts.
    Labels for these will still be computed from the annotation objects (not from questions).
    """
    if not q_json_path or not os.path.isfile(q_json_path):
        return []
    with open(q_json_path, "r") as f:
        try:
            data = json.load(f)
        except Exception:
            return []

    vocab_colors = {"gray","blue","brown","yellow","red","green","purple","cyan"}
    vocab_shapes = {"sphere","cube","cylinder"}
    vocab_mats   = {"rubber","metal"}

    cnt: Dict[Tuple[str,str], int] = {}
    def inc(kv): cnt[kv] = cnt.get(kv,0) + 1

    # Format varies: we only need raw text
    for q in data if isinstance(data, list) else data.get("questions", []):
        s = (q.get("question","") or q.get("text","")).lower()
        for w in re.findall(r"[a-z]+", s):
            if w in vocab_colors: inc(("color", w))
            if w in vocab_shapes: inc(("shape", w))
            if w in vocab_mats:   inc(("material", w))

    # sort by frequency, skip ones already in CONCEPTS
    existing = set()
    for _,flt in CONCEPTS:
        for k,v in flt.items(): existing.add((k,v))
    out = []
    for (k,v), n in sorted(cnt.items(), key=lambda x: -x[1]):
        if (k,v) in existing: continue
        name = f"exists_{k}_{v}"
        out.append((name, {k:v}))
        if len(out) >= max_extra: break
    return out

# --- Robust normalization of CLEVRER annotations (ids, nesting, attribute coding) ---

# ---- CLEVRER attribute vocab ----
_COLOR_MAP = {0:"gray",1:"red",2:"blue",3:"green",4:"brown",5:"purple",6:"cyan",7:"yellow"}
_SHAPE_MAP = {0:"cube",1:"sphere",2:"cylinder"}
_MAT_MAP   = {0:"rubber",1:"metal"}

def _normalize_attr(val, kind: str) -> str:
    """Map ints used by CLEVRER to strings; pass through strings; lowercase always."""
    if isinstance(val, (int, float)):
        v = int(val)
        if kind == "color":    return _COLOR_MAP.get(v, str(v)).lower()
        if kind == "shape":    return _SHAPE_MAP.get(v, str(v)).lower()
        if kind == "material": return _MAT_MAP.get(v, str(v)).lower()
    # sometimes meta stores booleans/strings like "metal"/"rubber"
    return str(val).lower()

def _walk_dicts(meta):
    """Yield every dict in the JSON tree."""
    if isinstance(meta, dict):
        yield meta
        for v in meta.values():
            yield from _walk_dicts(v)
    elif isinstance(meta, list):
        for v in meta:
            yield from _walk_dicts(v)

def _collect_objects(meta: dict):
    """
    Collect plausible object dicts anywhere in the annotation JSON.
    Any dict that either:
      - appears inside a list keyed 'objects', or
      - has any of the keys {'color','shape','material'} (possibly nested)
    is considered a candidate object.
    """
    objs = []
    for d in _walk_dicts(meta):
        # direct candidates
        if any(k in d for k in ("color", "shape", "material")):
            objs.append(d); continue
        # inside 'objects': [...]
        for k in ("objects", "objs"):
            if isinstance(d.get(k), list):
                for o in d[k]:
                    if isinstance(o, dict):
                        objs.append(o)
    return objs

def _collect_events(meta: dict):
    """
    Return a flat list of event dicts with at least:
      - 'type' in {'collision','enter','exit'}
      - normalized time key 'frame' if a 'frame_id' field exists.
    We scan *both* 'collision' (singular) and 'collisions' (plural), plus 'events'.
    """
    evs = []
    for d in _walk_dicts(meta):
        # list-valued containers
        for key in ("collision", "collisions", "events"):
            lst = d.get(key)
            if isinstance(lst, list):
                for e in lst:
                    if not isinstance(e, dict):
                        continue
                    typ = (e.get("type") or e.get("label") or "").strip().lower()
                    if key in ("collision", "collisions") and not typ:
                        typ = "collision"  # CLEVRER often omits explicit type inside these lists
                    if typ in ("collision", "enter", "exit"):
                        ee = {"type": typ, **e}
                        # normalize time field
                        if "frame_id" in ee and "frame" not in ee:
                            ee["frame"] = ee["frame_id"]
                        evs.append(ee)

        # singleton dicts (inline events)
        typ = (d.get("type") or d.get("label") or "").strip().lower()
        if typ in ("collision", "enter", "exit"):
            ee = {"type": typ, **d}
            if "frame_id" in ee and "frame" not in ee:
                ee["frame"] = ee["frame_id"]
            evs.append(ee)

    return evs

def _get_attr_any(o: dict, k: str):
    """
    Try several common nests for an attribute (color/shape/material):
    direct, then through {attributes, attr, state, props, object, obj}.
    """
    if k in o:
        return o[k]
    for nest in ("attributes", "attr", "state", "props", "object", "obj"):
        v = o.get(nest)
        if isinstance(v, dict) and (k in v):
            return v[k]
    return None

def _norm_collision(e):
    """
    Accepts either:
      - dicts like {'a':id1,'b':id2,'t_frame':int|None,'t_time':float|None}
      - tuples/lists like (a,b,t) or (a,b,t_frame,t_time)
    Returns a normalized dict with keys: a, b, t_frame, t_time.
    """
    if isinstance(e, dict):
        a = e.get('a') or e.get('A') or e.get('obj_a') or e.get('id1') or e.get('o1')
        b = e.get('b') or e.get('B') or e.get('obj_b') or e.get('id2') or e.get('o2')
        # prefer explicit keys if present
        tf = e.get('t_frame') or e.get('frame') or e.get('frame_id') or e.get('contact_frame')
        ts = e.get('t_time')  or e.get('time')  or e.get('contact_time') or e.get('t')
        try: a = int(a)
        except: a = None
        try: b = int(b)
        except: b = None
        try: tf = int(tf) if tf is not None else None
        except: tf = None
        try: ts = float(ts) if ts is not None else None
        except: ts = None
        return {'a': a, 'b': b, 't_frame': tf, 't_time': ts}

    if isinstance(e, (list, tuple)) and len(e) >= 2:
        try: a = int(e[0])
        except: a = None
        try: b = int(e[1])
        except: b = None
        tf, ts = None, None
        if len(e) >= 3 and e[2] is not None:
            # If it looks integer-ish treat as frame, else seconds
            if isinstance(e[2], (int,)) or (isinstance(e[2], float) and e[2].is_integer()):
                tf = int(e[2])
            else:
                try: ts = float(e[2])
                except: ts = None
        if len(e) >= 4 and e[3] is not None:
            try: ts = float(e[3])
            except: pass
        return {'a': a, 'b': b, 't_frame': tf, 't_time': ts}

    return {'a': None, 'b': None, 't_frame': None, 't_time': None}


def concept_vector_from_annotation(ann, CONCEPTS: List[Tuple[str,int]]) -> torch.Tensor:
    """
    Builds a 0/1 vector y over your CONCEPTS list.
    CONCEPTS is a list of (name, idx) where names can be:
      - static: 'blue_sphere', 'red_sphere', 'green_cube', 'yellow_cyl', 'metal_any', 'gray_cyl'
      - events: 'enter(shape=...)' / 'exit(color=...)'
                'collide(shape=...,shape=...)'
                'collide_before_half(shape=...,color=...)'
                'entered_then_collided(shape=...,shape=...)'
    """
    import torch
    id2shape, id2color, id2mat = _id_maps_from_ann(ann)
    first_vis, last_vis, T = _first_last_visible_from_ann(ann)
    y = torch.zeros(len(CONCEPTS), dtype=torch.long)

    # quick static availability flags
    any_shape = set(id2shape.values())
    any_color = set(id2color.values())
    any_mat   = set(id2mat.values())

    # pre-index which ids match each primitive filter used in events
    # to avoid recomputing
    def ids_matching(flt):
        return [oid for oid in id2shape.keys() if _matches(oid, flt, id2shape, id2color, id2mat)]

    # cache collision list once
    collisions = list(_iter_collisions(ann))  # [(a,b,t),...]
    # also build an index per pair for speed
    # map (min(a,b), max(a,b)) -> list of times
    from collections import defaultdict
    coll_idx = defaultdict(list)
    for a,b,t in collisions:
        k = (a,b) if a < b else (b,a)
        coll_idx[k].append(t)

    name_to_pos = {nm:i for i,(nm,_) in enumerate(CONCEPTS)}

    for pos, (nm, _) in enumerate(CONCEPTS):
        # 1) static six
        if nm == "blue_sphere":
            y[pos] = int(("sphere" in any_shape) and ("blue" in any_color))
            continue
        if nm == "red_sphere":
            y[pos] = int(("sphere" in any_shape) and ("red" in any_color))
            continue
        if nm == "green_cube":
            y[pos] = int(("cube" in any_shape) and ("green" in any_color))
            continue
        if nm == "yellow_cyl":
            y[pos] = int(("cylinder" in any_shape) and ("yellow" in any_color))
            continue
        if nm == "gray_cyl":
            y[pos] = int(("cylinder" in any_shape) and ("gray" in any_color))
            continue
        if nm == "metal_any":
            y[pos] = int(("metal" in any_mat))
            continue

        # 2) unary events: enter/exit
        kind, flt = _parse_unary(nm)
        if kind:
            ids = ids_matching(flt)
            if kind == "enter":
                # first time visible > 0 for any matching id
                flag = any((oid in first_vis and first_vis[oid] > 0) for oid in ids)
                y[pos] = int(flag)
            elif kind == "exit":
                # last visible < T-1 for any matching id
                flag = any((oid in last_vis and last_vis[oid] < (T-1)) for oid in ids)
                y[pos] = int(flag)
            continue

        # 3) binary events
        kind2, (flt1, flt2) = _parse_binary(nm)
        if not kind2:
            # name not recognized → leave 0
            continue

        ids1 = ids_matching(flt1)
        ids2 = ids_matching(flt2)
        if not ids1 or not ids2:
            y[pos] = 0
            continue

        if kind2 == "collide" or kind2 == "collide_before_half" or kind2 == "entered_then_collided":
            found = False
            half_t = (T // 2)
            for a in ids1:
                for b in ids2:
                    if a == b: 
                        continue
                    k = (a,b) if a < b else (b,a)
                    times = coll_idx.get(k, [])
                    if not times:
                        continue
                    if kind2 == "collide":
                        found = True
                        break
                    elif kind2 == "collide_before_half":
                        if any(t < half_t for t in times):
                            found = True
                            break
                    elif kind2 == "entered_then_collided":
                        # at least one object first visible after 0, and a collision happened
                        ent_a = (a in first_vis and first_vis[a] > 0)
                        ent_b = (b in first_vis and first_vis[b] > 0)
                        if (ent_a or ent_b):
                            found = True
                            break
                if found: 
                    break
            y[pos] = int(found)
            continue

        # default: unrecognized → 0
        y[pos] = 0

    return y




def _add_expr(expr, nodes, edges):
    """
    Recursively add nodes/edges to build a DGL-ready structure.
    nodes: list of dicts with keys {mask, y, x}
    edges: list of (src, dst, neg, pos)
    Returns: node_id (int)
    """
    kind = expr[0]
    if kind == "leaf":
        cid = int(expr[1])
        nid = len(nodes)
        nodes.append({"mask":1, "y":0, "x":cid})
        return nid

    if kind == "not":
        # represent NOT as an edge-level negation on the child connection at the PARENT call
        # so here we just return the child's node id and let the parent set neg = -1
        return _add_expr(expr[1], nodes, edges)

    # binary ops
    op_name, a, b = expr
    left  = _add_expr(a, nodes, edges)
    right = _add_expr(b, nodes, edges)
    nid = len(nodes)
    nodes.append({"mask":0, "y":OPC[op_name], "x":0})

    # Determine neg flags for children (if the child exprs are NOT(...))
    def _is_not(e): return isinstance(e, tuple) and len(e)>0 and e[0]=="not"
    neg_l = -1 if _is_not(a) else +1
    neg_r = -1 if _is_not(b) else +1

    # Edge pos: only meaningful for IMPLIES (left=0, right=1)
    if op_name == "IMPLIES":
        edges.append((left,  nid, neg_l, 0))
        edges.append((right, nid, neg_r, 1))
    else:
        edges.append((left,  nid, neg_l, 0))
        edges.append((right, nid, neg_r, 1))
    return nid

def build_graph_from_expr(expr) -> dgl.DGLGraph:
    nodes, edges = [], []
    root = _add_expr(expr, nodes, edges)
    g = dgl.graph(([], []), num_nodes=len(nodes))
    # set node data
    g.ndata["mask"] = torch.tensor([n["mask"] for n in nodes], dtype=torch.long)
    g.ndata["y"]    = torch.tensor([n["y"] for n in nodes], dtype=torch.long)
    g.ndata["x"]    = torch.tensor([n["x"] for n in nodes], dtype=torch.long)
    # add edges
    if edges:
        src = torch.tensor([s for (s,_,_,_) in edges], dtype=torch.long)
        dst = torch.tensor([d for (_,d,_,_) in edges], dtype=torch.long)
        g.add_edges(src, dst)
        g.edata["neg"] = torch.tensor([neg for (_,_,neg,_) in edges], dtype=torch.long)
        g.edata["pos"] = torch.tensor([pos for (_,_,_,pos) in edges], dtype=torch.long)
    return g

# -----------------------------
# Two-level COMPOUND rules (made from your concepts)
# -----------------------------
# ================================
# PATCH A: build rules by names
# ================================
from typing import Sequence

def build_seed_and_compound_rules_by_name(concept_names: Sequence[str]):
    """
    Returns:
      rules_simple: List[RuleSpec]
      compound_named_exprs: List[Tuple[str, tuple-expr]]
    where concept ids are resolved from concept_names at runtime, so they always match.
    """
    name_to_cid = {nm: i+1 for i, nm in enumerate(concept_names)}  # 1-based CIDs

    def cid(nm: str) -> int:
        if nm not in name_to_cid:
            raise KeyError(f"Concept '{nm}' not in concept_names={list(concept_names)}")
        return name_to_cid[nm]

    # ----- Simple (depth-1) rules (static) -----
    rules_simple = [
        RuleSpec("IMPLIES_NOT", cid("blue_sphere"), cid("red_sphere"), "blue_sphere -> not red_sphere"),
        RuleSpec("IMPLIES",     cid("green_cube"),  cid("metal_any"),  "green_cube -> metal_any"),
        RuleSpec("IMPLIES_NOT", cid("yellow_cyl"),  cid("gray_cyl"),   "yellow_cyl -> not gray_cyl"),
    ]

    # ----- Compound (depth-2) rules (static) -----
    compound_named_exprs: List[Tuple[str, tuple]] = [
        ("(blue_sphere ∧ green_cube) → metal_any",
         IMPLIES( AND(leaf(cid("blue_sphere")), leaf(cid("green_cube"))),
                  leaf(cid("metal_any")) )),
        ("yellow_cyl → (¬gray_cyl ∨ red_sphere)",
         IMPLIES( leaf(cid("yellow_cyl")),
                  OR(NOT(leaf(cid("gray_cyl"))), leaf(cid("red_sphere"))) )),
        ("green_cube → (blue_sphere ↔ ¬red_sphere)",
         IMPLIES( leaf(cid("green_cube")),
                  IFF(leaf(cid("blue_sphere")), NOT(leaf(cid("red_sphere"))) ))),
    ]

    # ----- NEW: add event-derived rules if event concepts are present -----
    # (entered_then_collided/ collide_before_half) -> collide
    try:
        event_rules = _build_event_rules_by_name(list(concept_names))
        # de-dup by name
        seen = set(nm for nm, _ in compound_named_exprs)
        for nm, expr in event_rules:
            if nm not in seen:
                compound_named_exprs.append((nm, expr))
                seen.add(nm)
    except Exception as e:
        print(f"[WARN] event-rule wiring skipped: {e}")

    return rules_simple, compound_named_exprs


# --- NEW: event-rule builder (auto-wires event concepts into rules) ---
# --- REPLACE the whole function with this inverted version ---
import re as _re

def _build_event_rules_by_name(concept_names: List[str]) -> List[Tuple[str, tuple]]:
    """
    Build *inverted* event rules to avoid tautologies:

        collide(A,B) -> collide_before_half(A,B)        [inv]
        collide(A,B) -> entered_then_collided(A,B)      [inv]

    Only wires rules when BOTH antecedent and consequent concepts exist in the
    current concept list. Names are tagged with [inv] to make the direction clear.
    """
    name_to_cid = {nm: i+1 for i, nm in enumerate(concept_names)}
    def has(nm: str) -> bool: return nm in name_to_cid
    def cid(nm: str) -> int:  return name_to_cid[nm]

    out: List[Tuple[str, tuple]] = []

    # find all collide( … ) concepts and point them to the stronger variants
    pat_pair = _re.compile(r"^collide\((.+?),\s*(.+?)\)$")
    for nm in concept_names:
        m = pat_pair.match(nm)
        if not m:
            continue
        A, B = m.group(1), m.group(2)

        # stronger variants that may or may not be present
        c_before = f"collide_before_half({A},{B})"
        c_enter  = f"entered_then_collided({A},{B})"

        # Inverted rule: collide(A,B) -> collide_before_half(A,B)
        if has(nm) and has(c_before):
            out.append((
                f"{nm} → {c_before} [inv]",
                IMPLIES(leaf(cid(nm)), leaf(cid(c_before)))
            ))

        # Inverted rule: collide(A,B) -> entered_then_collided(A,B)
        if has(nm) and has(c_enter):
            out.append((
                f"{nm} → {c_enter} [inv]",
                IMPLIES(leaf(cid(nm)), leaf(cid(c_enter)))
            ))

    return out


# ==============================
# CLEVRER: hard-coded + mined compound rules (no static rules)
# ==============================

_HARDCODED_RULE_SPECS = [
    # 6 chain rules
    ("[chain] (collide(shape=sphere,color=brown)→collide_before_half(shape=sphere,color=brown))∧(collide_before_half(shape=sphere,color=brown)→entered_then_collided(shape=sphere,color=brown))",
     ("collide(shape=sphere,color=brown)", "collide_before_half(shape=sphere,color=brown)", "entered_then_collided(shape=sphere,color=brown)")),
    ("[chain] (collide(shape=cube,color=cyan)→collide_before_half(shape=cube,color=cyan))∧(collide_before_half(shape=cube,color=cyan)→entered_then_collided(shape=cube,color=cyan))",
     ("collide(shape=cube,color=cyan)", "collide_before_half(shape=cube,color=cyan)", "entered_then_collided(shape=cube,color=cyan)")),
    ("[chain] (collide(shape=sphere,color=green)→entered_then_collided(shape=sphere,color=green))∧(entered_then_collided(shape=sphere,color=green)→collide_before_half(shape=sphere,color=green))",
     ("collide(shape=sphere,color=green)", "entered_then_collided(shape=sphere,color=green)", "collide_before_half(shape=sphere,color=green)")),
    ("[chain] (collide(shape=cube,color=blue)→entered_then_collided(shape=cube,color=blue))∧(entered_then_collided(shape=cube,color=blue)→collide_before_half(shape=cube,color=blue))",
     ("collide(shape=cube,color=blue)", "entered_then_collided(shape=cube,color=blue)", "collide_before_half(shape=cube,color=blue)")),
     

    # ("[chain] (enter(color=blue)→collide_before_half(shape=sphere,color=brown))∧(collide_before_half(shape=sphere,color=brown)→exit(color=green))",
    #  ("enter(color=blue)", "collide_before_half(shape=sphere,color=brown)", "exit(color=green)")),
    # ("[chain] (enter(color=cyan)→collide_before_half(shape=cube,color=blue))∧(collide_before_half(shape=cube,color=blue)→exit(shape=sphere))",
    #  ("enter(color=cyan)", "collide_before_half(shape=cube,color=blue)", "exit(shape=sphere)")),
    # ("[chain] (enter(color=green)→collide_before_half(shape=cylinder,color=yellow))∧(collide_before_half(shape=cylinder,color=yellow)→exit(shape=sphere))",
    #  ("enter(color=green)", "collide_before_half(shape=cylinder,color=yellow)", "exit(shape=sphere)")),

    # # 3 "not-iff" comparisons: we encode IFF(chain1, chain2) so that anomaly=1-truth fires on NOT-IFF.
    # ("[notiff] chain(blue→collide(cube,brown)→exit(green)) vs chain(blue→collide(sphere,brown)→exit(green))",
    #  ("enter(color=blue)", "collide(shape=cube,color=brown)",   "exit(color=green)",
    #   "enter(color=blue)", "collide(shape=sphere,color=brown)", "exit(color=green)")),
    # ("[notiff] chain(blue→collide(cube,yellow)→exit(brown)) vs chain(blue→collide(cube,brown)→exit(brown))",
    #  ("enter(color=blue)", "collide(shape=cube,color=yellow)",  "exit(color=brown)",
    #   "enter(color=blue)", "collide(shape=cube,color=brown)",   "exit(color=brown)")),
    # ("[notiff] chain(blue→collide(cylinder,cyan)→exit(brown)) vs chain(blue→collide(cube,brown)→exit(brown))",
    #  ("enter(color=blue)", "collide(shape=cylinder,color=cyan)","exit(color=brown)",
    #   "enter(color=blue)", "collide(shape=cube,color=brown)",   "exit(color=brown)")),
]


def _required_event_pairs_for_hardcoded_rules() -> List[Tuple[str,str]]:
    """
    Return unordered pairs of primitive filter strings used by hard-coded collision atoms,
    e.g. ("shape=sphere","color=brown").
    """
    pairs = set()
    for name, spec in _HARDCODED_RULE_SPECS:
        # chain: (enter, collide*, exit)
        if len(spec) == 3:
            _, col, _ = spec
            km, (a,b) = _parse_binary(col)
            if km:
                f1 = f"{a[0]}={a[1]}"; f2 = f"{b[0]}={b[1]}"
                pairs.add(tuple(sorted((f1,f2))))
        # notiff: two chains packed (enter,col,exit, enter,col,exit)
        elif len(spec) == 6:
            _, col1, _, _, col2, _ = spec
            for col in (col1, col2):
                km, (a,b) = _parse_binary(col)
                if km:
                    f1 = f"{a[0]}={a[1]}"; f2 = f"{b[0]}={b[1]}"
                    pairs.add(tuple(sorted((f1,f2))))
    return sorted(pairs)

def _concept_cid_map(concept_names: Sequence[str]) -> Dict[str,int]:
    # concept ids are 1-based (to match your evaluator/trainer conventions)
    return {nm: i+1 for i, nm in enumerate(concept_names)}

def _cid(name_to_cid: Dict[str,int], nm: str) -> int:
    """Return concept id for a concept name.

    CLEVRER collision concepts are symmetric in the annotations (object-id pairs),
    but the derived concept-name builder may emit either endpoint ordering.
    Hard-coded rules should therefore accept either ordering.
    """
    if nm in name_to_cid:
        return name_to_cid[nm]

    # Try swapped endpoint order for symmetric binary events
    m = re.match(r"^(collide|collide_before_half|entered_then_collided)\(([^,]+),([^\)]+)\)$", nm)
    if m:
        kind, a, b = m.group(1), m.group(2).strip(), m.group(3).strip()
        swapped = f"{kind}({b},{a})"
        if swapped in name_to_cid:
            return name_to_cid[swapped]

    raise KeyError(
        f"Required concept '{nm}' is missing (and swapped form not found). "
        f"Make sure --event_rich is on and --event_filters includes it, "
        f"and that the needed pair is included in event pairs."
    )

def _chain_expr(name_to_cid: Dict[str,int], enter_nm: str, collide_nm: str, exit_nm: str) -> tuple:
    return AND(
        IMPLIES(leaf(_cid(name_to_cid, enter_nm)),  leaf(_cid(name_to_cid, collide_nm))),
        IMPLIES(leaf(_cid(name_to_cid, collide_nm)), leaf(_cid(name_to_cid, exit_nm))),
    )

def build_clevrer_hardcoded_rule_exprs(concept_names: Sequence[str]) -> List[Tuple[str, tuple]]:
    """
    Build the exact hard-coded rule expressions requested by the user.
    For the "[notiff]" items, we return IFF(chain1, chain2) so that anomaly=1-truth is triggered on NOT-IFF.
    """
    name_to_cid = _concept_cid_map(concept_names)
    out: List[Tuple[str, tuple]] = []
    for rname, spec in _HARDCODED_RULE_SPECS:
        if len(spec) == 3:
            en, col, ex = spec
            out.append((rname, _chain_expr(name_to_cid, en, col, ex)))
        elif len(spec) == 6:
            en1, col1, ex1, en2, col2, ex2 = spec
            ch1 = _chain_expr(name_to_cid, en1, col1, ex1)
            ch2 = _chain_expr(name_to_cid, en2, col2, ex2)
            out.append((rname, IFF(ch1, ch2)))
        else:
            raise ValueError("Bad hard-coded rule spec length")
    return out

# ==============================
# NEW: symmetric pair-name resolver (collide / collide_before_half / entered_then_collided)
# ==============================

def _pair_name_present(concept_set: set, kind: str, a: str, b: str) -> Optional[str]:
    """
    Return the concept name present in concept_set for a symmetric binary event.
    Tries both endpoint orders:
      f"{kind}({a},{b})" and f"{kind}({b},{a})"
    """
    nm1 = f"{kind}({a},{b})"
    nm2 = f"{kind}({b},{a})"
    if nm1 in concept_set: return nm1
    if nm2 in concept_set: return nm2
    return None

# Backwards-compatible wrapper (some older code assumes this exists)
def _collide_name_present(concept_set: set, a: str, b: str) -> Optional[str]:
    return _pair_name_present(concept_set, "collide", a, b)

def mine_enter_implies_two_collides(ann_path: str,
                                   concept_list: List[Tuple[str,Dict]],
                                   concept_names: Sequence[str],
                                   scan_limit: int = 2000,
                                   min_hits: int = 5,
                                   min_conf: float = 0.20,
                                   min_viol: int = 1,
                                   max_rules: int = 64) -> List[Tuple[str, tuple]]:
    """
    (Repurposed miner.)

    Mine CLEVRER event-structure compound rules (from annotations, via concept_vector_from_annotation):

      R1: collide(A,B) -> ( collide_before_half(A,B) AND entered_then_collided(A,B) )
      R2: collide(A,B) -> ( collide_before_half(A,B) OR  entered_then_collided(A,B) )
      R3: collide(A,B) -> ( collide_before_half(A,B) -> entered_then_collided(A,B) )

    Threshold semantics (kept compatible with the old miner’s knobs):
      - antecedent support: ent = #(collide(A,B)=1)
      - hits:              #(collide(A,B)=1 AND consequent=1)
      - conf:              hits / ent
      - viol:              ent - hits   (i.e., #(collide=1 AND consequent=0))

    We keep at most `max_rules` TOTAL across the three families, roughly split evenly.
    """
    import glob, json
    from collections import defaultdict

    # list annotation files
    files: List[str] = []
    if os.path.isdir(ann_path):
        files = sorted(glob.glob(os.path.join(ann_path, "**", "annotation_*.json"), recursive=True))
    elif os.path.isfile(ann_path):
        files = [ann_path]
    if scan_limit and scan_limit > 0:
        files = files[:scan_limit]

    concept_set = set(concept_names)
    name_to_idx = {nm: i for i, nm in enumerate(concept_names)}  # 0-based for y

    # collect candidate collide(A,B) that also have BOTH stronger variants present
    collide_names = [
        nm for nm in concept_names
        if nm.startswith("collide(") and (not nm.startswith("collide_before_half("))
    ]

    pair_of: Dict[str, Tuple[str, str]] = {}  # collide_nm -> (before_nm, entered_nm)
    cand_idx: List[Tuple[str, int, str, int, str, int]] = []

    for nm in collide_names:
        km, (f1, f2) = _parse_binary(nm)
        if km != "collide":
            continue

        a = f"{f1[0]}={f1[1]}"
        b = f"{f2[0]}={f2[1]}"

        nm_before = _pair_name_present(concept_set, "collide_before_half", a, b)
        nm_enter  = _pair_name_present(concept_set, "entered_then_collided", a, b)
        if (nm_before is None) or (nm_enter is None):
            continue

        # must exist in name_to_idx to be measurable in y
        if (nm not in name_to_idx) or (nm_before not in name_to_idx) or (nm_enter not in name_to_idx):
            continue

        pair_of[nm] = (nm_before, nm_enter)
        cand_idx.append((nm, name_to_idx[nm], nm_before, name_to_idx[nm_before], nm_enter, name_to_idx[nm_enter]))

    if not cand_idx:
        return []

    # counts keyed by collide concept name (nm)
    cnt_ent = defaultdict(int)
    cnt_and = defaultdict(int)
    cnt_or  = defaultdict(int)
    cnt_imp = defaultdict(int)

    # scan
    for jf in files:
        try:
            with open(jf, "r") as f:
                ann = json.load(f)
        except Exception:
            continue

        metas = ann if isinstance(ann, list) else [ann]
        for meta in metas:
            y = concept_vector_from_annotation(meta, concept_list)  # (K,)

            for (nm_c, i_c, nm_b, i_b, nm_e, i_e) in cand_idx:
                if int(y[i_c].item()) != 1:
                    continue

                cnt_ent[nm_c] += 1
                b = (int(y[i_b].item()) == 1)
                e = (int(y[i_e].item()) == 1)

                # consequent forms
                if b and e:
                    cnt_and[nm_c] += 1
                if b or e:
                    cnt_or[nm_c] += 1
                if ((not b) or e):
                    cnt_imp[nm_c] += 1

    # build candidates per family
    cand_AND: List[Tuple[float,int,int,str,str,str]] = []
    cand_OR:  List[Tuple[float,int,int,str,str,str]] = []
    cand_IMP: List[Tuple[float,int,int,str,str,str]] = []

    for nm_c, (nm_b, nm_e) in pair_of.items():
        ent = int(cnt_ent.get(nm_c, 0))
        if ent <= 0:
            continue

        # R1
        hits = int(cnt_and.get(nm_c, 0))
        conf = hits / float(ent)
        viol = ent - hits
        if (hits >= min_hits) and (conf >= min_conf) and (viol >= min_viol):
            cand_AND.append((conf, hits, viol, nm_c, nm_b, nm_e))

        # R2
        hits = int(cnt_or.get(nm_c, 0))
        conf = hits / float(ent)
        viol = ent - hits
        if (hits >= min_hits) and (conf >= min_conf) and (viol >= min_viol):
            cand_OR.append((conf, hits, viol, nm_c, nm_b, nm_e))

        # R3
        hits = int(cnt_imp.get(nm_c, 0))
        conf = hits / float(ent)
        viol = ent - hits  # = #(collide=1 AND (before=1 AND entered=0))
        if (hits >= min_hits) and (conf >= min_conf) and (viol >= min_viol):
            cand_IMP.append((conf, hits, viol, nm_c, nm_b, nm_e))

    # sort: highest confidence, then hits, then violations (so we don’t keep ultra-tautological ones)
    cand_AND.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3], t[4], t[5]))
    cand_OR.sort( key=lambda t: (-t[0], -t[1], -t[2], t[3], t[4], t[5]))
    cand_IMP.sort(key=lambda t: (-t[0], -t[1], -t[2], t[3], t[4], t[5]))

    # cap TOTAL across families (roughly even split)
    per = max(1, max_rules // 3)
    kept_AND = cand_AND[:per]
    kept_OR  = cand_OR[:per]
    rem = max_rules - (len(kept_AND) + len(kept_OR))
    kept_IMP = cand_IMP[:max(0, rem)]

    name_to_cid = _concept_cid_map(concept_names)
    out: List[Tuple[str, tuple]] = []

    def _push(kind_tag: str, conf: float, hits: int, viol: int, nm_c: str, nm_b: str, nm_e: str, expr: tuple):
        rname = f"[mineC-{kind_tag}] {nm_c} → {expr_str} | conf={conf:.2f} hits={hits} viol={viol}"
        out.append((rname, expr))

    # emit AND rules
    for conf, hits, viol, nm_c, nm_b, nm_e in kept_AND:
        expr_str = f"({nm_b} ∧ {nm_e})"
        expr = IMPLIES(
            leaf(_cid(name_to_cid, nm_c)),
            AND(leaf(_cid(name_to_cid, nm_b)), leaf(_cid(name_to_cid, nm_e))),
        )
        rname = f"[mineC-AND] {nm_c} → ({nm_b} ∧ {nm_e}) | conf={conf:.2f} hits={hits} viol={viol}"
        out.append((rname, expr))

    # emit OR rules
    for conf, hits, viol, nm_c, nm_b, nm_e in kept_OR:
        expr = IMPLIES(
            leaf(_cid(name_to_cid, nm_c)),
            OR(leaf(_cid(name_to_cid, nm_b)), leaf(_cid(name_to_cid, nm_e))),
        )
        rname = f"[mineC-OR] {nm_c} → ({nm_b} ∨ {nm_e}) | conf={conf:.2f} hits={hits} viol={viol}"
        out.append((rname, expr))

    # emit inner-IMPLIES rules
    for conf, hits, viol, nm_c, nm_b, nm_e in kept_IMP:
        inner = IMPLIES(leaf(_cid(name_to_cid, nm_b)), leaf(_cid(name_to_cid, nm_e)))
        expr = IMPLIES(leaf(_cid(name_to_cid, nm_c)), inner)
        rname = f"[mineC-IMP] {nm_c} → ({nm_b} → {nm_e}) | conf={conf:.2f} hits={hits} viol={viol}"
        out.append((rname, expr))

    return out



# -----------------------------
# Scenes → concept vectors
# -----------------------------
def load_scenes(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        js = json.load(f)
    return {sc["image_filename"]: sc for sc in js["scenes"]}

def concept_vector_from_scene(scene: dict) -> torch.Tensor:
    y = torch.zeros(len(concept_names), dtype=torch.long)
    objs = scene["objects"]
    for i, (_, filt) in enumerate(CONCEPTS):
        match = any(all(o.get(k) == v for k, v in filt.items()) for o in objs)
        y[i] = 1 if match else 0
    return y

# --- NEW (video dataset) ---
class CLEVRERVidDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir: str, ann_path: str,
                 concepts: List[Tuple[str,Dict[str,str]]],
                 frames: int = 16, resize: Tuple[int,int] = (224,224),
                 half_frame_default: int = 64):

        self.video_dir = video_dir
        self.ann_path  = ann_path
        self.frames    = frames
        self.resize    = resize
        self.concepts  = concepts
        self.half_frame_default = half_frame_default

        self.items: List[Tuple[str, Union[str,dict]]] = []  # (video_path, ann_json_path or meta dict)
        self.to_tensor = T.ToTensor()

        # 1) Collect all videos recursively (…/videos/**/video_*.mp4|avi)
        # inside CLEVRERVidDataset.__init__(...)
        vid_files = sorted(
            glob.glob(os.path.join(video_dir, "**", "video_*.mp4"), recursive=True)
            + glob.glob(os.path.join(video_dir, "**", "video_*.avi"), recursive=True)
        )

        if os.path.isdir(ann_path):
            ann_files = glob.glob(os.path.join(ann_path, "**", "annotation_*.json"), recursive=True)
            ann_index: Dict[str, str] = {}
            for jf in ann_files:
                # was: r"annotation_(\d{5})\.json$"
                m = re.search(r"annotation_(\d+)\.json$", os.path.basename(jf))
                if m:
                    ann_index[m.group(1).lstrip("0")] = jf

            matched = 0
            for vp in vid_files:
                # was: r"video_(\d{5})\.(mp4|avi)$"
                m = re.search(r"video_(\d+)\.(mp4|avi)$", os.path.basename(vp))
                if not m: continue
                vid = m.group(1).lstrip("0")  # normalize both to no-leading-zeros
                jf = ann_index.get(vid)
                if jf:
                    self.items.append((vp, jf))
                    matched += 1

            print(f"[CLEVRER] videos(recursive): {len(vid_files)} | ann(recursive): {len(ann_files)} | matched pairs: {matched}")

        elif os.path.isfile(ann_path):
            with open(ann_path, "r") as f:
                js = json.load(f)
            entries = js if isinstance(js, list) else (js.get("videos") or js.get("scenes") or [])
            entry_by_id: Dict[str, dict] = {}
            for v in entries:
                fn = v.get("video_filename") or v.get("scene_filename")
                if fn:
                    m = re.search(r"video_(\d+)\.(mp4|avi)$", os.path.basename(fn))
                    if m:
                        entry_by_id[m.group(1).lstrip("0")] = v
                elif "scene_index" in v:
                    entry_by_id[str(int(v["scene_index"]))] = v  # normalize

            matched = 0
            for vp in vid_files:
                m = re.search(r"video_(\d+)\.(mp4|avi)$", os.path.basename(vp))
                if not m: continue
                vid = m.group(1).lstrip("0")
                meta = entry_by_id.get(vid)
                if meta:
                    self.items.append((vp, meta))
                    matched += 1

            print(f"[CLEVRER] videos(recursive): {len(vid_files)} | ann(mono entries): {len(entries)} | matched pairs: {matched}")


        else:
            print(f"[WARN] Annotation path not found: {ann_path}")

    def __len__(self): return len(self.items)

    def _sample_frame_indices(self, frame_count:int, T_out:int) -> List[int]:
        if frame_count <= 0: return []
        if T_out <= 1: return [0]
        return [int(round(i*(frame_count-1)/(T_out-1))) for i in range(T_out)]

    def _read_clip_cv2(self, path:str) -> List[Image.Image]:
        if cv2 is None:
            raise RuntimeError("OpenCV not available; install python-opencv or switch to imageio/decord.")
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idxs = self._sample_frame_indices(n, self.frames)
        imgs: List[Image.Image] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, self.resize, interpolation=cv2.INTER_AREA)
            imgs.append(Image.fromarray(frame))
        cap.release()
        return imgs

    def __getitem__(self, i):
        vpath, meta_or_path = self.items[i]
        # read video clip
        imgs = self._read_clip_cv2(vpath)
        if len(imgs) == 0:
            imgs = [Image.new("RGB", self.resize, (0,0,0)) for _ in range(self.frames)]
        clip = torch.stack([self.to_tensor(im) for im in imgs], dim=0)  # (T,3,H,W)

        # load meta if stored as path
        if isinstance(meta_or_path, str):
            with open(meta_or_path, "r") as f:
                meta = json.load(f)
        else:
            meta = meta_or_path

        y = concept_vector_from_annotation(meta, self.concepts)
        return clip, y



class CLEVRAtrDataset(Dataset):
    def __init__(self, img_dir: str, scenes_json_path: str, transform=None):
        self.img_dir = img_dir
        self.scenes = load_scenes(scenes_json_path)
        self.fnames = sorted(self.scenes.keys())
        self.transform = transform or T.Compose([T.Resize((224,224)), T.ToTensor()])
    def __len__(self):
        return len(self.fnames)
    def __getitem__(self, idx):
        fn = self.fnames[idx]
        lab = concept_vector_from_scene(self.scenes[fn])
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        x = self.transform(img)
        return x, lab

# -----------------------------
# Leaf bank (same as before)
# -----------------------------
class CLEVRLeafBank(nn.Module):
    def __init__(self, feat_dim: int = 256, backbone: str = "tiny"):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            import torchvision.models as M
            m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
            self.backbone = nn.Sequential(*(list(m.children())[:-1]))  # (B,512,1,1)
            in_ch = 512
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32,64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64,96, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(96,128,3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
            )
            in_ch = 128
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, feat_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(len(concept_names))])
        self.temp_scaler: Optional[TempScaler] = None

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        z = self.backbone(images)
        return self.proj(z)  # (B,F)

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        z = self.encoder(images)
        return torch.cat([h(z) for h in self.heads], dim=1)

    def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(images)
        if self.temp_scaler is not None:
            logits = self.temp_scaler(logits)
        return torch.sigmoid(logits)

class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.T.to(logits.device).clamp_min(1e-3)

# --- NEW (video leaf bank) ---
class CLEVRERLeafBank(nn.Module):
    def __init__(self, K: int, feat_dim: int = 256, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_name = backbone
        if backbone == "resnet18":
            import torchvision.models as M
            m = M.resnet18(weights=M.ResNet18_Weights.DEFAULT)
            self.frame_backbone = nn.Sequential(*(list(m.children())[:-1]))  # (B,512,1,1)
            in_ch = 512
        else:
            self.frame_backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(32,64, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(64,96, 3, 2, 1), nn.ReLU(),
                nn.Conv2d(96,128,3, 2, 1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((1,1)),
            ); in_ch = 128
        self.proj = nn.Sequential(nn.Flatten(), nn.Linear(in_ch, feat_dim), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(K)])
        self.temp_scaler: Optional[TempScaler] = None

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        frames: (B,T,3,H,W) or (T,3,H,W) -> (B,feat_dim) by mean temporal pool.
        """
        if frames.dim()==4:
            frames = frames.unsqueeze(0)
        B,T,C,H,W = frames.shape
        x = frames.view(B*T, C, H, W)
        z = self.frame_backbone(x)        # (B*T, Cb, 1, 1)
        z = self.proj(z)                  # (B*T, F)
        z = z.view(B, T, -1).mean(dim=1)  # (B, F)
        return z

    # keep API identical to your image leaf bank:
    def encoder(self, frames: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(frames)

    def forward_logits(self, frames: torch.Tensor) -> torch.Tensor:
        z = self.encode_frames(frames)
        return torch.cat([h(z) for h in self.heads], dim=1)  # (B,K)

    def forward_probs(self, frames: torch.Tensor) -> torch.Tensor:
        logits = self.forward_logits(frames)
        if self.temp_scaler is not None:
            logits = self.temp_scaler(logits)
        return torch.sigmoid(logits)


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
    if manual_tag:
        return f"manual_{manual_tag}"
    parts = []
    for attr in ["encoder", "backbone", "proj", "cnn", "trunk", "feature_extractor", "body"]:
        m = getattr(leaf_bank, attr, None)
        if isinstance(m, nn.Module):
            parts.append(m.state_dict())
    if parts:
        h = hashlib.sha256()
        for sd in parts:
            h.update(_hash_state_dict(sd).encode())
        return h.hexdigest()[:24]
    try:
        return _hash_state_dict(leaf_bank.state_dict())
    except Exception:
        return hashlib.sha256(repr(leaf_bank).encode()).hexdigest()[:24]

# -----------------------------
# Mining simple A=>B / A=>¬B rules (unchanged)
# -----------------------------
@dataclass
class Rule:
    kind: str
    head: int
    body: int
    support: float
    confidence: float

@dataclass
class MiningConfig:
    support_thresh: float = 0.05
    confidence_pos: float = 0.995
    confidence_neg: float = 0.005
    max_rules: int = 25

def mine_rules(loader: DataLoader, N: int, cfg: MiningConfig) -> List[RuleSpec]:
    cnt_A = torch.zeros(N, dtype=torch.long)
    cnt_AB = torch.zeros(N, N, dtype=torch.long)
    total = 0
    for _, labels in loader:
        B = labels.shape[0]; total += B
        cnt_A += labels.sum(dim=0)
        for b in range(B):
            idx = torch.nonzero(labels[b], as_tuple=False).flatten().tolist()
            for i in idx:
                for j in idx:
                    if i == j: continue
                    cnt_AB[i, j] += 1
    rules: List[RuleSpec] = []
    total = max(total, 1)
    for a in range(N):
        suppA = cnt_A[a].item() / total
        if suppA < cfg.support_thresh:
            continue
        for b in range(N):
            if a == b: continue
            conf_pos = (cnt_AB[a, b].item() / max(cnt_A[a].item(), 1)) if cnt_A[a] > 0 else 0.0
            if conf_pos >= cfg.confidence_pos:
                rules.append(RuleSpec("IMPLIES", a+1, b+1, f"c{a+1}->c{b+1}"))
            if conf_pos <= cfg.confidence_neg:
                rules.append(RuleSpec("IMPLIES_NOT", a+1, b+1, f"c{a+1}->¬c{b+1}"))
    uniq: Dict[Tuple[str,int,int], RuleSpec] = {}
    for r in rules:
        uniq[(r.kind, r.head_cid, r.body_cid)] = r
    return list(uniq.values())[: cfg.max_rules]

# -----------------------------
# Utilities (unchanged)
# -----------------------------
@torch.no_grad()
def tally_concepts(loader: DataLoader) -> torch.Tensor:
    cnt = torch.zeros(len(concept_names), dtype=torch.long)
    for _, y in tqdm(loader, desc="[Data] tally", leave=False):
        cnt += y.sum(dim=0)
    for i, (name, _) in enumerate(CONCEPTS, start=1):
        print(f"  c{i:02d} {name:24s} : {cnt[i-1].item()}")
    return cnt

@torch.no_grad()
def tally_from_annotations(ann_path, concepts: List[Tuple[str, Dict]], half_frame_default: int = 64) -> torch.Tensor:
    """
    Count positives per concept by running concept_vector_from_annotation
    over every annotation file (or list) we can find.
    """
    import glob, json, re
    def _iter_ann_files(p):
        if os.path.isdir(p):
            return glob.glob(os.path.join(p, "**", "annotation_*.json"), recursive=True)
        return [p] if os.path.isfile(p) else []

    cnt = torch.zeros(len(concepts), dtype=torch.long)
    files = _iter_ann_files(ann_path)
    for jf in files:
        try:
            with open(jf, "r") as f:
                meta = json.load(f)
        except Exception:
            continue

        # A file may be a single annotation dict or a list of them.
        metas = meta if isinstance(meta, list) else [meta]
        for m in metas:
            y = concept_vector_from_annotation(m, concepts)
            cnt += y.long()
    return cnt



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

def train_leaf_bank(bank: CLEVRLeafBank, loader: DataLoader, epochs: int, lr: float, device: str,
                    weight_decay: float = 0.0, pos_weight: Optional[torch.Tensor] = None):
    bank = bank.to(device)
    opt = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bank.train()
    pbar_epochs = trange(1, epochs+1, desc="[LeafBank] epochs", leave=True)
    for ep in pbar_epochs:
        total = 0.0; n=0
        pbar = tqdm(loader, desc=f"[LeafBank] epoch {ep}/{epochs}", leave=False)
        for j,(imgs,labels) in enumerate(pbar):
            #if j > 2:                               #-------------------------------DEBUG-------------------------
                #break
            imgs = imgs.to(device); labels = labels.to(device).float()
            logits = bank.forward_logits(imgs)
            loss = crit(logits, labels)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.item()); n += 1
            pbar.set_postfix(avg_loss=f"{total/max(n,1):.4f}")
        pbar_epochs.set_postfix(avg_loss=f"{total/max(n,1):.4f}")

# -----------------------------
# Calibration (memory-safe)
# -----------------------------
def fit_temperature(bank: CLEVRLeafBank, loader: torch.utils.data.DataLoader,
                    device: str, max_samples: int | None = None) -> 'TempScaler':
    bank.eval()
    all_logits, all_labels, seen = [], [], 0
    with torch.inference_mode():
        for imgs, y in loader:
            imgs = imgs.to(device, non_blocking=True)
            logits = bank.forward_logits(imgs).detach().cpu()
            all_logits.append(logits); all_labels.append(y.float())
            seen += imgs.size(0)
            if max_samples is not None and seen >= max_samples:
                break
    logits_cpu = torch.cat(all_logits, dim=0)
    labels_cpu = torch.cat(all_labels, dim=0)
    scaler = TempScaler()
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = crit(scaler(logits_cpu), labels_cpu)
        loss.backward(); return loss
    opt.step(closure)
    return scaler

# -----------------------------
# Aggregation
# -----------------------------
class LearnedAggregator(nn.Module):
    def __init__(self, R: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(R))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        x = torch.logit(torch.clamp(probs, 1e-6, 1-1e-6))
        return torch.sigmoid(x @ self.w + self.b)

# REPLACED: compute truths from graphs (works for simple & compound)
@torch.no_grad()
def compute_rule_truths_batch_from_graphs(y_batch: torch.Tensor,
                                          rule_graphs: List[dgl.DGLGraph]) -> torch.Tensor:
    """
    y_batch: (B, K) binary ground-truth concept matrix.
    rule_graphs: list of DGL graphs where leaves have:
        - ndata['mask']==1
        - ndata['x'] = 1..K concept ids (1-based)
      internal nodes have:
        - ndata['mask']==0
        - ndata['y'] in {1: IFF, 2: IMPLIES, 3: AND, 4: OR}
      edges have:
        - edata['neg'] in {-1, +1}  (child negation)
        - edata['pos'] in {0,1}     (left/right for binary ops; optional for AND/OR)
    Returns: (B, R) with 1 if the rule holds for the sample, 0 otherwise.
    """
    device = y_batch.device
    B, K = y_batch.shape
    out = []

    for g in rule_graphs:
        g = g  # single rule graph
        num_nodes = g.num_nodes()
        mask = g.ndata['mask'].to(device)        # 1=leaf, 0=op
        x_ids = g.ndata['x'].to(device)          # concept ids (1-based for leaves, 0 for ops)
        ops   = g.ndata['y'].to(device)          # operator code for ops; 0 for leaves

        # node truth values for the whole batch
        vals = torch.zeros(B, num_nodes, dtype=torch.long, device=device)

        # 1) set leaves from y_batch (watch 1-based → 0-based)
        leaf_idx = (mask == 1).nonzero(as_tuple=False).flatten()
        if leaf_idx.numel() > 0:
            cids = x_ids[leaf_idx] - 1                        # (L,)
            if (cids < 0).any() or (cids >= K).any():
                raise ValueError("Leaf concept id out of range (check x is 1..K).")
            # gather per sample
            vals[:, leaf_idx] = y_batch[:, cids]

        # 2) topological evaluate internal nodes
        topo_order = []
        for batch in dgl.topological_nodes_generator(g):
            if isinstance(batch, torch.Tensor):
                topo_order.extend(batch.tolist())
            else:
                for b in batch:
                    topo_order.extend(b.tolist())

        for n in topo_order:
            if mask[n] == 1:
                continue  # leaf already set
            op = int(ops[n].item())  # 1=IFF,2=IMPLIES,3=AND,4=OR

            src, dst, eid = g.in_edges(n, form='all')  # tensors
            if src.numel() == 0:
                # isolated op node → treat as True by convention
                vals[:, n] = 1
                continue

            # collect child truth values with negation and (optional) pos
            child = []
            for si, ei in zip(src.tolist(), eid.tolist()):
                v = vals[:, si].clone()
                if 'neg' in g.edata:
                    neg = int(g.edata['neg'][ei].item())
                    if neg == -1:
                        v = 1 - v
                p = int(g.edata['pos'][ei].item()) if 'pos' in g.edata else -1
                child.append((p, v))

            # For binary asymmetric ops, order by pos (0=left, 1=right)
            if op in (1, 2):  # IFF, IMPLIES
                # fall back to current order if pos missing
                if all(p in (0, 1) for p, _ in child) and len(child) >= 2:
                    child.sort(key=lambda t: t[0])
                # take first two
                a = child[0][1]
                b = child[1][1]
                if op == 2:   # IMPLIES: (!a) or b
                    vals[:, n] = ((1 - a) | b).long()
                else:         # IFF: a == b
                    vals[:, n] = (a == b).long()

            elif op == 3:  # AND
                v = child[0][1]
                for _, vv in child[1:]:
                    v = (v & vv).long()
                vals[:, n] = v

            elif op == 4:  # OR
                v = child[0][1]
                for _, vv in child[1:]:
                    v = (v | vv).long()
                vals[:, n] = v

            else:
                # unknown op → conservative False
                vals[:, n] = 0

        # 3) rule truth is the truth at the root (node with out_degree == 0)
        roots = (g.out_degrees() == 0).nonzero(as_tuple=False).flatten()
        assert roots.numel() == 1, "Rule graph must have exactly one root."
        out.append(vals[:, roots[0]])

    return torch.stack(out, dim=1)  # (B, R)


# -----------------------------
# Scoring
# -----------------------------
@torch.no_grad()
def anomaly_score_batch(images: torch.Tensor, trainers, bank: CLEVRLeafBank, device: str,
                        agg: str = "geo", learned: Optional[LearnedAggregator] = None) -> torch.Tensor:
    images = images.to(device)
    scores = []
    for _, trainer in trainers:
        if getattr(trainer, "_enc_fprint", None) is None:
            trainer._enc_fprint = encoder_fingerprint(bank)
        if hasattr(trainer, "predict_root_batch"):
            p = trainer.predict_root_batch(images, bank)  # (B,)
        else:
            probs = [trainer.predict_root(images[b], bank) for b in range(images.size(0))]
            p = torch.tensor(probs, device=device)
        scores.append(torch.clamp(p, 1e-6, 1.0))
    probs = torch.stack(scores, dim=1)  # (B,R)
    if agg == "min":
        p = probs.min(dim=1).values
    elif agg == "mean":
        p = probs.mean(dim=1)
    elif agg == "learned" and learned is not None:
        p = learned(probs)
    else:  # geo
        p = torch.exp(torch.log(probs).mean(dim=1))
    return 1.0 - p

def tally_concepts_generic(dataset, K: int, num_workers: int = 2) -> torch.Tensor:
    """Count positives per concept for any dataset that yields (x, y) with y shape (K,)."""
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=num_workers)
    cnt = torch.zeros(K, dtype=torch.long)
    for _, y in loader:
        cnt += y.sum(dim=0).long()
    return cnt

# -----------------------------
# Main
# -----------------------------
def main():
    global CONCEPTS
    args = build_args()
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

    base_concepts: List[Tuple[str, Dict[str, Any]]] = [
        ("blue_sphere",   {"color":"blue",  "shape":"sphere"}),
        ("red_sphere",    {"color":"red",   "shape":"sphere"}),
        ("green_cube",    {"color":"green", "shape":"cube"}),
        ("yellow_cyl",    {"color":"yellow","shape":"cylinder"}),
        ("metal_any",     {"material":"metal"}),
        ("gray_cyl",      {"color":"gray",  "shape":"cylinder"}),
    ]
    CONCEPTS = list(base_concepts)
    # --- NEW (dataset switch) ---
    if args.dataset == "clevrer":
        # Resolve paths (supports either monolithic JSON or per-video folders)
        # Prefer folder style: annotations/train and annotations/validation
        root = args.clevrer_root

        vid_train = None if args.eval_only else _pick_video_dir(root, "train")
        vid_val   = _pick_video_dir(root, "validation")

        ann_train = None if args.eval_only else _pick_ann_path(root, "train")
        ann_val   = _pick_ann_path(root, "validation")

        # ====================
        # Event-rich concepts
        # ====================
        EVENT_RICH: List[Tuple[str, Dict]] = []
        if args.dataset == "clevrer" and args.event_rich:
            # Choose attribute filters
            shapes  = ["sphere","cube","cylinder"]
            colors  = ["red","green","blue","yellow","gray","brown","purple","cyan"]
            mats    = ["rubber","metal"]

            filters: List[Dict[str,str]] = []
            if args.event_filters.startswith("shapes"):
                filters += [{"shape":s} for s in shapes]
            if "colors" in args.event_filters:
                filters += [{"color":c} for c in colors]
            if "materials" in args.event_filters:
                filters += [{"material":m} for m in mats]

            # Dedup filters (in case overlapping)
            uniq = []
            seen = set()
            for f in filters:
                k = tuple(sorted(f.items()))
                if k not in seen:
                    uniq.append(f); seen.add(k)
            filters = uniq

            # limit pair combos to avoid explosion
            pair_candidates: List[Tuple[Dict,str,Dict,str]] = []  # (A,Anm,B,Bnm)
            def _nm(f: Dict[str,str]) -> str:
                return "&".join([f"{k}={v}" for k,v in sorted(f.items())])

            for i in range(len(filters)):
                for j in range(i, len(filters)):
                    A, B = filters[i], filters[j]
                    pair_candidates.append((A, _nm(A), B, _nm(B)))
            # truncate but always include pairs required by hard-coded rules
            pair_candidates = pair_candidates[:args.max_event_pairs]

            # Ensure hard-coded collision atoms exist even if --max_event_pairs is small
            required_pairs = _required_event_pairs_for_hardcoded_rules()
            # map "k=v" -> dict
            def _flt(s: str) -> Dict[str,str]:
                k,v = s.split("=")
                return {k:v}

            def _prio(s: str) -> int:
                k = s.split("=", 1)[0]
                if k == "shape": return 0
                if k == "color": return 1
                if k == "material": return 2
                return 9

            present = set()
            for (A,Anm,B,Bnm) in pair_candidates:
                present.add(tuple(sorted((Anm,Bnm))))
            added = 0
            for u,v in required_pairs:
                # keep canonical ordering used by EVENT_RICH builder (shapes before colors before materials)
                uo, vo = (u, v) if _prio(u) <= _prio(v) else (v, u)
                key = tuple(sorted((uo,vo)))
                if key in present:
                    continue
                A = _flt(uo); B = _flt(vo)
                pair_candidates.append((A, uo, B, vo))
                present.add(key)
                added += 1
            if added > 0:
                print(f"[EventRich] added {added} required event pairs for hard-coded rules (effective max_event_pairs={len(pair_candidates)})")



            # (1) Object-conditioned single-object events
            for F in filters:
                nm = _nm(F)
                EVENT_RICH.append((f"enter({nm})", {"_event":"enter", "A":F}))
                EVENT_RICH.append((f"exit({nm})",  {"_event":"exit",  "A":F}))

            # (2) Pairwise collisions (attribute-bound)
            for (A,Anm,B,Bnm) in pair_candidates:
                # collide(A,B)
                EVENT_RICH.append((f"collide({Anm},{Bnm})", {"_event":"collide_pair","A":A,"B":B}))
                # collision_before_half(A,B)
                EVENT_RICH.append((f"collide_before_half({Anm},{Bnm})", {"_event":"collide_before_half","A":A,"B":B}))
                # entered_then_collided(A,B)
                EVENT_RICH.append((f"entered_then_collided({Anm},{Bnm})", {"_event":"entered_then_collided","A":A,"B":B}))

            if args.add_event_concepts:
                CONCEPTS.extend(EVENT_CONCEPTS)   
            if args.event_rich:
                CONCEPTS.extend(EVENT_RICH)       

            if args.event_rich and not args.no_stats:
                from collections import Counter
                import itertools, glob, json
                ann_paths = sorted(glob.glob(os.path.join(args.clevrer_root, "annotations", "train", "**", "annotation_*.json"), recursive=True))[:500]
                cnt = Counter()
                for p in ann_paths:
                    ann = json.load(open(p, "r"))
                    y = concept_vector_from_annotation(ann, CONCEPTS)
                    for i, (nm, _) in enumerate(CONCEPTS):
                        if y[i].item() == 1: cnt[nm] += 1
                tot = max(len(ann_paths),1)
                print("[EventStats] supports over", tot, "videos (top 20):")
                for nm, c in cnt.most_common(20):
                    print(f"  {nm:50s}: {c/tot:6.2%}")

        # Build concept list
        concept_list = list(CONCEPTS)
        if args.add_question_concepts:
            # choose train or val questions depending on mode
            q_path = args.q_train_json or os.path.join(root, "questions", "train.json")
            if args.eval_only:
                q_path = args.q_val_json or os.path.join(root, "questions", "validation.json")
            mined = mine_attribute_concepts_from_questions(q_path, max_extra=6)
            concept_list += mined

        K = len(concept_list)
        print(f"[Concepts] K={K} -> {[name for name,_ in concept_list]}")
        concept_names = [nm for (nm, _) in concept_list]
        n_concepts    = len(concept_names)

        # ---- add this in main(), after concept_list/K are known ----
        if args.dataset == "clevrer":
            n_concepts = len(concept_list)            # == K
            concept_names = [nm for (nm, _) in concept_list]
        else:
            n_concepts = len(concept_names)                   # existing constant in your CLEVR (images) code
            # if you already have a names list for CLEVR, reuse it; otherwise:
            try:
                concept_names = [nm for (nm, _) in CONCEPTS]
            except Exception:
                concept_names = [f"c{i+1}" for i in range(n_concepts)]

        # datasets
        train_loader = None
        if not args.eval_only:
            ds_tr = CLEVRERVidDataset(vid_train, ann_train, concepts=concept_list,
                                    frames=args.frames, resize=(args.resize,args.resize),
                                    half_frame_default=args.event_half)
            train_loader = DataLoader(ds_tr, batch_size=args.batch_train, shuffle=True,
                                    num_workers=args.num_workers, pin_memory=True)

        # prevalence counts for pos_weight="auto"
        cnt = None
        if (not args.eval_only) and (args.pos_weight == "auto") and (train_loader is not None):
            cnt = tally_from_annotations(ann_train, concept_list, half_frame_default=args.event_half)
            if not args.no_stats:
                print("[Data] Concept counts (train):")
                for i, (nm, _) in enumerate(concept_list, start=1):
                    print(f"  c{i:02d} {nm:24s} : {cnt[i-1].item()}")


        ds_val = CLEVRERVidDataset(vid_val, ann_val, concepts=concept_list,
                                frames=args.frames, resize=(args.resize,args.resize),
                                half_frame_default=args.event_half)
        val_loader = DataLoader(ds_val, batch_size=args.batch_eval, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)

        # leaf bank (videos)
        bank = CLEVRERLeafBank(K=K, feat_dim=args.feat_dim, backbone=args.backbone).to(device)

    else:

        P = Paths(args.clevr_root)
        if not (os.path.exists(P.train_scenes) and os.path.exists(P.img_train)):
            print("Please set --clevr_root correctly. Expected scenes+images under:", args.clevr_root); return

        # Datasets
        val_tf = make_transforms("val", use_aug=False)
        ds_val = CLEVRAtrDataset(P.img_val, P.val_scenes, transform=val_tf)
        val_loader = DataLoader(ds_val, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)

        train_loader = None
        if not args.eval_only:
            train_tf = make_transforms("train", use_aug=args.augment)
            train_ds = CLEVRAtrDataset(P.img_train, P.train_scenes, transform=train_tf)
            train_loader = DataLoader(train_ds, batch_size=args.batch_train, shuffle=True, num_workers=args.num_workers)
            if not args.no_stats:
                print("[Data] Concept counts (train):")
                cnt = tally_concepts(DataLoader(train_ds, batch_size=128, shuffle=False))
            else:
                cnt = None
        else:
            cnt = None

        # Leaf bank
        bank = CLEVRLeafBank(feat_dim=args.feat_dim, backbone=args.backbone).to(device)

    # Load or train leaf bank
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
        # pos_weight
        # pos_weight (works for both CLEVR and CLEVRER)
        # ---- replace your pos_weight block with this ----
        pos_weight = None
        if args.pos_weight == "auto" and cnt is not None:
            total = (len(train_loader.dataset) if args.dataset == "clevrer" else len(train_ds))
            pw = []
            for i in range(n_concepts):
                p = max(int(cnt[i].item()), 1)
                n = max(total - p, 1)
                pw.append(n / p)
            pos_weight = torch.tensor(pw, dtype=torch.float32, device=device)
        else:
            # numeric value or leave None
            try:
                val = float(args.pos_weight)
                pos_weight = torch.full((n_concepts,), val, dtype=torch.float32, device=device)
            except Exception:
                pos_weight = None


        if args.epochs_leaf > 0:
            print("[LeafBank] Training …")
            assert train_loader is not None
            train_leaf_bank(bank, train_loader, epochs=args.epochs_leaf, lr=args.lr_leaf, device=device,
                            weight_decay=args.weight_decay, pos_weight=pos_weight)
            torch.save(bank.state_dict(), args.leaf_ckpt)


            print(f"[LeafBank] Saved checkpoint → {args.leaf_ckpt}")
        else:
            bank.eval()
                    # --- Evaluate leaf bank on VAL (same style as OI) ---
        idx_to_name = {i: concept_names[i] for i in range(len(concept_names))}
        evaluate_leaf_bank(bank, val_loader, idx_to_name, device=device, threshold=0.5)

    # Temperature scaling
    if args.calib_train and not args.eval_only and args.epochs_leaf > 0:
        calib_loader = DataLoader(train_loader.dataset, batch_size=args.batch_eval, shuffle=False, num_workers=args.num_workers)
        if device == "cuda":
            torch.cuda.empty_cache()
        scaler = fit_temperature(bank, calib_loader, device=device, max_samples=None)
        torch.save(scaler.state_dict(), args.calib_ckpt)
        print(f"[Calib] Saved temperature scaler → {args.calib_ckpt}")
    if os.path.exists(args.calib_ckpt):
        scaler = TempScaler(); scaler.load_state_dict(torch.load(args.calib_ckpt, map_location="cpu"))
        bank.temp_scaler = scaler.to(device)
        print(f"[Calib] Loaded temperature scaler: {args.calib_ckpt}")

    # -------- Build rule graphs (simple + compound) and trainers --------
    # Simple rules (3-node graphs as before)
    def build_graph_for_rule(rule: RuleSpec) -> dgl.DGLGraph:
        g = dgl.graph(([], []), num_nodes=3)
        mask = torch.tensor([1,1,0], dtype=torch.long)
        y    = torch.tensor([0,0,2], dtype=torch.long)  # root IMPLIES
        x    = torch.tensor([rule.head_cid, rule.body_cid, 0], dtype=torch.long)
        src = torch.tensor([0,1]); dst = torch.tensor([2,2])
        g.add_edges(src, dst)
        g.ndata["mask"] = mask; g.ndata["y"] = y; g.ndata["x"] = x
        neg = [+1,-1] if rule.kind == "IMPLIES_NOT" else [+1,+1]
        g.edata["neg"] = torch.tensor(neg, dtype=torch.long)
        g.edata["pos"] = torch.tensor([0,1], dtype=torch.long)
        return g

    # ================================
    # Rule construction
    #   - For CLEVRER: NO static rules. Add hard-coded chain + not-iff rules, plus mined compound rules.
    #   - For CLEVR (images): keep the previous static + compound wiring.
    # ================================
    rule_items: List[Tuple[str, dgl.DGLGraph]] = []

    if args.dataset == "clevrer":
        # hard-coded CLEVRER rules
        hard_exprs = build_clevrer_hardcoded_rule_exprs(concept_names)
        for name, expr in hard_exprs:
            rule_items.append((name, build_graph_from_expr(expr)))

        # mined compound rules: enter(E) -> (collide(A,E) & collide(A,B))
        if not args.no_mine_compound_rules:
            scan_ann = ann_val if args.compound_scan_split == "val" else ann_train
            if scan_ann is None:
                scan_ann = ann_val
            mined_exprs = mine_enter_implies_two_collides(
                ann_path=scan_ann,
                concept_list=concept_list,
                concept_names=concept_names,
                scan_limit=args.compound_scan_limit,
                min_hits=args.compound_min_hits,
                min_conf=args.compound_min_conf,
                min_viol=args.compound_min_viol,
                max_rules=args.max_compound_rules,
            )
            print(f"[Mine2] kept {len(mined_exprs)} compound rules (cap={args.max_compound_rules})")
            for name, expr in mined_exprs:
                rule_items.append((name, build_graph_from_expr(expr)))

    else:
        # ================================
        # PATCH B: build rules from names at runtime
        # ================================
        rules_simple, compound_named_exprs = build_seed_and_compound_rules_by_name(concept_names)

        # Seed simple graphs
        for r in rules_simple:
            rule_items.append((r.name, build_graph_for_rule(r)))

        # Add compound graphs (2-level)
        for name, expr in compound_named_exprs:
            rule_items.append((name, build_graph_from_expr(expr)))

        # Optional: auto-mine simple rules (still depth-1)
        if (not args.eval_only) and args.auto_mine and train_loader is not None:
            mined = mine_rules(DataLoader(train_loader.dataset, batch_size=256, shuffle=False), len(concept_names),
                               MiningConfig(args.support_thresh, args.confidence_pos, args.confidence_neg, args.max_rules))
            print(f"[Mining] Mined {len(mined)} rules")
            for r in mined:
                rule_items.append((r.name, build_graph_for_rule(r)))

    # Build trainers per rule graph
    trainers = []
    rule_graphs = []
    for i, (rname, g) in enumerate(rule_items, start=1):
        rule_graphs.append(g)
        rule_cache = os.path.join(args.cache_dir, f"rule_{i:02d}")
        cache = SubtreeCache(CacheConfig(root_dir=rule_cache))
        trainer = LevelwiseTrainer(
            g, N_concepts=len(concept_names), cache=cache, device=device,
            cache_leafroot_only=False, lineage_aware=True,
        )
        trainer._enc_fprint = encoder_fingerprint(bank, manual_tag=(args.encoder_tag or None))
        root = int(torch.nonzero(g.out_degrees()==0, as_tuple=False).flatten()[0].item())
        print(f"[Trainer] Rule {i:02d}: {rname} | topo_sig={subtree_signature(g, root)}")

        if not args.eval_only:
            kwargs = dict(dataset=train_loader, leaf_bank=bank, epochs_per_level=args.epochs_level,
                            lr=args.lr_level, negatives=args.negatives, use_soft_leaves=True, verbose=True, use_tqdm=True)
            if args.train_missing_only:
                try:
                    trainer.train(**kwargs, train_missing_only=True)  # type: ignore[arg-type]
                except TypeError:
                    print("[Trainer] train_missing_only not supported by your trainer; training all nodes.")
                    trainer.train(**kwargs)
            else:
                trainer.train(**kwargs)
        trainers.append((rname, trainer))

    # Save manifest
    manifest = {
        "leaf_ckpt": os.path.abspath(args.leaf_ckpt),
        "cache_dir": os.path.abspath(args.cache_dir),
        "encoder_fingerprint": encoder_fingerprint(bank, manual_tag=(args.encoder_tag or None)),
        "feat_dim": args.feat_dim,
        "rules": [{"i": i, "name": nm} for i, (nm, _) in enumerate(rule_items, start=1)],
        "concepts": [{"i": i, "name": nm, "filter": f} for i, (nm, f) in enumerate(concept_list, start=1)],
        "agg": args.agg,
        "backbone": args.backbone,
    }
    with open(os.path.join(args.run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Run] Manifest saved → {os.path.join(args.run_dir, 'manifest.json')}")

    # -------------------------
    # Learned aggregator (optional)
    # -------------------------
    learned_agg: Optional[LearnedAggregator] = None
    if args.agg == "learned":
        if args.eval_only and os.path.exists(args.agg_ckpt):
            learned_agg = LearnedAggregator(len(trainers))
            learned_agg.load_state_dict(torch.load(args.agg_ckpt, map_location="cpu"))
            learned_agg.to(device).eval()
            print(f"[Agg] Loaded learned aggregator: {args.agg_ckpt}")
        elif not args.eval_only:
            # Fit on train set: anomaly = 1 - (all rules true by GT labels from graphs)
            X_list, y_list = [], []
            for imgs, labels in tqdm(train_loader, desc="[Agg] collect", leave=False):
                imgs = imgs.to(device)
                probs = []
                for _, trainer in trainers:
                    if hasattr(trainer, "predict_root_batch"):
                        p = trainer.predict_root_batch(imgs, bank)
                    else:
                        p = torch.tensor([trainer.predict_root(imgs[b], bank) for b in range(imgs.size(0))], device=device)
                    probs.append(torch.clamp(p, 1e-6, 1.0))
                P = torch.stack(probs, dim=1)  # (B,R)
                truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)  # (B,R)
                is_normal = truths.min(dim=1).values  # 1 if all true
                y = 1 - is_normal.float()             # anomaly target
                X_list.append(P.detach().cpu()); y_list.append(y.detach().cpu())
            X = torch.cat(X_list, dim=0); Y = torch.cat(y_list, dim=0)
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

    # -------------------------
    # Evaluation on val
    # -------------------------
    print("[Eval] Scoring anomaly on validation set …")
    all_scores, all_truths = [], []
    for imgs, labels in tqdm(val_loader, desc="[Eval] batches"):
        scores = anomaly_score_batch(imgs, trainers, bank, device=device, agg=args.agg, learned=learned_agg)  # (B,)
        truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)  # (B,R)
        is_normal = truths.min(dim=1).values
        y = (1 - is_normal).tolist()  # anomaly=1
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

    # Per-rule AUROC
    try:
        from sklearn.metrics import roc_auc_score
        print("[Diag] Per-rule AUROC (higher is better):")
        R = len(trainers)
        per_rule = [ [] for _ in range(R) ]
        y_rule = [ [] for _ in range(R) ]
        probs_total = []
        y_total = []
        for imgs, labels in tqdm(val_loader, desc="[Diag] collect", leave=False):
            imgs = imgs.to(device)
            probs = []
            for _, trainer in enumerate(trainers):
                if hasattr(trainer[1], "predict_root_batch"):
                    p = trainer[1].predict_root_batch(imgs, bank)
                else:
                    p = torch.tensor([trainer[1].predict_root(imgs[b], bank) for b in range(imgs.size(0))], device=device)
                probs.append(p.detach().cpu())
            P = torch.stack(probs, dim=1)  # (B,R)
            probs_total.append(P)
            truths = compute_rule_truths_batch_from_graphs(labels, rule_graphs)  # (B,R)
            y_total.append(truths)
            for r_idx in range(R):
                per_rule[r_idx].extend(P[:, r_idx].tolist())
                y_rule[r_idx].extend((1 - truths[:, r_idx]).tolist())  # anomaly label per rule

        probs_total = 1 - torch.cat(probs_total, dim=0)
        y_total = 1 - torch.cat(y_total, dim=0)
        np.savez_compressed("./A_score_CLEVRER_.npz", A_score=probs_total.cpu().numpy())
        np.savez_compressed("./Test_Y_CLEVRER_.npz", Test_Y=y_total.cpu().numpy())
        for r_idx, (rname, _) in enumerate(trainers):
            auc_r = roc_auc_score(y_rule[r_idx], [1.0 - s for s in per_rule[r_idx]]) if len(set(y_rule[r_idx])) > 1 else float('nan')
            print(f"  - {r_idx+1:02d} {rname:40s} : AUROC={auc_r:.3f}")
    except Exception:
        pass

    # Show a few samples
    # Samples (OI-style detail)
    print("[Samples]")
    S = 100
    anom_log = os.path.join(args.run_dir, "anom_samples.txt")
    norm_log = os.path.join(args.run_dir, "normal_samples.txt")
    os.makedirs(args.run_dir, exist_ok=True)
    # clear previous logs for a fresh run (comment these two lines if you prefer append-only)
    open(anom_log, 'w').close()
    open(norm_log, 'w').close()

    for i in range(min(S, len(ds_val))):
        x, y_vec = ds_val[i]  # <-- avoid shadowing; was: y

        # robust sample id (images: .fnames; videos: dataset.items[i][0])
        if hasattr(ds_val, "fnames"):
            img_id = ds_val.fnames[i]
        elif hasattr(ds_val, "items") and isinstance(ds_val.items[i], (list, tuple)) and len(ds_val.items[i]) >= 1:
            # first element is the video path in CLEVRER dataset
            img_id = os.path.basename(ds_val.items[i][0])
        else:
            img_id = f"idx_{i:06d}"
        # anomaly score via your current aggregator
        # sc = anomaly_score_batch(
        #     x.unsqueeze(0), trainers, bank, device=device, agg=args.agg, learned=learned_agg
        # )[0].item()
        sc = probs_total[i,:]


        # rule truths from *graphs* (1 = rule holds)
        truths = compute_rule_truths_batch_from_graphs(y_vec.unsqueeze(0), rule_graphs)
        status = "NORMAL" if truths.min(dim=1).values.item() == 1 else "ANOMALY"

        # names of ALL labels true for this sample
        on_idx = (y_vec == 1).nonzero(as_tuple=False).flatten().tolist()
        if 'concept_names' in globals() or 'concept_names' in locals():
            label_names = [concept_names[j] for j in on_idx]
        else:
            # fallback for legacy CLEVR image code
            try:
                label_names = [CONCEPTS[j][0] for j in on_idx]
            except Exception:
                label_names = [f"c{j+1}" for j in on_idx]
        labels_str = ", ".join(label_names) if label_names else "(none)"

        torch.set_printoptions(precision=2, sci_mode=False)
        line = f"{img_id} | status={status:7s} | score={sc} | labels=[{labels_str}]"

        if status == "ANOMALY":
            # list violated rules by name (limit to 5 like OI)
            violated_idx = (truths[0] == 0).nonzero(as_tuple=False).flatten().tolist()
            if violated_idx:
                violated_names = [f"{r+1:03d} {trainers[r][0]}" for r in violated_idx]
                shown = "; ".join(violated_names[:5]) + ("" if len(violated_names) <= 5 else "; …")
                line += f" | violated=[{shown}]"

        # append to logs under run_dir (portable vs hardcoded paths)
        with open(anom_log if status == "ANOMALY" else norm_log, "a") as f:
            f.write(f"{img_id}\n")

        print(line)



if __name__ == "__main__":
    main()
