#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cifar10_and_pair_grids.py

CIFAR-10 learned-evaluator diagnostic for pairwise impossible rule:

    A AND B,   with A != B

CIFAR-10 is single-label, so A∧B should be false for every real test image.
For each requested pair (A,B), this script:
  * trains/loads one CIFAR-10 leaf bank once;
  * trains/loads the learned gate for the rule A∧B;
  * scores CIFAR-10 test images;
  * saves separate score-sorted grids for test images whose true class is A;
  * saves separate score-sorted grids for test images whose true class is B;
  * saves both the lowest-score grid and greatest-score grid in the same run.

By default --pairs all runs all unordered CIFAR-10 class pairs.
Use --pairs "cat,dog;ship,truck" or --a cat --b dog for selected pairs.
Numeric class ids 0..9 are also accepted:

  0 airplane, 1 automobile, 2 bird, 3 cat, 4 deer,
  5 dog, 6 frog, 7 horse, 8 ship, 9 truck

Outputs per pair/test class:
  scores_pair_A_B_testclass_D_anom_asc.csv
  scores_pair_A_B_testclass_D_lowest_anomaly_asc.csv
  scores_pair_A_B_testclass_D_greatest_anomaly_asc.csv
  grids/grid_pair_A_B_testclass_D_lowest_anomaly_asc.png
  grids/grid_pair_A_B_testclass_D_greatest_anomaly_asc.png

Here anomaly_score = root_prob = P_hat(A AND B). Since the rule is impossible
under one-hot CIFAR-10 labels, larger root_prob is the suspicious/high-anomaly case.
"""

import os
import csv
import json
import argparse
import random
import re
from typing import Any, Dict, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T
from torchvision.utils import make_grid
import dgl
from tqdm import tqdm

from chimera_logic.trainer import LevelwiseTrainer, SubtreeCache, CacheConfig, encoder_fingerprint as trainer_encoder_fingerprint
from chimera_logic.evaluator import fill_node_vector_from_concepts


N_CONCEPTS = 10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CIFAR10_CLASSES)}
# A few common aliases.
CLASS_TO_IDX.update({"car": 1, "auto": 1, "plane": 0, "aeroplane": 0})


# -----------------------------
# Repro / small utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def device_from_arg(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def slugify(s: str) -> str:
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "class"


def parse_class_token(tok: Any) -> int:
    s = str(tok).strip().lower()
    if s == "":
        raise ValueError("Empty class token.")
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        i = int(s)
        if not 0 <= i <= 9:
            raise ValueError(f"CIFAR-10 class index must be in 0..9, got {i}")
        return i
    if s not in CLASS_TO_IDX:
        raise ValueError(
            f"Unknown CIFAR-10 class {tok!r}. Use one of {CIFAR10_CLASSES} or an index 0..9."
        )
    return CLASS_TO_IDX[s]


def parse_pairs(args: argparse.Namespace) -> List[Tuple[int, int]]:
    if args.a is not None or args.b is not None:
        if args.a is None or args.b is None:
            raise ValueError("Pass both --a and --b, or neither.")
        pairs = [(parse_class_token(args.a), parse_class_token(args.b))]
    else:
        s = str(args.pairs).strip().lower()
        if s == "all":
            pairs = [(a, b) for a in range(10) for b in range(a + 1, 10)]
        else:
            pairs = []
            for chunk in s.split(";"):
                chunk = chunk.strip()
                if not chunk:
                    continue
                bits = [x.strip() for x in chunk.split(",")]
                if len(bits) != 2:
                    raise ValueError(f"Bad pair chunk {chunk!r}. Expected form A,B;C,D")
                pairs.append((parse_class_token(bits[0]), parse_class_token(bits[1])))

    out: List[Tuple[int, int]] = []
    seen = set()
    for a, b in pairs:
        if a == b:
            raise ValueError(f"A and B must be different; got ({a},{b})")
        # A∧B is commutative, so use unordered canonical pair and avoid duplicated work.
        aa, bb = (a, b) if a < b else (b, a)
        if (aa, bb) not in seen:
            seen.add((aa, bb))
            out.append((aa, bb))
    return out


# -----------------------------
# CIFAR-10 dataset and leaf bank
# -----------------------------

class CIFAR10Concepts(Dataset):
    def __init__(self, root: str, train: bool, download: bool = True, augment: bool = False):
        if train and augment:
            tx = T.Compose([
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
        else:
            tx = T.Compose([T.ToTensor()])
        self.ds = torchvision.datasets.CIFAR10(root=root, train=train, download=download, transform=tx)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, cls = self.ds[idx]  # img: (3,32,32), cls: int 0..9
        labels = torch.zeros(N_CONCEPTS, dtype=torch.long)
        labels[int(cls)] = 1
        return img, labels


class CIFAR10LeafBank(nn.Module):
    """Shared CNN encoder + 10 independent sigmoid heads.

    Exposes the same interface expected by LevelwiseTrainer:
      encoder(images) -> (B,F)
      forward_probs(images) -> (B,10)
    """
    def __init__(self, feat_dim: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.enc = nn.Sequential(
            nn.Linear(256, feat_dim), nn.ReLU(),
        )
        self.heads = nn.ModuleList([nn.Linear(feat_dim, 1) for _ in range(N_CONCEPTS)])

    def encoder(self, images: torch.Tensor) -> torch.Tensor:
        return self.enc(self.cnn(images))

    def forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        z = self.encoder(images)
        return torch.cat([h(z) for h in self.heads], dim=1)

    def forward_probs(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(images))

    def vectorize(self, x_single: torch.Tensor, N: int = N_CONCEPTS) -> torch.Tensor:
        if x_single.dim() == 3:
            x_single = x_single.unsqueeze(0)
        with torch.no_grad():
            p = self.forward_probs(x_single)
        return p[0]


def train_leaf_bank(bank: CIFAR10LeafBank, loader: DataLoader, device: str, epochs: int = 10, lr: float = 1e-3,
                    weight_decay: float = 0.0) -> None:
    bank = bank.to(device)
    opt = torch.optim.Adam(bank.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    bank.train()
    for ep in range(1, epochs + 1):
        total = 0.0
        n = 0
        for imgs, labels in tqdm(loader, desc=f"[LeafBank] epoch {ep}/{epochs}", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            logits = bank.forward_logits(imgs)
            loss = crit(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            n += 1
        print(f"[LeafBank] epoch {ep:02d} | loss={total / max(n, 1):.4f}")


@torch.no_grad()
def evaluate_leaf_bank(bank: CIFAR10LeafBank, loader: DataLoader, device: str, threshold: float = 0.5) -> None:
    bank.eval().to(device)
    correct_argmax = 0
    total = 0
    bce_sum = 0.0
    crit = nn.BCELoss(reduction="sum")
    for imgs, labels in tqdm(loader, desc="[LeafEval]", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels_f = labels.to(device, non_blocking=True).float()
        probs = bank.forward_probs(imgs)
        bce_sum += float(crit(probs, labels_f).item())
        pred = probs.argmax(dim=1).detach().cpu()
        true = labels.argmax(dim=1)
        correct_argmax += int((pred == true).sum().item())
        total += int(imgs.size(0))
    print(f"[LeafEval] argmax acc={correct_argmax / max(total, 1):.4f} | BCE/sample={bce_sum / max(total, 1):.4f}")


# -----------------------------
# Rule graph: A AND B
# -----------------------------

def build_and_graph_for_pair(a_cls: int, b_cls: int) -> dgl.DGLGraph:
    if a_cls == b_cls:
        raise ValueError("A and B must be different for A AND B diagnostic.")
    src = torch.tensor([0, 1], dtype=torch.long)
    dst = torch.tensor([2, 2], dtype=torch.long)
    g = dgl.graph((src, dst), num_nodes=3)
    g.ndata["mask"] = torch.tensor([1, 1, 0], dtype=torch.long)
    g.ndata["y"] = torch.tensor([0, 0, 3], dtype=torch.long)  # root AND = 3
    g.ndata["x"] = torch.tensor([a_cls + 1, b_cls + 1, 0], dtype=torch.long)
    g.edata["neg"] = torch.tensor([+1, +1], dtype=torch.long)
    g.edata["pos"] = torch.tensor([0, 1], dtype=torch.long)
    return g


@torch.no_grad()
def predict_root_batch(trainer: LevelwiseTrainer, images: torch.Tensor, leaf_bank: CIFAR10LeafBank, device: str) -> torch.Tensor:
    """Batch version of LevelwiseTrainer.predict_root, returning CPU tensor (B,)."""
    g = trainer.g.to(device)
    if trainer.lineage_aware and trainer._enc_fprint is None:
        trainer._enc_fprint = trainer_encoder_fingerprint(leaf_bank)

    images = images.to(device, non_blocking=True)
    if images.dim() == 3:
        images = images.unsqueeze(0)

    leaf_feats, leaf_probs = trainer._leaf_feats_probs(images, leaf_bank)
    Fdim = trainer.feat_dim or leaf_feats.shape[-1]
    B = images.shape[0]
    num_nodes = g.num_nodes()

    node_feats = torch.zeros(B, num_nodes, Fdim, device=device)
    node_probs = torch.zeros(B, num_nodes, device=device)

    leaf_prob_nodes = fill_node_vector_from_concepts(g, leaf_probs)
    node_probs[:, g.ndata["mask"].bool()] = leaf_prob_nodes[:, g.ndata["mask"].bool()]

    cid = g.ndata["x"].long()
    leaf_idx = torch.nonzero(cid > 0, as_tuple=False).flatten()
    if leaf_idx.numel() > 0:
        cids0 = (cid[leaf_idx] - 1).tolist()
        node_feats[:, leaf_idx, :] = leaf_feats[:, cids0, :]

    for d in range(1, len(trainer.levels)):
        for n in trainer.levels[d]:
            gate = trainer.runtime_gates.get(n)
            if gate is None:
                key = trainer.cache_key(n)
                if trainer.cache.has(key):
                    gate = trainer.cache.load(key, arity=len(g.in_edges(n)[0]), feat_dim=Fdim)
                    trainer.runtime_gates[n] = gate
            if gate is None:
                raise RuntimeError(f"No trained/cached gate for node {n}; train this pair before scoring.")
            gate = gate.to(device).eval()
            child_flat = trainer._gather_child_flat(n, node_feats)
            par_feat, par_prob = gate(child_flat)
            node_feats[:, n, :] = par_feat
            node_probs[:, n] = par_prob

    root = int(torch.nonzero(g.out_degrees() == 0, as_tuple=False).flatten()[0].item())
    return node_probs[:, root].detach().cpu()


# -----------------------------
# CSV / grids
# -----------------------------

def write_records_csv(path: str, records: List[Dict[str, Any]]) -> None:
    fields = [
        "pair_a", "pair_b", "pair_a_name", "pair_b_name", "test_class", "test_class_name", "image_index",
        "root_prob", "anomaly_score", "leaf_prob_a", "leaf_prob_b",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            w.writerow({k: r.get(k, "") for k in fields})


def save_cifar_grid_from_records(records: List[Dict[str, Any]], test_ds: Dataset,
                                 out_path: str, grid_n: int, grid_cols: int, title: str) -> None:
    selected = records[:grid_n]
    if not selected:
        print(f"[Grid] no records for {out_path}")
        return
    imgs = []
    for rec in selected:
        x, _ = test_ds[int(rec["image_index"])]
        imgs.append(x)
    imgs_t = torch.stack(imgs, dim=0).cpu().clamp(0, 1)
    grid = make_grid(imgs_t, nrow=grid_cols, padding=2)

    rows = (len(selected) + grid_cols - 1) // grid_cols
    fig_w = max(8, 1.35 * min(grid_cols, len(selected)))
    fig_h = max(3, 1.45 * rows)
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)
    ax = plt.subplot(1, 1, 1)
    ax.imshow(grid.permute(1, 2, 0).numpy(), vmin=0, vmax=1)
    ax.set_axis_off()
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, transparent=False)
    plt.close(fig)
    print(f"[Grid] saved {out_path}")


def score_pair_on_test_classes(trainer: LevelwiseTrainer, bank: CIFAR10LeafBank, test_loader: DataLoader,
                               device: str, a: int, b: int, max_eval_batches: int = 0) -> Dict[int, List[Dict[str, Any]]]:
    """Return records for only true class A and true class B."""
    records_by_class: Dict[int, List[Dict[str, Any]]] = {a: [], b: []}
    offset = 0
    bank.eval()
    for batch_idx, (imgs, labels) in enumerate(tqdm(test_loader, desc=f"[Score] pair {a}_{b}", leave=False)):
        if max_eval_batches and batch_idx >= max_eval_batches:
            break
        B = imgs.size(0)
        root_probs = predict_root_batch(trainer, imgs, bank, device=device)
        # For A AND B on single-label CIFAR-10, high root_prob is the suspicious case.
        anomaly_scores = root_probs
        classes = torch.argmax(labels, dim=1)
        with torch.no_grad():
            leaf_probs = bank.forward_probs(imgs.to(device, non_blocking=True)).detach().cpu()

        for local_i in range(B):
            d = int(classes[local_i].item())
            if d not in (a, b):
                continue
            records_by_class[d].append({
                "pair_a": a,
                "pair_b": b,
                "pair_a_name": CIFAR10_CLASSES[a],
                "pair_b_name": CIFAR10_CLASSES[b],
                "test_class": d,
                "test_class_name": CIFAR10_CLASSES[d],
                "image_index": offset + local_i,
                "root_prob": float(root_probs[local_i].item()),
                "anomaly_score": float(anomaly_scores[local_i].item()),
                "leaf_prob_a": float(leaf_probs[local_i, a].item()),
                "leaf_prob_b": float(leaf_probs[local_i, b].item()),
            })
        offset += B
    return records_by_class


def save_low_high_outputs(records: List[Dict[str, Any]], test_ds: Dataset, out_dir: str, grids_dir: str,
                          a: int, b: int, test_class: int, grid_n: int, grid_cols: int) -> List[Dict[str, Any]]:
    records_asc = sorted(records, key=lambda r: r["anomaly_score"])
    low_records = records_asc[:grid_n]
    high_records = records_asc[-grid_n:] if grid_n > 0 else []  # remains ascending because source is ascending

    a_name = slugify(CIFAR10_CLASSES[a])
    b_name = slugify(CIFAR10_CLASSES[b])
    d_name = slugify(CIFAR10_CLASSES[test_class])
    stem = f"pair_{a}_{b}_{a_name}_{b_name}_testclass_{test_class}_{d_name}"

    all_csv = os.path.join(out_dir, f"scores_{stem}_anom_asc.csv")
    low_csv = os.path.join(out_dir, f"scores_{stem}_lowest_anomaly_asc.csv")
    high_csv = os.path.join(out_dir, f"scores_{stem}_greatest_anomaly_asc.csv")
    write_records_csv(all_csv, records_asc)
    write_records_csv(low_csv, low_records)
    write_records_csv(high_csv, high_records)
    print(f"[CSV] saved {all_csv}")
    print(f"[CSV] saved {low_csv}")
    print(f"[CSV] saved {high_csv}")

    def score_range(rs: List[Dict[str, Any]]) -> str:
        if not rs:
            return "empty"
        return f"{rs[0]['anomaly_score']:.4f} -> {rs[-1]['anomaly_score']:.4f}"

    low_grid = os.path.join(grids_dir, f"grid_{stem}_lowest_anomaly_asc.png")
    high_grid = os.path.join(grids_dir, f"grid_{stem}_greatest_anomaly_asc.png")

    title_low = (
        f"CIFAR-10 {CIFAR10_CLASSES[test_class]} only | rule {CIFAR10_CLASSES[a]} AND {CIFAR10_CLASSES[b]} | "
        f"lowest root/anom scores | left-to-right increasing | range={score_range(low_records)} | n={len(low_records)}"
    )
    title_high = (
        f"CIFAR-10 {CIFAR10_CLASSES[test_class]} only | rule {CIFAR10_CLASSES[a]} AND {CIFAR10_CLASSES[b]} | "
        f"greatest root/anom scores | left-to-right increasing | range={score_range(high_records)} | n={len(high_records)}"
    )
    save_cifar_grid_from_records(low_records, test_ds, low_grid, grid_n, grid_cols, title_low)
    save_cifar_grid_from_records(high_records, test_ds, high_grid, grid_n, grid_cols, title_high)
    return records_asc


# -----------------------------
# Main
# -----------------------------

def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("CIFAR-10 A AND B learned-evaluator grids")
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--run_dir", type=str, default="runs/cifar_cat_dog")
    p.add_argument("--cache_dir", type=str, default="artifacts/cache/cifar10")
    p.add_argument("--leaf_ckpt", type=str, default="artifacts/checkpoints/cifar10_leaf_bank.pt")

    p.add_argument("--pairs", type=str, default="cat,dog", help="'all' or semicolon list like 'cat,dog;ship,truck' or '3,5;8,9'.")
    p.add_argument("--a", type=str, default=None, help="Optional single-pair A class name or index.")
    p.add_argument("--b", type=str, default=None, help="Optional single-pair B class name or index.")

    p.add_argument("--feat_dim", type=int, default=256)
    p.add_argument("--epochs_leaf", type=int, default=10)
    p.add_argument("--epochs_level", type=int, default=2)
    p.add_argument("--lr_leaf", type=float, default=1e-3)
    p.add_argument("--lr_level", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--batch_train", type=int, default=128)
    p.add_argument("--batch_eval", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--train_frac", type=float, default=1.0)
    p.add_argument("--max_eval_batches", type=int, default=0, help="0 = all test batches.")

    p.add_argument("--augment", action="store_true", help="Use random crop/flip for CIFAR-10 leaf/gate training images.")
    p.add_argument("--negatives", type=str, default="chimeras_only",
                   choices=["true_only", "legacy", "ad_strict", "ad_chimera_pos", "chimera_plus_true", "chimeras_only"])
    p.add_argument("--train_missing_only", action="store_true")
    p.add_argument("--force_train_leaf", action="store_true", help="Retrain leaf bank even if --leaf_ckpt exists.")
    p.add_argument("--eval_leaf", action="store_true", help="Print CIFAR-10 leaf-bank argmax accuracy on test split after train/load.")

    p.add_argument("--grid_n", type=int, default=10)
    p.add_argument("--grid_cols", type=int, default=10)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> None:
    args = build_args()
    set_seed(args.seed)
    device = device_from_arg(args.device)
    ensure_dir(args.run_dir)
    grids_dir = os.path.join(args.run_dir, "grids")
    ensure_dir(grids_dir)
    ensure_dir(args.cache_dir)
    if os.path.dirname(args.leaf_ckpt):
        ensure_dir(os.path.dirname(args.leaf_ckpt))

    pairs = parse_pairs(args)
    print(f"[Device] {device}")
    print("[Classes] " + ", ".join(f"{i}:{n}" for i, n in enumerate(CIFAR10_CLASSES)))
    print(f"[Pairs] {len(pairs)} pair(s): {[(CIFAR10_CLASSES[a], CIFAR10_CLASSES[b]) for a, b in pairs]}")

    train_ds_full = CIFAR10Concepts(root=args.data_root, train=True, download=True, augment=args.augment)
    test_ds = CIFAR10Concepts(root=args.data_root, train=False, download=True, augment=False)

    if args.train_frac < 1.0:
        n_keep = max(1, int(len(train_ds_full) * args.train_frac))
        generator = torch.Generator().manual_seed(args.seed)
        idx = torch.randperm(len(train_ds_full), generator=generator).tolist()[:n_keep]
        train_ds = Subset(train_ds_full, idx)
        print(f"[Data] train_frac={args.train_frac} -> TRAIN kept {len(train_ds)} / {len(train_ds_full)}")
    else:
        train_ds = train_ds_full

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_train, shuffle=True, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_eval, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=(device == "cuda"),
    )

    bank = CIFAR10LeafBank(feat_dim=args.feat_dim).to(device)
    if os.path.exists(args.leaf_ckpt) and not args.force_train_leaf:
        bank.load_state_dict(torch.load(args.leaf_ckpt, map_location="cpu"), strict=True)
        bank.to(device).eval()
        print(f"[LeafBank] loaded {args.leaf_ckpt}")
    else:
        print("[LeafBank] training ...")
        train_leaf_bank(
            bank, train_loader, device=device, epochs=args.epochs_leaf,
            lr=args.lr_leaf, weight_decay=args.weight_decay,
        )
        torch.save(bank.state_dict(), args.leaf_ckpt)
        print(f"[LeafBank] saved {args.leaf_ckpt}")
    bank.eval()
    if args.eval_leaf:
        evaluate_leaf_bank(bank, test_loader, device=device)

    manifest = {
        "dataset": "CIFAR-10",
        "classes": CIFAR10_CLASSES,
        "rule": "A AND B",
        "note": "CIFAR-10 is single-label; root truth is always 0 for A != B. anomaly_score = root_prob.",
        "pairs": [{"a": a, "b": b, "a_name": CIFAR10_CLASSES[a], "b_name": CIFAR10_CLASSES[b]} for a, b in pairs],
        "leaf_ckpt": os.path.abspath(args.leaf_ckpt),
        "feat_dim": args.feat_dim,
        "epochs_level": args.epochs_level,
        "negatives": args.negatives,
        "augment": bool(args.augment),
        "grid_policy": "For each pair and test class A/B, save lowest and greatest anomaly-score grids, both increasing left-to-right.",
    }
    with open(os.path.join(args.run_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    all_records: List[Dict[str, Any]] = []

    for a, b in pairs:
        a_name = CIFAR10_CLASSES[a]
        b_name = CIFAR10_CLASSES[b]
        pair_slug = f"pair_{a}_{b}_{slugify(a_name)}_{slugify(b_name)}"
        print(f"\n[Rule] {pair_slug}: {a_name} AND {b_name}")
        g = build_and_graph_for_pair(a, b)
        cache = SubtreeCache(CacheConfig(root_dir=os.path.join(args.cache_dir, pair_slug)))
        trainer = LevelwiseTrainer(g, N_concepts=N_CONCEPTS, cache=cache, feat_dim=args.feat_dim, device=device)

        print(f"[Trainer] training/loading gate for {pair_slug} ...")
        trainer.train(
            dataset=train_loader,
            leaf_bank=bank,
            epochs_per_level=args.epochs_level,
            negatives=args.negatives,
            lr=args.lr_level,
            use_soft_leaves=True,
            verbose=True,
            train_missing_only=args.train_missing_only,
        )

        print(f"[Eval] scoring test samples for only class {a_name}, then only class {b_name} ...")
        records_by_class = score_pair_on_test_classes(
            trainer=trainer,
            bank=bank,
            test_loader=test_loader,
            device=device,
            a=a,
            b=b,
            max_eval_batches=args.max_eval_batches,
        )

        for d in (a, b):
            recs = records_by_class[d]
            print(f"[Score] {pair_slug}, test class {d}:{CIFAR10_CLASSES[d]}: {len(recs)} records")
            recs_asc = save_low_high_outputs(
                records=recs,
                test_ds=test_ds,
                out_dir=args.run_dir,
                grids_dir=grids_dir,
                a=a,
                b=b,
                test_class=d,
                grid_n=args.grid_n,
                grid_cols=args.grid_cols,
            )
            all_records.extend(recs_asc)

    all_csv = os.path.join(args.run_dir, "scores_all_pairs_anom_asc.csv")
    write_records_csv(
        all_csv,
        sorted(all_records, key=lambda r: (r["pair_a"], r["pair_b"], r["test_class"], r["anomaly_score"])),
    )
    print(f"\n[Done] saved all records: {all_csv}")
    print(f"[Done] grids directory: {grids_dir}")


if __name__ == "__main__":
    main()
