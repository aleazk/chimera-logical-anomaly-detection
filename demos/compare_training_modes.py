#!/usr/bin/env python3
"""Run identical forbidden-conjunction demos with and without Chimera training."""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
import subprocess
import sys
from typing import Iterable

from PIL import Image, ImageDraw

CIFAR_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CIFAR_TO_ID = {name: i for i, name in enumerate(CIFAR_CLASSES)}
CIFAR_TO_ID.update({"car": 1, "auto": 1, "plane": 0, "aeroplane": 0})


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def first_match(patterns: Iterable[str]) -> Path:
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return Path(matches[0])
    raise FileNotFoundError("No generated grid matched: " + ", ".join(patterns))


def labelled(image: Image.Image, label: str) -> Image.Image:
    margin = 34
    canvas = Image.new("RGB", (image.width, image.height + margin), "white")
    canvas.paste(image.convert("RGB"), (0, margin))
    ImageDraw.Draw(canvas).text((10, 9), label, fill="black")
    return canvas


def assemble(paths: list[tuple[Path, str]], output: Path) -> None:
    imgs = [labelled(Image.open(path), label) for path, label in paths]
    width = max(img.width for img in imgs)
    height = max(img.height for img in imgs)
    canvas = Image.new("RGB", (2 * width, 2 * height), "white")
    for index, img in enumerate(imgs):
        x = (index % 2) * width
        y = (index // 2) * height
        canvas.paste(img, (x, y))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print(f"[Done] comparison image: {output}")


def parse_cifar_pair(pair: str) -> tuple[int, int, str, str]:
    left, right = [part.strip().lower() for part in pair.split(",", 1)]
    def index(token: str) -> int:
        if token.isdigit():
            return int(token)
        return CIFAR_TO_ID[token]
    a, b = index(left), index(right)
    if a > b:
        a, b = b, a
    return a, b, CIFAR_CLASSES[a], CIFAR_CLASSES[b]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--pair", default="1,7")
    parser.add_argument("--root", default="runs/comparison")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--epochs_leaf", type=int, default=None)
    parser.add_argument("--epochs_level", type=int, default=2)
    parser.add_argument("--train_frac", type=float, default=1.0)
    parser.add_argument("--max_eval_batches", type=int, default=0)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    script = repo / "demos" / (
        "mnist_forbidden_conjunction.py" if args.dataset == "mnist"
        else "cifar10_forbidden_conjunction.py"
    )
    root = Path(args.root) / args.dataset / args.pair.replace(",", "_")
    leaf_ckpt = root / "shared_leaf_bank.pt"
    cache_root = root / "cache"

    for mode in ("true_only", "chimeras_only"):
        run_dir = root / mode
        cmd = [
            sys.executable, str(script),
            "--pairs", args.pair,
            "--negatives", mode,
            "--run_dir", str(run_dir),
            "--cache_dir", str(cache_root / mode),
            "--leaf_ckpt", str(leaf_ckpt),
            "--device", args.device,
            "--epochs_level", str(args.epochs_level),
            "--train_frac", str(args.train_frac),
            "--max_eval_batches", str(args.max_eval_batches),
        ]
        if args.epochs_leaf is not None:
            cmd.extend(["--epochs_leaf", str(args.epochs_leaf)])
        if args.dataset == "cifar10":
            cmd.extend(["--augment", "--eval_leaf"])
        run(cmd)

    if args.dataset == "mnist":
        a, b = sorted(int(x.strip()) for x in args.pair.split(",", 1))
        focus = b
        stem = f"pair_{a}_{b}_testdigit_{focus}"
    else:
        a, b, aname, bname = parse_cifar_pair(args.pair)
        focus = b
        stem = f"pair_{a}_{b}_{aname}_{bname}_testclass_{focus}_{bname}"

    panels = []
    for mode, label in (("true_only", "Same-image training"), ("chimeras_only", "Chimera training")):
        grid_dir = root / mode / "grids"
        low = first_match([str(grid_dir / f"grid_{stem}_lowest_anomaly_asc.png")])
        high = first_match([str(grid_dir / f"grid_{stem}_greatest_anomaly_asc.png")])
        panels.extend([(low, f"{label}: lowest scores"), (high, f"{label}: highest scores")])

    assemble(panels, root / "comparison.png")


if __name__ == "__main__":
    main()
