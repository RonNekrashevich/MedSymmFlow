"""Train the MedSymmFlow generator FROM SCRATCH on a disjoint part of PneumoniaMNIST.

Split hygiene: data_split.gen_clf_split divides the 4708-image train split once
(stratified, seeded) into a generator set and a classifier pool. This script trains
on the generator set only; augmentation.py (Config gen_frac) restricts every
classifier arm to the complement, so the generator and the ResNet never see the
same real image.

Architecture, resolution (28px source resized to 32) and beta match the published
RGB_28 checkpoint, so the existing generation and C1-classification subprocess
calls work on the resulting checkpoint unchanged.

Example:
    python project/train_msf_scratch.py \
        --out /storage/ronne/medsymm/weights/scratch/FM_pneumoniamnist_scratch_g0.5_ss0_beta4.0_rgb.pt \
        --gen-frac 0.5 --split-seed 0 --epochs 600
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MEDSYMMFLOW_ROOT = SRC_ROOT / "medsymmflow"
for candidate in [str(SRC_ROOT), str(MEDSYMMFLOW_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

import numpy as np
from medmnist import PneumoniaMNIST

from data_split import gen_clf_split, split_fingerprint
from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass
from config import models_dir


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train MSF from scratch on the generator half")
    p.add_argument("--out", type=str, required=True, help="final checkpoint destination (.pt)")
    p.add_argument("--gen-frac", type=float, default=0.5)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--snapshots", type=int, default=10)
    p.add_argument("--sample-freq", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=2)
    return p


def main():
    args = build_arg_parser().parse_args()
    # OneCycleLR needs a non-empty phase on both sides of the warmup peak.
    assert args.epochs >= 2, "need at least 2 epochs (OneCycleLR warmup + anneal)"
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Same preprocessing as the repo's pneumoniamnist loaders: 28px source, resize
    # to 32, normalize to [-1, 1].
    tf = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    full_train = PneumoniaMNIST(split="train", download=True, transform=tf)
    assert len(full_train) == 4708, len(full_train)
    labels = np.array(full_train.labels).reshape(-1)
    gen_idx, clf_idx = gen_clf_split(labels, args.gen_frac, args.split_seed)
    gen_labels = labels[gen_idx]
    print(f"disjoint split (seed {args.split_seed}): generator {len(gen_idx)} "
          f"(normal {int((gen_labels == 0).sum())} / pneumonia {int((gen_labels == 1).sum())}), "
          f"classifier pool {len(clf_idx)} — held out from this training entirely")

    train_loader = DataLoader(Subset(full_train, gen_idx), batch_size=args.batch_size,
                              shuffle=True, pin_memory=True, num_workers=args.num_workers)
    # Genuine val split, used only for the periodic sample visualisation during
    # training (the upstream repo used the TEST split here — kept out on purpose).
    val_set = PneumoniaMNIST(split="val", download=True, transform=tf)
    val_loader = DataLoader(val_set, batch_size=16, shuffle=True, pin_memory=True)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv
    model_args.train = True
    model_args.dataset = "pneumoniamnist"
    model_args.n_classes = 2
    model_args.batch_size = args.batch_size
    model_args.n_epochs = args.epochs
    model_args.lr = args.lr
    # OneCycleLR needs pct_start = warmup/n_epochs <= 1, so cap for short smoke runs.
    model_args.warmup = min(args.warmup, max(1, args.epochs // 2))
    model_args.beta = args.beta
    model_args.rgb_mask = True
    model_args.size = 32
    model_args.no_wandb = True
    model_args.num_workers = args.num_workers
    model_args.snapshots = max(1, min(args.snapshots, args.epochs))
    model_args.sample_and_save_freq = args.sample_freq
    # Architecture of the published RGB_28 checkpoint (from its state-dict shapes).
    model_args.model_channels = 64
    model_args.num_res_blocks = 2
    model_args.channel_mult = (1, 2, 2, 2)
    model_args.num_heads = 4
    model_args.num_head_channels = 64
    model_args.attention_resolutions = (2,)

    model = SymmFMClass(model_args, 32, 1)
    n_params = sum(p.numel() for p in model.model.parameters())
    print(f"training from scratch: {n_params/1e6:.1f}M params, {args.epochs} epochs, "
          f"{len(train_loader)} steps/epoch, batch {args.batch_size}")

    start = time.time()
    model.train_model(train_loader, val_loader)

    # train_model saved EMA snapshots under models_dir; pick the newest one this run
    # produced (mtime >= start guards against stale files from earlier runs).
    snap_dir = Path(models_dir) / "SymmetricalFlowMatchingClass"
    pattern = f"FM_pneumoniamnist_beta{args.beta}_rgb_epoch*.pt"
    snaps = [p for p in snap_dir.glob(pattern) if p.stat().st_mtime >= start - 60]
    assert snaps, f"no fresh snapshot matching {pattern} in {snap_dir}"
    latest = max(snaps, key=lambda p: int(p.stem.split("epoch")[1]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(latest, out)
    manifest = {
        "gen_frac": args.gen_frac, "split_seed": args.split_seed,
        "n_gen": len(gen_idx), "n_clf_pool": len(clf_idx),
        "gen_idx_sha1": split_fingerprint(gen_idx),
        "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
        "beta": args.beta, "seed": args.seed, "source_snapshot": latest.name,
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out.with_suffix(".json").write_text(json.dumps(manifest, indent=2))
    print(f"checkpoint -> {out}  (from {latest.name}, "
          f"{(time.time() - start)/3600:.2f} h)")
    print(f"manifest   -> {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
