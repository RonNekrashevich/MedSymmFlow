"""Classify the real PneumoniaMNIST test split with MedSymmFlow's own reverse-flow
classifier (the discriminative direction of the shared velocity field) and write
per-image predictions to CSV.

Used by project/augmentation.py to compute the distillation-agreement fingerprint:
does a synthetic-trained ResNet copy MSF's predictions -- especially MSF's errors --
more than a real-trained one? Mirrors generate_pneumoniamnist.py's model setup.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MEDSYMMFLOW_ROOT = SRC_ROOT / "medsymmflow"
for candidate in [str(SRC_ROOT), str(MEDSYMMFLOW_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.data.Dataloaders import pick_dataset
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass
from medmnist import PneumoniaMNIST


def build_arg_parser():
    p = argparse.ArgumentParser(description="Classify PneumoniaMNIST test split with MedSymmFlow")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--output_csv", type=str, required=True)
    p.add_argument("--dataset", type=str, default="pneumoniamnist")
    p.add_argument("--n_classes", type=int, default=2)
    p.add_argument("--image_size", type=int, default=32)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--solver", type=str, default="euler")
    p.add_argument("--step_size", type=float, default=0.04)
    p.add_argument("--solver_lib", type=str, default="torchdiffeq")
    p.add_argument("--model_channels", type=int, default=64)
    p.add_argument("--num_res_blocks", type=int, default=2)
    p.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2])
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_head_channels", type=int, default=64)
    p.add_argument("--attention_resolutions", type=int, nargs="+", default=[2])
    p.add_argument("--rgb_mask", action="store_true", default=False)
    p.add_argument("--latent", action="store_true", default=False)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--split", type=str, default="test")
    return p


def main():
    args = build_arg_parser().parse_args()

    image_shape, channels, _ = pick_dataset(args.dataset, "val", args.image_size, 1, args.num_workers)

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]
        model_args = parse_args_SymmetricFlowMatchingClass()
    finally:
        sys.argv = original_argv
    model_args.train = False
    model_args.sample = False
    model_args.classification = True
    model_args.dataset = args.dataset
    model_args.checkpoint = args.checkpoint
    model_args.n_classes = args.n_classes
    model_args.batch_size = args.batch_size
    model_args.num_workers = args.num_workers
    model_args.model_channels = args.model_channels
    model_args.num_res_blocks = args.num_res_blocks
    model_args.channel_mult = tuple(args.channel_mult)
    model_args.attention_resolutions = tuple(args.attention_resolutions)
    model_args.num_heads = args.num_heads
    model_args.num_head_channels = args.num_head_channels
    model_args.solver_lib = args.solver_lib
    model_args.solver = args.solver
    model_args.step_size = args.step_size
    model_args.beta = args.beta
    model_args.rgb_mask = args.rgb_mask
    model_args.latent = args.latent
    model_args.size = args.image_size

    model = SymmFMClass(model_args, image_shape, channels)
    model.load_checkpoint(args.checkpoint)
    model.eval()

    # Match the repo's pneumoniamnist preprocessing: grayscale, resize to image_size,
    # normalise to [-1, 1]. size=64 source -> image_size mirrors pick_dataset's choice.
    tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    ds = PneumoniaMNIST(split=args.split, download=True, size=64)

    trues, preds = [], []
    batch_x, batch_y = [], []

    @torch.no_grad()
    def flush():
        if not batch_x:
            return
        x = torch.stack(batch_x).to(model.device)
        mask = model.segment(x.shape[0], x, train=False, eval=True)
        mean_pred, _ = model.quantize_class(mask)
        preds.extend(mean_pred.long().cpu().numpy().tolist())
        trues.extend(batch_y)
        batch_x.clear()
        batch_y.clear()

    for i in range(len(ds)):
        img, label = ds[i]
        batch_x.append(tf(img))
        batch_y.append(int(np.asarray(label).reshape(-1)[0]))
        if len(batch_x) >= args.batch_size:
            flush()
    flush()

    trues, preds = np.array(trues), np.array(preds)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "true", "msf_pred"])
        for i, (t, p) in enumerate(zip(trues, preds)):
            writer.writerow([i, int(t), int(p)])

    acc = float((trues == preds).mean())
    print(f"MSF test accuracy: {acc:.4f} over {len(trues)} images -> {args.output_csv}")


if __name__ == "__main__":
    main()
