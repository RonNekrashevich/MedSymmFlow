import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
MEDSYMMFLOW_ROOT = SRC_ROOT / "medsymmflow"
for candidate in [str(SRC_ROOT), str(MEDSYMMFLOW_ROOT)]:
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from medsymmflow.models.SymmFMClass import SymmFMClass
from medsymmflow.data.Dataloaders import pick_dataset
from medsymmflow.utils.util import parse_args_SymmetricFlowMatchingClass


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Generate PneumoniaMNIST-like synthetic samples with MedSymmFlow")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the pretrained MedSymmFlow checkpoint")
    parser.add_argument("--num_normal", type=int, default=8, help="Number of normal class samples to generate")
    parser.add_argument("--num_pneumonia", type=int, default=8, help="Number of pneumonia class samples to generate")
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "outputs" / "pneumoniamnist"), help="Root output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--beta", type=float, default=4.0, help="Mask dequantization beta")
    parser.add_argument("--image_size", type=int, default=256, help="Image size")
    parser.add_argument("--solver", type=str, default="euler", help="ODE solver")
    parser.add_argument("--step_size", type=float, default=0.04, help="ODE solver step size")
    parser.add_argument("--dataset", type=str, default="pneumoniamnist", help="Dataset name")
    parser.add_argument("--model_channels", type=int, default=128, help="Model channel count")
    parser.add_argument("--num_res_blocks", type=int, default=2, help="Residual blocks")
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 2, 2], help="Channel multipliers")
    parser.add_argument("--num_heads", type=int, default=4, help="Attention heads")
    parser.add_argument("--num_head_channels", type=int, default=64, help="Attention head channels")
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=[2], help="Attention resolutions")
    parser.add_argument("--solver_lib", type=str, default="torchdiffeq", help="Solver library")
    parser.add_argument("--rgb_mask", action="store_true", default=False, help="Use RGB masks")
    parser.add_argument("--latent", action="store_true", default=False, help="Use latent implementation")
    parser.add_argument("--n_classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers")
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_images(samples: torch.Tensor, output_dir: Path, prefix: str, checkpoint_path: str, requested_class: int, requested_class_name: str, seed: int, beta: float, rgb_mask: bool, image_size: int, solver: str, step_size: float, metadata_rows: list) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, sample in enumerate(samples):
        image_tensor = sample.detach().cpu().float().clamp(0.0, 1.0)
        image = image_tensor.permute(1, 2, 0).numpy()
        if image.shape[2] == 1:
            image = image[:, :, 0]
            pil_image = Image.fromarray((image * 255).astype("uint8"), mode="L")
        else:
            pil_image = Image.fromarray((image * 255).astype("uint8"))
        image_path = output_dir / f"{prefix}_{idx:03d}.png"
        pil_image.save(image_path)
        metadata_rows.append({
            "image_path": str(image_path),
            "requested_class": requested_class,
            "requested_class_name": requested_class_name,
            "seed": seed,
            "checkpoint_path": checkpoint_path,
            "beta": beta,
            "rgb_mask": rgb_mask,
            "image_size": image_size,
            "solver": solver,
            "step_size": step_size,
        })


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    set_seed(args.seed)

    output_root = Path(args.output_dir)
    normal_dir = output_root / "normal"
    pneumonia_dir = output_root / "pneumonia"
    normal_dir.mkdir(parents=True, exist_ok=True)
    pneumonia_dir.mkdir(parents=True, exist_ok=True)

    image_shape, channels, _ = pick_dataset(args.dataset, "val", args.image_size, args.num_workers, args.num_workers)
    model_args = parse_args_SymmetricFlowMatchingClass()
    model_args.train = False
    model_args.sample = True
    model_args.dataset = args.dataset
    model_args.checkpoint = args.checkpoint
    model_args.num_samples = 1
    model_args.n_classes = args.n_classes
    model_args.batch_size = 1
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

    metadata_rows = []
    if args.num_normal > 0:
        labels = torch.tensor([0] * args.num_normal, dtype=torch.long, device=model.device)
        samples = model.sample(args.num_normal, labels=labels, train=False, fid=True)
        save_images(samples, normal_dir, "normal", args.checkpoint, 0, "normal", args.seed, args.beta, args.rgb_mask, args.image_size, args.solver, args.step_size, metadata_rows)

    if args.num_pneumonia > 0:
        labels = torch.tensor([1] * args.num_pneumonia, dtype=torch.long, device=model.device)
        samples = model.sample(args.num_pneumonia, labels=labels, train=False, fid=True)
        save_images(samples, pneumonia_dir, "pneumonia", args.checkpoint, 1, "pneumonia", args.seed, args.beta, args.rgb_mask, args.image_size, args.solver, args.step_size, metadata_rows)

    metadata_path = output_root / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "requested_class", "requested_class_name", "seed", "checkpoint_path", "beta", "rgb_mask", "image_size", "solver", "step_size"])
        writer.writeheader()
        writer.writerows(metadata_rows)

    print(f"Saved {len(metadata_rows)} images to {output_root}")
    print(f"Metadata written to {metadata_path}")


if __name__ == "__main__":
    main()
