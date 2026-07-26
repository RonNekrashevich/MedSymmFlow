import argparse
import os
import re
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="List checkpoint files under an extracted MedSymmFlow models directory")
    parser.add_argument("--models_dir", type=str, required=True, help="Path to the extracted models directory")
    return parser.parse_args()


def infer_metadata(path: Path):
    name = path.name
    parts = name.replace(".pt", "").replace(".pth", "").replace(".ckpt", "")
    lower = parts.lower()

    dataset = "unknown"
    if "pneumoniamnist" in lower:
        dataset = "pneumoniamnist"
    elif "bloodmnist" in lower:
        dataset = "bloodmnist"
    elif "dermamnist" in lower:
        dataset = "dermamnist"
    elif "retinamnist" in lower:
        dataset = "retinamnist"

    latent = "latent" if lower.startswith("latfm") or "latfm" in lower else "non-latent"
    rgb = "RGB" if "_rgb" in lower or "rgb" in lower else "grayscale"

    beta = None
    beta_match = re.search(r"beta([0-9]+(?:\.[0-9]+)?)", lower)
    if beta_match:
        beta = beta_match.group(1)

    epoch = None
    epoch_match = re.search(r"epoch([0-9]+)", lower)
    if epoch_match:
        epoch = epoch_match.group(1)

    return {
        "path": str(path),
        "filename": name,
        "dataset": dataset,
        "latent": latent,
        "rgb": rgb,
        "beta": beta,
        "epoch": epoch,
    }


def main():
    args = parse_args()
    models_dir = Path(args.models_dir).resolve()

    if not models_dir.exists():
        raise SystemExit(f"Models directory not found: {models_dir}")

    checkpoints = []
    for path in sorted(models_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            checkpoints.append(infer_metadata(path))

    if not checkpoints:
        print("No checkpoint files found.")
        return

    print("{:<120} | {:<40} | {:<20} | {:<12} | {:<10} | {:<8} | {:<8}".format(
        "full_checkpoint_path",
        "filename",
        "dataset",
        "latent",
        "RGB",
        "beta",
        "epoch",
    ))
    print("-" * 170)
    for entry in checkpoints:
        highlight = "<-- PNEUMONIAMNIST" if entry["dataset"] == "pneumoniamnist" else ""
        print("{:<120} | {:<40} | {:<20} | {:<12} | {:<10} | {:<8} | {:<8} {}".format(
            entry["path"],
            entry["filename"],
            entry["dataset"],
            entry["latent"],
            entry["rgb"],
            entry["beta"] if entry["beta"] is not None else "",
            entry["epoch"] if entry["epoch"] is not None else "",
            highlight,
        ))


if __name__ == "__main__":
    main()
