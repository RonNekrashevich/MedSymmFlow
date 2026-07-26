import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from PIL import Image


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Display a balanced grid of generated normal and pneumonia samples")
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "outputs" / "pneumoniamnist"), help="Root directory with normal and pneumonia folders")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    normal_dir = output_dir / "normal"
    pneumonia_dir = output_dir / "pneumonia"

    normal_images = sorted(normal_dir.glob("*.png")) if normal_dir.exists() else []
    pneumonia_images = sorted(pneumonia_dir.glob("*.png")) if pneumonia_dir.exists() else []

    if not normal_images or not pneumonia_images:
        raise SystemExit("No generated images found. Run generate_pneumoniamnist.py first.")

    n = max(len(normal_images), len(pneumonia_images))
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 5))
    for i in range(n):
        if i < len(normal_images):
            img = Image.open(normal_images[i]).convert("RGB")
            axes[0, i].imshow(img)
            axes[0, i].axis("off")
            axes[0, i].set_title("normal")
        else:
            axes[0, i].axis("off")

        if i < len(pneumonia_images):
            img = Image.open(pneumonia_images[i]).convert("RGB")
            axes[1, i].imshow(img)
            axes[1, i].axis("off")
            axes[1, i].set_title("pneumonia")
        else:
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
