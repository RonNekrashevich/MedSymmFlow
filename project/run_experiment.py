"""Headless entrypoint for the PneumoniaMNIST augmentation experiment.

Runs the full pipeline (baselines B0-B2, generation, filtering, synthetic arms
S1-S3, D1 diagnostic, C1 reference, summary, distillation fingerprint) and writes
all results + figures to --out. Designed for a batch job on a GPU cluster (Run:AI)
where there is no interactive display and outputs must land on a mounted volume.

Prereqs on the node: the repo present (this file lives in it), a CUDA PyTorch, and
the light deps (medmnist torchdiffeq diffusers accelerate zuko scikit-learn scipy
loguru python-dotenv datasets). Install them in the image or via --pip.

Example (from the repo root, GPU available):
    python project/run_experiment.py --out /storage/medsymm_out --seeds 0 1 2 3 4
"""
import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: no display

HERE = Path(__file__).resolve().parent          # .../MedSymmFlow/project
REPO = HERE.parent                              # .../MedSymmFlow
sys.path.insert(0, str(HERE))

LIGHT_DEPS = ["medmnist", "torchdiffeq", "diffusers", "accelerate", "zuko",
              "scikit-learn", "scipy", "loguru", "python-dotenv", "datasets"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/storage/medsymm_out", help="output dir on the mounted volume")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 500, 1000])
    ap.add_argument("--quick", action="store_true", help="fast smoke test (overrides seeds/budgets)")
    ap.add_argument("--pip", action="store_true", help="pip install the light deps first")
    args = ap.parse_args()

    if args.pip:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *LIGHT_DEPS], check=True)

    from augmentation import Experiment, Config

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = Config(
        quick=args.quick,
        save_dir=str(out),
        medsymm_root=str(REPO),
        scratch_dir=str(out / "scratch"),
        fig_dir=str(out / "figures"),
        seeds=(None if args.quick else args.seeds),
        budgets=(None if args.quick else args.budgets),
    )
    exp = Experiment(cfg)

    print("== data =="); print(exp.setup_data().to_string(index=False))
    print("== baselines =="); exp.run_baselines()
    print("== generation =="); exp.download_weights(); exp.generate_synthetic(); exp.visualize_samples()
    print("== filtering =="); exp.filter_synthetic()
    print("== synthetic arms =="); exp.run_synthetic()
    print("== D1 diagnostic =="); exp.run_diagnostic_d1()
    print("== C1 reference =="); exp.record_c1()

    print("== summary ==")
    summary, comparison = exp.summarize()
    summary.to_csv(out / "summary.csv", index=False)
    comparison.to_csv(out / "comparison.csv", index=False)
    print(summary.to_string(index=False))
    print("\n== synthetic vs strongest baseline ==")
    print(comparison.to_string(index=False))
    exp.plot(summary)

    print("\n== distillation fingerprint ==")
    try:
        fp = exp.distillation_agreement()
        fp.to_csv(out / "fingerprint.csv", index=False)
        print(fp.to_string(index=False))
    except Exception as e:  # isolated: the rest of the run already succeeded
        print("distillation_agreement failed:", repr(e))

    print("\nDONE. Outputs ->", out)


if __name__ == "__main__":
    main()
