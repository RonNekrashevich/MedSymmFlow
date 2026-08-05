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
    ap.add_argument("--weights-root", default=None,
                    help="persistent dir for the 755 MB MedSymmFlow weights (default: repo dir). "
                         "Point this at the mounted volume so the download happens once.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--budgets", type=int, nargs="+", default=[250, 500, 1000])
    ap.add_argument("--quick", action="store_true", help="fast smoke test (overrides seeds/budgets)")
    ap.add_argument("--pip", action="store_true", help="pip install the light deps first")
    ap.add_argument("--filter-mode", default="keep_confident",
                    choices=["none", "keep_confident", "keep_uncertain", "random_match"])
    ap.add_argument("--filter-scorer", default="local", choices=["none", "local", "full"])
    ap.add_argument("--conf-thresh", type=float, default=0.60)
    ap.add_argument("--mem-reference", default="local", choices=["none", "local", "full"])
    ap.add_argument("--beta", type=float, default=None, help="generation label-noise amplitude")
    ap.add_argument("--run-tag", default="", help="label this run in the ledger")
    ap.add_argument("--legacy-filter", action="store_true",
                    help="reproduce the old (budget-dependent, leaky) filter semantics")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--filter-ablation", action="store_true",
                    help="after the main run, re-run the synthetic arms for every filter "
                         "mode (reuses the caches, so nearly free)")
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
        weights_root=args.weights_root,
        scratch_dir=str(out / "scratch"),
        fig_dir=str(out / "figures"),
        seeds=(None if args.quick else args.seeds),
        budgets=(None if args.quick else args.budgets),
        filter_mode=args.filter_mode,
        filter_scorer=args.filter_scorer,
        conf_thresh=args.conf_thresh,
        mem_reference=args.mem_reference,
        legacy_filter=args.legacy_filter,
        resume=not args.no_resume,
        run_tag=args.run_tag,
        **({"gen_beta": args.beta} if args.beta is not None else {}),
    )
    exp = Experiment(cfg)

    print("== data =="); print(exp.setup_data().to_string(index=False))
    print("== selftest =="); exp.selftest_repro(strict=False)   # needs train_set
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
        fp.to_csv(out / "fingerprint.csv")
        print(fp.to_string())
        print(exp.measure_c1())
    except Exception as e:  # isolated: the rest of the run already succeeded
        print("distillation_agreement failed:", repr(e))

    if args.filter_ablation:
        # Filter modes reuse the cached embeddings and scorer probabilities, so only the
        # keep-mask and the training runs change. Tests whether keeping CONFIDENT samples
        # (the published default) is actually better than keeping the hard ones.
        for mode in ("none", "keep_uncertain", "random_match"):
            print(f"\n== filter ablation: {mode} ==")
            exp.cfg.filter_mode = mode
            exp.cfg.run_tag = f"{args.run_tag}filter={mode}"
            exp._filter_cache.clear()
            exp.filter_synthetic(plot=False)
            exp.run_synthetic()
        abl = exp.summarize(select=False)[0]
        abl.to_csv(out / "filter_ablation.csv", index=False)
        print(abl.to_string(index=False))

    print("\nDONE. Outputs ->", out)


if __name__ == "__main__":
    main()
