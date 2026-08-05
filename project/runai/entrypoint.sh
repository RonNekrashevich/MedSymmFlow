#!/usr/bin/env bash
# Run:ai job entrypoint for the PneumoniaMNIST augmentation experiment.
#
# Everything persistent lives under $DATA_ROOT (your mounted volume):
#     $DATA_ROOT/weights   the 755 MB MedSymmFlow archive, downloaded ONCE
#     $DATA_ROOT/pipcache  pip wheel cache, so later jobs start faster
#     $DATA_ROOT/runs/$RUN_NAME   results.csv ledger, figures, models, caches
#
# The container filesystem is treated as disposable: the repo is cloned fresh each
# job, so a job always runs the current code.
#
# Usage inside a Run:ai job:
#     bash project/runai/entrypoint.sh --seeds 0 1 2 3 4 --budgets 250 500 1000
# Any arguments are passed straight through to project/run_experiment.py.
set -euo pipefail

: "${DATA_ROOT:?set DATA_ROOT to your mounted volume, e.g. -e DATA_ROOT=/storage/medsymm}"
REPO_URL="${REPO_URL:-https://github.com/RonNekrashevich/MedSymmFlow.git}"
REPO_DIR="${REPO_DIR:-/workspace/MedSymmFlow}"
RUN_NAME="${RUN_NAME:-run}"

echo "=== Run:ai job: $RUN_NAME ==="
echo "DATA_ROOT=$DATA_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "WARNING: no GPU visible"

mkdir -p "$DATA_ROOT/weights" "$DATA_ROOT/pipcache" "$DATA_ROOT/runs/$RUN_NAME"
export PIP_CACHE_DIR="$DATA_ROOT/pipcache"
export HF_HOME="$DATA_ROOT/hf"          # keeps diffusers/datasets caches off the container
export MPLBACKEND=Agg                   # headless matplotlib

# ---- code: always current -----------------------------------------------------
if [ -d "$REPO_DIR/.git" ]; then
  git -C "$REPO_DIR" fetch -q --all && git -C "$REPO_DIR" reset -q --hard origin/main
else
  git clone -q "$REPO_URL" "$REPO_DIR"
fi
echo "repo at $(git -C "$REPO_DIR" rev-parse --short HEAD)"

# ---- deps: torch comes from the image, these do not ---------------------------
python -m pip install -q --no-input \
  medmnist torchdiffeq diffusers accelerate zuko scikit-learn scipy \
  loguru python-dotenv datasets

# ---- run ----------------------------------------------------------------------
cd "$REPO_DIR"
exec python project/run_experiment.py \
  --out "$DATA_ROOT/runs/$RUN_NAME" \
  --weights-root "$DATA_ROOT/weights" \
  "$@"
