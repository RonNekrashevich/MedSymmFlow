# Running on Run:ai

Everything is parameterised by two variables, so you only edit them once:

| Variable | What it is | Example |
|---|---|---|
| `PVC` | your Run:ai persistent volume claim | `my-lab-pvc` |
| `DATA_ROOT` | path *inside the container* where that volume is mounted, plus a project folder | `/storage/medsymm` |

---

## 0. Find your storage (do this first)

Nothing else works until you know which volume you get and where it lands.

```bash
runai submit probe -i ubuntu --command -- bash -c "df -h; echo ---; ls -la /; echo ---; mount | grep -Ev 'proc|sys|cgroup|dev'"
runai logs probe
runai delete job probe
```

Look for a large mount that is **not** `overlay` or `tmpfs` — that is your persistent storage.
If nothing is mounted by default, list the claims you can attach:

```bash
runai list projects
kubectl get pvc            # if you have kubectl access
```

Then set, for example, `PVC=my-lab-pvc` and `DATA_ROOT=/storage/medsymm`.

---

## 1. Smoke test (~10 minutes, 1 GPU)

Confirms the image, the volume, the weights download and the pipeline all work.

```bash
runai submit medsymm-smoke \
  -i nvcr.io/nvidia/pytorch:24.12-py3 \
  -g 1 \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/medsymm \
  -e RUN_NAME=smoke \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --quick"
```

```bash
runai logs medsymm-smoke -f
```

You want to see, in order: a GPU name, `Split sizes OK`, `selftest_repro`, the baselines,
`Checkpoint present`, the arms, and `DONE`.

**The first job downloads 755 MB of weights. Every later job reuses them** — that is the
whole point of putting `--weights-root` on the volume.

---

## 2. The real run (5 budgets x 5 seeds)

```bash
runai submit medsymm-full \
  -i nvcr.io/nvidia/pytorch:24.12-py3 \
  -g 1 \
  --pvc my-lab-pvc:/storage \
  -e DATA_ROOT=/storage/medsymm \
  -e RUN_NAME=full-5seed \
  --command -- bash -c \
  "git clone -q https://github.com/RonNekrashevich/MedSymmFlow.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --seeds 0 1 2 3 4 --budgets 250 500 1000 2000 4708"
```

Resume is automatic: results are appended to a ledger keyed by
`(arm, budget, seed, filter_key)`, so a job that is pre-empted or killed can simply be
re-submitted with the same `RUN_NAME` and it continues where it stopped.

---

## 3. Parallel sweeps — one job per configuration

This is what the cluster is actually for. Each job writes to its own `RUN_NAME`, so they
never collide; combine the ledgers afterwards.

```bash
# filter-direction ablation: is keeping CONFIDENT samples backwards?
for mode in keep_confident keep_uncertain random_match none; do
  runai submit "medsymm-filter-$mode" \
    -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 \
    --pvc my-lab-pvc:/storage \
    -e DATA_ROOT=/storage/medsymm -e "RUN_NAME=filter-$mode" \
    --command -- bash -c \
    "git clone -q https://github.com/RonNekrashevich/MedSymmFlow.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --seeds 0 1 2 3 4 --budgets 500 --filter-mode $mode --run-tag filter=$mode"
done

# beta sweep: does sharper class conditioning help downstream?
for b in 1 2 4 6; do
  runai submit "medsymm-beta$b" \
    -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 \
    --pvc my-lab-pvc:/storage \
    -e DATA_ROOT=/storage/medsymm -e "RUN_NAME=beta-$b" \
    --command -- bash -c \
    "git clone -q https://github.com/RonNekrashevich/MedSymmFlow.git /workspace/m && bash /workspace/m/project/runai/entrypoint.sh --seeds 0 1 2 --budgets 500 --beta $b --run-tag beta=$b"
done
```

Each `beta` job regenerates its own synthetic pool, so give them separate `RUN_NAME`s —
they must not share a synthetic directory.

---

## 4. Collecting results

Ledgers are plain CSV. Concatenate and analyse them anywhere, including on your laptop:

```python
import pandas as pd, glob
df = pd.concat([pd.read_csv(f) for f in glob.glob("/storage/medsymm/runs/*/results.csv")])
df.to_csv("all_results.csv", index=False)

import sys; sys.path.insert(0, "project")
from paired_stats import paired_tests_from_csv
paired_tests_from_csv(df, select={"run_tag": ""})   # select= avoids mixing configurations
```

`paired_tests_from_csv` raises on duplicate `(arm, budget, seed)` rows rather than silently
averaging them, so pass `select=` to pick one configuration at a time.

---

## Notes and gotchas

- **Image.** `nvcr.io/nvidia/pytorch:24.12-py3` already contains CUDA PyTorch; the entrypoint
  installs only the small extras. If your cluster blocks NGC, `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`
  works too.
- **`--pvc` syntax** varies by Run:ai version: newer builds use
  `--existing-pvc claimname=my-lab-pvc,path=/storage`. Check `runai submit --help`.
- **One GPU is enough.** Nothing here is distributed; parallelism comes from running many
  single-GPU jobs, not from multi-GPU jobs.
- **Interactive debugging:** `runai submit -i nvcr.io/nvidia/pytorch:24.12-py3 -g 1 --interactive --attach --command -- bash`
- **The weights download is the slowest first step.** If the cluster has no outbound internet,
  copy `models.zip` onto the volume manually into `$DATA_ROOT/weights/` and the entrypoint
  will unpack it instead of downloading.
