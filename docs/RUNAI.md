# Running the PneumoniaMNIST experiment on TAU Run:AI

Colab is out of GPU quota, so this runs the same experiment on the TAU Run:AI
cluster. The experiment is headless-ready: `project/run_experiment.py` runs the
whole pipeline and writes results + figures to a mounted volume.

Reference: https://www.tau.ac.il/~udi/run-ai/ (install, login, Jupyter, Q&A pages).

---

## Who does what

**I have prepared (in this repo):**
- `project/run_experiment.py` — headless entrypoint (baselines, generation,
  filtering, S1/S2/S3, D1, C1, summary, distillation fingerprint) that saves
  `summary.csv`, `comparison.csv`, `fingerprint.csv`, and `figures/*.png`.
- Portable paths: `Config(save_dir=..., medsymm_root=..., scratch_dir=..., fig_dir=...)`.
- Matplotlib forced to a non-display backend, so it works on a compute node.

**You must do (needs your TAU credentials / machine / VPN — I cannot):**
- Install Docker + the `runai` CLI, log in, submit jobs.

---

## 1. One-time setup (your machine, university VPN on)

Follow the TAU install pages, then (Ubuntu):

```bash
source /etc/profile.d/runai.sh
kube-config
runai login
runai config project <YOUR_PROJECT_NAME>
```

macOS: `source /usr/local/bin/runai.zsh; rehash` instead of the first line.
Verify with `runai list`.

---

## 2. First: an interactive smoke test (get a GPU shell)

```bash
runai submit --name msf-jup -g 1 --pvc=storage:/storage -i <IMAGE> --interactive --working-dir /storage
runai port-forward msf-jup --port 8888:8888     # wait for "forwarding" message
runai logs msf-jup                              # copy the http://...?token=... URL
```

Open that URL, launch a Terminal in Jupyter, and run:

```bash
cd /storage
git clone https://github.com/RonNekrashevich/MedSymmFlow.git
cd MedSymmFlow
python project/run_experiment.py --pip --quick --out /storage/medsymm_out
```

`--quick` finishes in minutes and confirms the environment (GPU, deps, generation,
all arms). If it prints `DONE`, you're ready for the full run.

---

## 3. Then: the full run (batch job, unattended)

```bash
runai submit --name msf-run -g 1 --pvc=storage:/storage -i <IMAGE> -- \
  bash -lc "cd /storage && (test -d MedSymmFlow || git clone https://github.com/RonNekrashevich/MedSymmFlow.git) && cd MedSymmFlow && git pull -q && python project/run_experiment.py --pip --seeds 0 1 2 3 4 --budgets 250 500 1000 --out /storage/medsymm_out"

runai list                 # STATUS: Running -> Succeeded
runai logs msf-run -f      # follow progress
```

Outputs land in `/storage/medsymm_out/`:
`summary.csv`, `comparison.csv`, `fingerprint.csv`, `figures/gain_vs_budget.png`,
`figures/memorisation_screen.png`, `figures/synthetic_samples.png`, `results.csv`.

Retrieve them via the TAU `samba_`/`sshfs_` mount pages, or `kube_bash_storage`.

---

## 4. Confirm these with TAU before running (cluster-specific, I can't know them)

1. **Project name** for `runai config project`.
2. **Image (`-i`)** — must have **CUDA + PyTorch**. The docs' `uuddii/helo` is a
   Jupyter demo image; verify it has GPU PyTorch, or use a CUDA PyTorch image.
   `--pip` installs the lighter deps; it does *not* install torch/CUDA.
3. **Internet on compute nodes.** The job does `git clone`, `pip install`, downloads
   the 755 MB MSF weights (Zenodo) and the MedMNIST npz. If nodes are air-gapped,
   pre-stage all of this into `/storage` from a node that has internet, then drop
   `--pip` and point the run at the pre-downloaded files.
4. **GPU quota and max job time** — a 5-seed run is a few hours; make sure that fits.

To trim compute: fewer seeds (`--seeds 0 1 2`) or budgets, or `--quick`.
