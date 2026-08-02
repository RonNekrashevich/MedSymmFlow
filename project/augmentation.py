"""PneumoniaMNIST + MedSymmFlow synthetic-augmentation experiment.

Imported by notebooks/pneumoniamnist_augmentation.ipynb so that logic fixes land
via `git pull` instead of a notebook re-upload. The notebook holds only config,
narrative, and calls; all behaviour lives here.

Protocol: pneumoniamnist_augmentation_protocol v2.0.
Arms: B0/B1/B2 (baselines), S1/S2/S3 (synthetic), C1 (MSF reference).
"""
import hashlib
import io
import json
import os
import shutil
import subprocess
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from scipy import stats
from medmnist import PneumoniaMNIST

# Dependency-light stats module, also runnable standalone against results.csv.
from paired_stats import paired_tests_from_csv
# Deterministic filtering: pure-numpy keep-mask maths + cache keys (no torch).
import filtering as flt

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Config:
    """Experiment knobs. `quick=True` is a fast smoke test; False is the real run."""

    def __init__(self, quick=True, save_dir="/content/drive/MyDrive/MedSymmFlow_Project",
                 medsymm_root="/content/MedSymmFlow", image_size=28, gen_image_size=32,
                 use_amp=True, budgets=None, seeds=None, epochs=None, syn_per_class=None,
                 scratch_dir="/content", fig_dir=None, **overrides):
        self.quick = quick
        self.save_dir = Path(save_dir)
        self.medsymm_root = medsymm_root
        self.scratch_dir = scratch_dir        # ephemeral temp (Colab: /content; cluster: PVC)
        self.fig_dir = fig_dir                # if set, figures are also saved here (headless batch)
        self.image_size = image_size          # classifier / real-data resolution
        self.gen_image_size = gen_image_size  # MSF RGB_28 checkpoint trains at 32
        self.use_amp = use_amp
        self.checkpoint_path = (
            f"{medsymm_root}/models_extracted/models/SymmetricalFlowMatchingClass/"
            "RGB_28/FM_pneumoniamnist_beta4.0_rgb.pt"
        )
        if quick:
            self.budgets, self.seeds, self.epochs, self.syn_per_class = [500], [0], 5, 200
        else:
            self.budgets, self.seeds, self.epochs, self.syn_per_class = [500, 4708], [0, 1, 2], 15, 1000
        # explicit overrides win
        if budgets is not None: self.budgets = budgets
        if seeds is not None: self.seeds = seeds
        if epochs is not None: self.epochs = epochs
        if syn_per_class is not None: self.syn_per_class = syn_per_class
        self.gen_beta = 4.0        # paper default for PneumoniaMNIST MSF
        self.gen_step_size = 0.04  # euler, ~25 steps
        self.gen_solver = "euler"
        self.gen_chunk = 200       # images per class per subprocess call (reduce if OOM)
        self.c1_auc, self.c1_acc = 0.952, 0.880  # published MSF (28px) test reference

        # ---- training knobs (were hardcoded; needed for arch/resolution sweeps later)
        self.arch = "resnet18"
        self.pretrained = True
        self.batch_size = 64
        self.lr = 1e-4
        self.lr_finetune = 1e-5    # S1's fine-tune stage (was a literal 1e-5)
        self.n_classes = 2

        # ---- filtering (see filtering.py). Defaults are the scientifically defensible
        # ones: the filter may only use labels the hypothetical institution owns, so no
        # budget-N information leaks into a budget-500 arm.
        self.filter_mode = "keep_confident"     # none|keep_confident|keep_uncertain|random_match
        self.filter_scorer = "local"            # none|local|full  (local = the arm's own model)
        self.filter_scorer_budget = None        # explicit override for ablations
        self.filter_scorer_seed = None
        self.conf_thresh = 0.60
        self.filter_require_correct = True
        self.mem_reference = "local"            # none|local|full
        self.mem_mode = "quantile"              # quantile|absolute
        self.mem_quantile = 0.015
        self.mem_thresh = None
        self.embed_id = "imagenet_resnet18_224"  # memorisation encoder; independent of arch
        self.filter_random_seed = 12345
        self.s1_pretrain_filter = "mem_only"    # S1 pretrains on the scorer-free pool so it
                                                # stays shared across budgets (see protocol)
        self.legacy_filter = False              # reproduce the old (leaky) semantics for audit
        self.fingerprint_budget = None          # None => max(budgets)

        # ---- run bookkeeping
        self.resume = True
        self.run_tag = ""

        # Any remaining keyword sets an attribute directly, so every knob above is
        # reachable as Config(..., filter_mode="keep_uncertain", gen_beta=1.0).
        for key, value in overrides.items():
            if not hasattr(self, key):
                raise TypeError(f"Config got an unexpected keyword {key!r}")
            setattr(self, key, value)

        # Applied AFTER overrides so `Config(legacy_filter=True)` actually takes effect.
        if self.legacy_filter:
            self.filter_scorer = "full"
            self.filter_scorer_budget = max(self.budgets)
            self.mem_reference = "full"

    @property
    def run_dir(self) -> Path:
        """Single root for every artefact. Kept equal to save_dir for PneumoniaMNIST so
        the existing Drive layout (and its cached synthetic images) stays valid."""
        return self.save_dir


class PathDataset(Dataset):
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = list(paths), list(labels), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert("L")), self.labels[i]


class IntLabel(Dataset):
    """Normalise labels to plain ints so a real Subset and a PathDataset concat
    without a collate-time shape clash ((1,) array vs scalar)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        x, y = self.ds[i]
        return x, int(np.asarray(y).reshape(-1)[0])


class Experiment:
    def __init__(self, cfg=None, **kw):
        self.cfg = cfg or Config(**kw)
        assert torch.cuda.is_available(), \
            "Enable a GPU runtime: Runtime > Change runtime type > T4 GPU"
        self.device = torch.device("cuda")
        c = self.cfg
        c.save_dir.mkdir(parents=True, exist_ok=True)
        self.synthetic_dir = c.run_dir / "synthetic_28"
        self.filtered_dir = c.run_dir / "synthetic_28_filtered"
        self.results_path = c.run_dir / "results.csv"
        self.cache_dir = c.run_dir / "cache"        # embeddings + scorer probabilities
        self.models_dir = c.run_dir / "models"      # persisted B0 / S1-pretrain weights
        self.filters_dir = c.run_dir / "filters"    # one subdir per filter_key
        self.scratch = Path(c.scratch_dir)
        self.fig_dir = Path(c.fig_dir) if c.fig_dir else None
        for d in (self.synthetic_dir, self.filtered_dir, self.cache_dir, self.models_dir,
                  self.filters_dir, self.scratch, self.fig_dir):
            if d is not None:
                d.mkdir(parents=True, exist_ok=True)

        self.train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(c.image_size, scale=(0.8, 1.0)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.eval_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.embed_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.results = []
        self.baseline_models = {}
        self.synthetic_meta = None
        self.filtered = None          # scorer-free (mem-only) pool; S1 pretrains on this
        self._filter_cache = {}       # (budget, seed) -> filtered DataFrame
        self._emb_cache = {}
        self._pool_hash = None

        # Resume: the ledger is the source of truth, so a dead Colab session costs at
        # most the cell that was running.
        self.ledger = self._load_ledger()
        print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
        print("quick:", c.quick, "| budgets:", c.budgets, "| seeds:", c.seeds, "| epochs:", c.epochs)
        print(f"filter: mode={c.filter_mode} scorer={c.filter_scorer} mem={c.mem_reference}"
              + ("  [LEGACY - reproduces the old leaky semantics]" if c.legacy_filter else ""))
        if len(self.ledger):
            print(f"ledger: {len(self.ledger)} existing rows at {self.results_path}"
                  + ("  (resume ON)" if c.resume else "  (resume OFF - will re-run)"))

    # ---------------------------------------------------------------- utilities
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def loader(self, ds, batch_size=64, shuffle=False, sampler=None):
        return DataLoader(ds, batch_size=batch_size, shuffle=(shuffle and sampler is None),
                          sampler=sampler, num_workers=2, pin_memory=True)

    def _savefig(self, name):
        # Persist the current figure to fig_dir for headless/batch runs (no-op interactively).
        if self.fig_dir is not None:
            plt.savefig(self.fig_dir / f"{name}.png", dpi=120, bbox_inches="tight")

    # ------------------------------------------------------ ledger / resume (A2)
    LEDGER_KEY = ["dataset", "arch", "arm", "budget", "seed", "filter_key", "run_tag"]

    def _load_ledger(self):
        if self.results_path.exists():
            try:
                return pd.read_csv(self.results_path)
            except Exception as e:      # a truncated write should not kill the session
                print("WARNING: could not read ledger:", e)
        return pd.DataFrame()

    def _flush_ledger(self):
        """Write via scratch + move: Drive's FUSE mount corrupts partial in-place writes."""
        tmp = self.scratch / "results.tmp.csv"
        self.ledger.to_csv(tmp, index=False)
        shutil.move(str(tmp), str(self.results_path))

    def already_done(self, arm, budget, seed, filter_key=None):
        if not self.cfg.resume or not len(self.ledger):
            return False
        key = {"dataset": "pneumoniamnist", "arch": self.cfg.arch, "arm": arm,
               "budget": budget, "seed": seed,
               "filter_key": filter_key if filter_key is not None else "",
               "run_tag": self.cfg.run_tag}
        m = pd.Series(True, index=self.ledger.index)
        for col, val in key.items():
            if col not in self.ledger.columns:
                return False
            m &= self.ledger[col].fillna("").astype(str) == str(val)
        return bool(m.any())

    def _provenance(self, filter_key="", n_syn_used=0):
        c = self.cfg
        return {"dataset": "pneumoniamnist", "arch": c.arch, "filter_key": filter_key,
                "filter_mode": c.filter_mode, "filter_scorer": c.filter_scorer,
                "filter_scorer_budget": c.filter_scorer_budget,
                "n_syn_used": n_syn_used, "pool_hash": self._pool_hash or "",
                "run_tag": c.run_tag, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}

    # ------------------------------------------------------------ selftest (A1)
    @staticmethod
    def _sha1_state_dict(sd):
        h = hashlib.sha1()
        for k in sorted(sd):
            h.update(k.encode())
            buf = io.BytesIO()
            torch.save(sd[k].detach().cpu().contiguous(), buf)
            h.update(buf.getvalue())
        return h.hexdigest()

    def selftest_repro(self, fixtures_path=None, strict=True):
        """Prove the refactor is numerically inert.

        `train_classifier` seeds and then constructs the model, and Conv2d/Linear consume
        the global torch RNG at construction -- so any reordering of module creation
        shifts every result by an amount that looks exactly like a real effect. Compare
        against fixtures recorded from the pre-refactor code by record_fixtures.py.
        """
        path = Path(fixtures_path or (self.cfg.run_dir / "fixtures_prerefactor.json"))
        self.set_seed(0)
        got = {"model_init_sha1_seed0": self._sha1_state_dict(self.build_model().state_dict())}
        for n in (250, 500, 1000):
            idx, _ = self.stratified_subset(n, 0)
            got[f"subset_{n}_seed0_sha1"] = hashlib.sha1(
                ",".join(str(int(i)) for i in idx).encode()).hexdigest()
        flt._selftest()

        if not path.exists():
            print(f"selftest: no fixtures at {path} -- nothing to compare against.")
            print("          run project/record_fixtures.py on the PRE-refactor code first.")
            return got
        want = json.loads(path.read_text())
        bad = [k for k, v in got.items() if k in want and want[k] != v]
        for k in got:
            if k in want:
                print(f"  {'OK  ' if k not in bad else 'FAIL'} {k}")
        if bad and strict:
            raise AssertionError(
                "selftest_repro FAILED for " + ", ".join(bad) +
                " -- the refactor changed initialisation or subsampling. Results from "
                "before and after are NOT comparable.")
        if not bad:
            print("selftest_repro: refactor is numerically inert.")
        return got

    # -------------------------------------------------------------- data (G1)
    def setup_data(self):
        c = self.cfg
        self.train_set = PneumoniaMNIST(split="train", transform=self.train_tf, download=True, size=c.image_size)
        self.val_set = PneumoniaMNIST(split="val", transform=self.eval_tf, download=True, size=c.image_size)
        self.test_set = PneumoniaMNIST(split="test", transform=self.eval_tf, download=True, size=c.image_size)
        assert len(self.train_set) == 4708, len(self.train_set)
        assert len(self.val_set) == 524, len(self.val_set)
        assert len(self.test_set) == 624, len(self.test_set)
        self.train_labels_all = np.array(self.train_set.labels).reshape(-1)
        print("Split sizes OK: train 4708 / val 524 / test 624")

        rows = []
        for name, ds in [("train", self.train_set), ("val", self.val_set), ("test", self.test_set)]:
            y = np.array(ds.labels).reshape(-1)
            n0, n1 = int((y == 0).sum()), int((y == 1).sum())
            rows.append({"split": name, "normal": n0, "pneumonia": n1,
                         "pneumonia_frac": round(n1 / (n0 + n1), 3)})
        return pd.DataFrame(rows)

    # ------------------------------------------------------- model & training
    def build_model(self, num_classes=None, pretrained=None):
        """Build the classifier with the small-image stem adaptation.

        RNG-ORDER CRITICAL. `train_classifier` calls `set_seed(seed)` and then this, and
        every `nn.Conv2d`/`nn.Linear` consumes the global torch RNG at construction. The
        resnet18 branch must keep the exact sequence resnet18 -> conv1 -> maxpool -> fc,
        with nothing RNG-consuming inserted before, between or after it, or every result
        shifts by an amount indistinguishable from a real effect (see selftest_repro).
        """
        num_classes = self.cfg.n_classes if num_classes is None else num_classes
        pretrained = self.cfg.pretrained if pretrained is None else pretrained
        arch = self.cfg.arch
        if arch != "resnet18":
            raise ValueError(f"arch {arch!r} not available yet (Stage B adds the registry)")
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 28px stem
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def build_resnet18(self, num_classes=2, pretrained=True):
        """Back-compat alias; delegates without touching the construction order."""
        return self.build_model(num_classes, pretrained)

    def run_epoch(self, model, loader, criterion, optimizer=None, scaler=None):
        training = optimizer is not None
        model.train(training)
        losses = []
        for images, labels in loader:
            images = images.to(self.device)
            # reshape(-1), not squeeze(): a final batch of size 1 gives shape (1,1), and
            # squeeze() collapses it to 0-dim -> CrossEntropyLoss errors. Never fired at
            # 4708/524/624 but will as soon as a filtered set has size = 1 (mod batch).
            labels = labels.reshape(-1).long().to(self.device)
            with torch.set_grad_enabled(training):
                with torch.autocast("cuda", enabled=self.cfg.use_amp):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    if scaler is not None:
                        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward(); optimizer.step()
            losses.append(loss.item())
        return {"loss": float(np.mean(losses))}

    @torch.no_grad()
    def predict_probs(self, model, loader):
        model.eval()
        ys, ps = [], []
        for images, labels in loader:
            images = images.to(self.device)
            with torch.autocast("cuda", enabled=self.cfg.use_amp):
                logits = model(images)
            ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
            ys.append(labels.reshape(-1).long().numpy())    # see run_epoch: squeeze() breaks n=1
        return np.concatenate(ys), np.concatenate(ps)

    @torch.no_grad()
    def predict_proba(self, model, loader):
        """Full (n, K) probability matrix -- what the confidence filter caches."""
        model.eval()
        ys, ps = [], []
        for images, labels in loader:
            images = images.to(self.device)
            with torch.autocast("cuda", enabled=self.cfg.use_amp):
                logits = model(images)
            ps.append(torch.softmax(logits.float(), 1).cpu().numpy())
            ys.append(labels.reshape(-1).long().numpy())
        return np.concatenate(ys), np.concatenate(ps, axis=0)

    def best_threshold_on_val(self, model):
        y, p = self.predict_probs(model, self.loader(self.val_set))
        ts = np.linspace(0.05, 0.95, 19)
        j = [balanced_accuracy_score(y, (p >= t).astype(int)) for t in ts]
        return float(ts[int(np.argmax(j))])

    def evaluate_on_test(self, model, threshold):
        y, p = self.predict_probs(model, self.loader(self.test_set))
        pred = (p >= threshold).astype(int)
        return {
            "test_auc": roc_auc_score(y, p),
            "test_acc": accuracy_score(y, pred),
            "test_balacc": balanced_accuracy_score(y, pred),
            "test_f1": f1_score(y, pred),
        }

    def class_weights_for(self, subset_labels):
        counts = np.bincount(subset_labels, minlength=2)
        w = counts.sum() / (2.0 * np.maximum(counts, 1))
        return torch.tensor(w, dtype=torch.float32, device=self.device)

    def train_classifier(self, train_ds, train_labels, seed, epochs=None, lr=None,
                         init_state=None, weighted=False, sampler=None, tag=""):
        epochs = epochs or self.cfg.epochs
        lr = self.cfg.lr if lr is None else lr
        self.set_seed(seed)
        model = self.build_model()
        if init_state is not None:
            model.load_state_dict(init_state)
        criterion = nn.CrossEntropyLoss(
            weight=self.class_weights_for(train_labels) if weighted else None)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=self.cfg.use_amp)
        loader = self.loader(train_ds, shuffle=True, sampler=sampler)
        best_auc, best_state = -1.0, None
        for _ in range(epochs):
            self.run_epoch(model, loader, criterion, optimizer, scaler)
            vy, vp = self.predict_probs(model, self.loader(self.val_set))
            val_auc = roc_auc_score(vy, vp)
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        print(f"  [{tag} seed {seed}] best val AUC {best_auc:.4f}")
        return model, best_auc

    def stratified_subset(self, n, seed):
        if n >= len(self.train_set):
            return list(range(len(self.train_set))), self.train_labels_all
        rng = np.random.default_rng(seed)
        idx = []
        for c in (0, 1):
            c_idx = np.where(self.train_labels_all == c)[0]
            take = int(round(n * (len(c_idx) / len(self.train_labels_all))))
            idx.extend(rng.choice(c_idx, size=take, replace=False).tolist())
        idx = sorted(idx)
        return idx, self.train_labels_all[idx]

    # ------------------------------------------------------------- results api
    def add_result(self, arm, budget, seed, metrics, val_auc, filter_key="", n_syn_used=0):
        row = {"arm": arm, "budget": budget, "seed": seed, "val_auc": val_auc, **metrics,
               **self._provenance(filter_key, n_syn_used)}
        self.results.append(row)
        # Append + flush now, so a dead session costs at most the running cell.
        self.ledger = pd.concat([self.ledger, pd.DataFrame([row])], ignore_index=True)
        self._flush_ledger()
        print(f"  -> {arm} n={budget} seed={seed}: test AUC {metrics['test_auc']:.4f} "
              f"acc {metrics['test_acc']:.4f}")

    def run_supervised(self, arm, train_ds, train_labels, budget, seed,
                       filter_key="", n_syn_used=0, **kw):
        model, val_auc = self.train_classifier(train_ds, train_labels, seed,
                                               tag=f"{arm} n={budget}", **kw)
        self.add_result(arm, budget, seed,
                        self.evaluate_on_test(model, self.best_threshold_on_val(model)),
                        val_auc, filter_key=filter_key, n_syn_used=n_syn_used)
        return model

    # ------------------------------------------------- persisted baseline models
    def _baseline_path(self, budget, seed):
        return self.models_dir / f"B0_{self.cfg.arch}_b{budget}_s{seed}.pt"

    def baseline_model(self, budget, seed, train_if_missing=True):
        """B0 for (budget, seed), from memory -> disk -> freshly trained.

        Persisting matters twice over: a session that died after run_baselines used to
        make filter_synthetic() throw KeyError, and filter_scorer='local' needs one of
        these per arm."""
        if (budget, seed) in self.baseline_models:
            return self.baseline_models[(budget, seed)]
        path = self._baseline_path(budget, seed)
        if path.exists():
            model = self.build_model()
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()
            self.baseline_models[(budget, seed)] = model
            return model
        if not train_if_missing:
            return None
        idx, sub_labels = self.stratified_subset(budget, seed)
        model, _ = self.train_classifier(Subset(self.train_set, idx), sub_labels, seed,
                                         weighted=False, tag=f"B0* scorer n={budget}")
        self._persist_baseline(model, budget, seed)
        return model

    def _persist_baseline(self, model, budget, seed):
        torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()},
                   self._baseline_path(budget, seed))
        self.baseline_models[(budget, seed)] = model

    def oversampler(self, sub_labels):
        class_w = 1.0 / np.bincount(sub_labels, minlength=2).clip(min=1)
        return WeightedRandomSampler(class_w[sub_labels], num_samples=len(sub_labels), replacement=True)

    # ---------------------------------------------------------- baselines (3)
    def run_baselines(self):
        for budget in self.cfg.budgets:
            for seed in self.cfg.seeds:
                idx, sub_labels = self.stratified_subset(budget, seed)
                sub = Subset(self.train_set, idx)

                if self.already_done("B0", budget, seed):
                    print(f"  [skip] B0 n={budget} seed={seed} (in ledger)")
                    self.baseline_model(budget, seed)          # ensure weights are loaded
                else:
                    model = self.run_supervised("B0", sub, sub_labels, budget, seed, weighted=False)
                    self._persist_baseline(model, budget, seed)

                for arm, kw in (("B1", dict(weighted=True)),
                                ("B2", dict(weighted=False, sampler=self.oversampler(sub_labels)))):
                    if self.already_done(arm, budget, seed):
                        print(f"  [skip] {arm} n={budget} seed={seed} (in ledger)")
                        continue
                    self.run_supervised(arm, sub, sub_labels, budget, seed, **kw)
        print("\nB0 reproduction target (protocol): ResNet-18 @28px AUC ~= 94.4")

    # -------------------------------------------------------- generation (MSF)
    def download_weights(self):
        root = self.cfg.medsymm_root
        if not os.path.exists(f"{root}/models.zip"):
            subprocess.run(["wget", "-q", "-O", "models.zip",
                            "https://zenodo.org/records/16086025/files/models.zip?download=1"],
                           cwd=root, check=True)
        if not os.path.exists(f"{root}/models_extracted"):
            subprocess.run(["unzip", "-q", "models.zip", "-d", "models_extracted"], cwd=root, check=True)
        assert os.path.exists(self.cfg.checkpoint_path), self.cfg.checkpoint_path
        print("Checkpoint present:", self.cfg.checkpoint_path)

    def generate_synthetic(self, base_seed=1000):
        c = self.cfg
        per_class, out_dir = c.syn_per_class, self.synthetic_dir
        meta_path = out_dir / "metadata.csv"
        if meta_path.exists():
            existing = pd.read_csv(meta_path)
            if (existing["label"] == 0).sum() >= per_class and (existing["label"] == 1).sum() >= per_class:
                print("Synthetic set already present:", len(existing), "images")
                self.synthetic_meta = existing
                return existing
        for name in ("normal", "pneumonia"):
            (out_dir / name).mkdir(parents=True, exist_ok=True)

        rows, chunk_id = [], 0
        for start in range(0, per_class, c.gen_chunk):
            n = min(c.gen_chunk, per_class - start)
            tmp = self.scratch / f"_gen_chunk_{chunk_id}"
            if tmp.exists():
                shutil.rmtree(tmp)
            cmd = [
                "python", "project/generate_pneumoniamnist.py",
                "--checkpoint", c.checkpoint_path,
                "--dataset", "pneumoniamnist", "--n_classes", "2",
                "--num_normal", str(n), "--num_pneumonia", str(n),
                "--seed", str(base_seed + chunk_id),
                "--beta", str(c.gen_beta), "--image_size", str(c.gen_image_size),
                "--rgb_mask", "--solver", "euler", "--step_size", str(c.gen_step_size),
                "--output_dir", str(tmp),
                # Architecture of the published RGB_28 checkpoint (from its state-dict shapes).
                "--model_channels", "64", "--num_res_blocks", "2",
                "--channel_mult", "1", "2", "2", "2",
                "--num_heads", "4", "--num_head_channels", "64",
                "--attention_resolutions", "2",
            ]
            env = dict(os.environ, PYTHONPATH=f"{c.medsymm_root}/src")
            print("chunk", chunk_id, "->", n, "per class")
            res = subprocess.run(cmd, cwd=c.medsymm_root, env=env, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout[-3000:]); print(res.stderr[-3000:])
                err = [l for l in res.stderr.strip().splitlines()
                       if l.strip() and not l.startswith((" ", "\t", "Traceback"))]
                raise RuntimeError("generation failed: " + (err[-1] if err else "see output above"))
            for cls_name, label in [("normal", 0), ("pneumonia", 1)]:
                for png in sorted((tmp / cls_name).glob("*.png")):
                    dst = out_dir / cls_name / f"{cls_name}_{chunk_id:03d}_{png.stem}.png"
                    img = Image.open(png).convert("L")
                    if img.size != (c.image_size, c.image_size):
                        img = img.resize((c.image_size, c.image_size), Image.LANCZOS)
                    img.save(dst)
                    rows.append({"image_path": str(dst), "label": label,
                                 "class_name": cls_name, "gen_seed": base_seed + chunk_id})
            shutil.rmtree(tmp)
            chunk_id += 1

        meta = pd.DataFrame(rows)
        meta.to_csv(meta_path, index=False)
        print("Generated", len(meta), "synthetic images ->", out_dir)
        self.synthetic_meta = meta
        return meta

    def visualize_samples(self, per_class=8):
        fig, axes = plt.subplots(2, per_class, figsize=(2 * per_class, 4))
        for r, cls in enumerate(["normal", "pneumonia"]):
            paths = self.synthetic_meta[self.synthetic_meta.class_name == cls]["image_path"].tolist()[:per_class]
            for a, pth in zip(axes[r], paths):
                a.imshow(Image.open(pth), cmap="gray"); a.set_title(cls, fontsize=8); a.axis("off")
        plt.tight_layout(); self._savefig("synthetic_samples"); plt.show()

    # --------------------------------------------------------- filtering (Sec 7)
    # Expensive GPU work is cached as CONTINUOUS scores (embeddings, probabilities); the
    # keep/drop decision is cheap pure arithmetic in filtering.derive_keep. Mode and
    # threshold ablations therefore cost zero GPU and are exactly reproducible -- fp16
    # autocast on a T4 vs an A100 can otherwise flip a borderline confidence across 0.60.

    def _gen_manifest(self):
        c = self.cfg
        return {"checkpoint": Path(c.checkpoint_path).name, "beta": c.gen_beta,
                "step_size": c.gen_step_size, "solver": c.gen_solver,
                "gen_image_size": c.gen_image_size, "image_size": c.image_size,
                "syn_per_class": c.syn_per_class}

    def pool_hash(self):
        if self._pool_hash is None:
            self._pool_hash = flt.pool_hash(self.synthetic_meta.image_path,
                                            self.synthetic_meta.label, self._gen_manifest())
        return self._pool_hash

    def _encoder(self):
        if "enc" not in self._emb_cache:
            enc = resnet18(weights=ResNet18_Weights.DEFAULT)   # fixed, independent of cfg.arch
            enc.fc = nn.Identity()
            self._emb_cache["enc"] = enc.to(self.device).eval()
        return self._emb_cache["enc"]

    @torch.no_grad()
    def _embed_paths(self, paths):
        enc = self._encoder()
        out = []
        for x, _ in self.loader(PathDataset(paths, [0] * len(paths), self.embed_tf), batch_size=128):
            out.append(nn.functional.normalize(enc(x.to(self.device)), dim=1).cpu())
        return torch.cat(out).numpy()

    def _real_train_paths(self):
        real_dir = self.scratch / "_real_train_png"
        real_dir.mkdir(parents=True, exist_ok=True)
        raw = PneumoniaMNIST(split="train", download=True, size=self.cfg.image_size)
        paths = []
        for i in range(len(raw)):
            p = real_dir / f"r_{i:05d}.png"
            if not p.exists():
                raw[i][0].convert("L").save(p)
            paths.append(str(p))
        return paths

    def _embed_real(self):
        path = self.cache_dir / f"emb_real_{self.cfg.image_size}_{self.cfg.embed_id}.npy"
        if "real" not in self._emb_cache:
            if path.exists():
                self._emb_cache["real"] = np.load(path)
            else:
                emb = self._embed_paths(self._real_train_paths())
                np.save(path, emb)
                self._emb_cache["real"] = emb
        return self._emb_cache["real"]

    def _embed_syn(self):
        path = self.cache_dir / f"emb_syn_{self.pool_hash()}_{self.cfg.embed_id}.npy"
        if "syn" not in self._emb_cache:
            if path.exists():
                self._emb_cache["syn"] = np.load(path)
            else:
                emb = self._embed_paths(self.synthetic_meta.image_path.tolist())
                np.save(path, emb)
                self._emb_cache["syn"] = emb
        return self._emb_cache["syn"]

    def _mem_distances(self, reference_idx=None):
        """Nearest-real-neighbour cosine distance per synthetic image.

        `reference_idx=None` compares against the whole train split; passing a budget's
        own indices keeps full-dataset information out of a low-budget arm (the second
        leak in the original implementation).
        """
        real = self._embed_real()
        if reference_idx is not None:
            real = real[np.asarray(reference_idx)]
        return 1.0 - (self._embed_syn() @ real.T).max(axis=1)

    def _scorer_probs(self, budget, seed):
        """Cached (n, K) probabilities from the resolved scorer model."""
        key = flt.make_key(self._scorer_manifest(budget, seed))
        path = self.cache_dir / f"probs_{key}.npy"
        if path.exists():
            return np.load(path)
        model = self.baseline_model(budget, seed)
        _, probs = self.predict_proba(model, self.loader(
            PathDataset(self.synthetic_meta.image_path,
                        self.synthetic_meta.label.tolist(), self.eval_tf)))
        np.save(path, probs)
        return probs

    def _scorer_manifest(self, budget, seed):
        c = self.cfg
        return flt.scorer_manifest(
            dataset="pneumoniamnist", arch=c.arch, pool=self.pool_hash(),
            scorer=c.filter_scorer, scorer_budget=budget, scorer_seed=seed,
            pretrained=c.pretrained, epochs=c.epochs, lr=c.lr,
            batch_size=c.batch_size, image_size=c.image_size, use_amp=c.use_amp)

    def _filter_manifest(self, budget, seed):
        c = self.cfg
        resolved = flt.resolve_scorer(c.filter_scorer, budget=budget, seed=seed,
                                      train_size=len(self.train_set), seeds0=c.seeds[0],
                                      scorer_budget=c.filter_scorer_budget,
                                      scorer_seed=c.filter_scorer_seed)
        sb, ss = resolved if resolved else (None, None)
        mem_b, mem_s = ((budget, seed) if c.mem_reference == "local" else (None, None))
        return flt.filter_manifest(
            self._scorer_manifest(sb, ss) if resolved else
            flt.scorer_manifest(dataset="pneumoniamnist", arch=c.arch, pool=self.pool_hash(),
                                scorer="none", scorer_budget=None, scorer_seed=None,
                                pretrained=c.pretrained, epochs=c.epochs, lr=c.lr,
                                batch_size=c.batch_size, image_size=c.image_size,
                                use_amp=c.use_amp),
            mode=c.filter_mode, conf_thresh=c.conf_thresh,
            require_correct=c.filter_require_correct, mem_reference=c.mem_reference,
            mem_mode=c.mem_mode, mem_quantile=c.mem_quantile, mem_thresh=c.mem_thresh,
            embed_id=c.embed_id, random_seed=c.filter_random_seed,
            mem_budget=mem_b, mem_seed=mem_s), resolved

    def filtered_for(self, budget, seed):
        """The synthetic subset this arm may train on. Deterministic and cached."""
        if (budget, seed) in self._filter_cache:
            return self._filter_cache[(budget, seed)]
        c = self.cfg
        manifest, resolved = self._filter_manifest(budget, seed)
        key = flt.make_key(manifest)
        out_dir = self.filters_dir / key
        meta_path = out_dir / "metadata.csv"
        if meta_path.exists():
            df = pd.read_csv(meta_path)
            self._filter_cache[(budget, seed)] = (df, key)
            return df, key

        ref = None
        if c.mem_reference == "local":
            ref, _ = self.stratified_subset(budget, seed)
        nn_dist = None if c.mem_reference == "none" else self._mem_distances(ref)
        probs = self._scorer_probs(*resolved) if resolved else None
        labels = np.array(self.synthetic_meta.label)

        keep, keep_mem, keep_score = flt.derive_keep(
            nn_dist, probs, labels, mode=c.filter_mode, conf_thresh=c.conf_thresh,
            require_correct=c.filter_require_correct, mem_mode=c.mem_mode,
            mem_quantile=c.mem_quantile, mem_thresh=c.mem_thresh,
            random_seed=c.filter_random_seed)

        df = self.synthetic_meta[keep].reset_index(drop=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(meta_path, index=False)
        stats_ = flt.summarise_keep(keep, keep_mem, keep_score, labels, c.n_classes)
        (out_dir / "manifest.json").write_text(json.dumps({**manifest, **stats_,
                                                           "budget": budget, "seed": seed}, indent=2))
        self._filter_cache[(budget, seed)] = (df, key)
        return df, key

    def filter_synthetic(self, plot=True, **overrides):
        """Populate the filter caches for the current sweep; return a summary table.

        Also sets `self.filtered` to the scorer-free (memorisation-only, full-reference)
        pool. That pool is deliberately budget-independent, and is what S1 pretrains on,
        so one pretrain can be shared across budgets instead of one per (budget, seed).
        """
        for k, v in overrides.items():
            setattr(self.cfg, k, v)
        c = self.cfg

        # Scorer-free pool for S1 / D1 -- no labels beyond the requested class are used.
        nn_dist_full = self._mem_distances(None)
        labels = np.array(self.synthetic_meta.label)
        keep_mem_full, _, _ = flt.derive_keep(nn_dist_full, None, labels, mode="none",
                                              mem_mode=c.mem_mode, mem_quantile=c.mem_quantile,
                                              mem_thresh=c.mem_thresh)
        self.filtered = self.synthetic_meta[keep_mem_full].reset_index(drop=True)
        self.filtered.to_csv(self.filtered_dir / "metadata.csv", index=False)

        if plot:
            cut = np.quantile(nn_dist_full, c.mem_quantile)
            plt.hist(nn_dist_full, bins=40); plt.axvline(cut, color="r", ls="--")
            plt.xlabel("nearest-real distance"); plt.ylabel("count")
            plt.title("Memorisation screen (full-train reference)")
            self._savefig("memorisation_screen"); plt.show()
        print(f"Memorisation discard (full ref): {int((~keep_mem_full).sum())}/{len(keep_mem_full)} "
              f"({(~keep_mem_full).mean()*100:.1f}%)  -> S1/D1 pool = {len(self.filtered)}")

        rows = []
        for budget in c.budgets:
            for seed in c.seeds:
                df, key = self.filtered_for(budget, seed)
                rows.append({"budget": budget, "seed": seed, "mode": c.filter_mode,
                             "scorer": c.filter_scorer, "mem_ref": c.mem_reference,
                             "n_pool": len(self.synthetic_meta), "n_kept": len(df),
                             "kept_class0": int((df.label == 0).sum()),
                             "kept_class1": int((df.label == 1).sum()),
                             "filter_key": key})
        summary = pd.DataFrame(rows)
        print(summary.to_string(index=False))
        return summary

    # --------------------------------------------------- synthetic arms (S1-3)
    def _synth_ds(self, df):
        return PathDataset(df.image_path.tolist(), df.label.tolist(), self.train_tf)

    def run_synthetic(self):
        minority = int(np.argmin(np.bincount(self.train_labels_all, minlength=self.cfg.n_classes)))

        for seed in self.cfg.seeds:
            # S1 pretrains on the scorer-free, budget-independent pool (protocol choice:
            # per-budget filtering here would multiply pretraining cost by len(budgets)).
            pre_state = None

            def pretrain_state():
                nonlocal pre_state
                if pre_state is None:
                    path = self.models_dir / f"S1pre_{self.pool_hash()}_{self.cfg.arch}_s{seed}.pt"
                    if path.exists():
                        pre_state = torch.load(path, map_location="cpu")
                    else:
                        m, _ = self.train_classifier(self._synth_ds(self.filtered),
                                                     np.array(self.filtered.label), seed,
                                                     weighted=False, tag="S1-pretrain")
                        pre_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                        torch.save(pre_state, path)
                return pre_state

            for budget in self.cfg.budgets:
                idx, sub_labels = self.stratified_subset(budget, seed)
                real_sub = Subset(self.train_set, idx)
                syn_df, fkey = self.filtered_for(budget, seed)
                syn_labels = np.array(syn_df.label)

                if self.already_done("S1", budget, seed, fkey):
                    print(f"  [skip] S1 n={budget} seed={seed}")
                else:
                    self.run_supervised("S1", real_sub, sub_labels, budget, seed,
                                        lr=self.cfg.lr_finetune, init_state=pretrain_state(),
                                        filter_key=fkey, n_syn_used=len(self.filtered))

                if self.already_done("S2", budget, seed, fkey):
                    print(f"  [skip] S2 n={budget} seed={seed}")
                else:
                    mix_ds = ConcatDataset([IntLabel(real_sub), self._synth_ds(syn_df)])
                    self.run_supervised("S2", mix_ds, np.concatenate([sub_labels, syn_labels]),
                                        budget, seed, weighted=False,
                                        filter_key=fkey, n_syn_used=len(syn_df))

                if self.already_done("S3", budget, seed, fkey):
                    print(f"  [skip] S3 n={budget} seed={seed}")
                else:
                    syn_minority = syn_df[syn_df.label == minority].reset_index(drop=True)
                    n0, n1 = int((sub_labels == 0).sum()), int((sub_labels == 1).sum())
                    add_df = syn_minority.iloc[:min(abs(n1 - n0), len(syn_minority))]
                    s3_ds = ConcatDataset([IntLabel(real_sub), self._synth_ds(add_df)])
                    self.run_supervised("S3", s3_ds,
                                        np.concatenate([sub_labels, np.array(add_df.label)]),
                                        budget, seed, weighted=False,
                                        filter_key=fkey, n_syn_used=len(add_df))

    # ---------------------------------------------- distillation diagnostics
    def run_diagnostic_d1(self):
        """D1: train on synthetic ONLY, test on real. If D1 recovers baseline/C1-level
        AUC, the synthetic set alone carries the decision function -> distillation."""
        self.d1_models = {}
        for seed in self.cfg.seeds:
            model, val_auc = self.train_classifier(
                self._synth_ds(self.filtered), np.array(self.filtered.label), seed,
                weighted=False, tag="D1")
            self.d1_models[seed] = model
            if self.already_done("D1", 0, seed):
                print(f"  [skip-record] D1 seed={seed} already in ledger")
                continue
            self.add_result("D1", 0, seed,
                            self.evaluate_on_test(model, self.best_threshold_on_val(model)),
                            val_auc, n_syn_used=len(self.filtered))
        aucs = [r["test_auc"] for r in self.results if r["arm"] == "D1"]
        if not aucs:
            aucs = self.ledger.query("arm == 'D1'")["test_auc"].tolist()
        print(f"\nD1 (synthetic-only) mean test AUC: {np.mean(aucs):.4f}  "
              f"[C1 MSF ref {self.cfg.c1_auc}; real baselines ~0.94]")
        print("D1 near or above C1 means the synthetic set alone carries the class structure. "
              "Whether that is DISTILLATION of MSF's decision function or plain coverage of "
              "the data manifold is decided by the fingerprint in distillation_agreement(), "
              "not by this number.")

    def msf_test_predictions(self):
        """MSF's own reverse-flow classification of the real test split (subprocess), cached to Drive."""
        # Versioned filename: the old cache held only hard predictions, so a stale file
        # would be reused forever and measure_c1() could never compute an AUC.
        c = self.cfg
        out_csv = (c.run_dir /
                   f"msf_preds_pneumoniamnist_test_{c.gen_image_size}_beta{c.gen_beta}"
                   f"_{c.gen_solver}{c.gen_step_size}.csv")
        if out_csv.exists():
            return pd.read_csv(out_csv)
        # classify_medmnist.py also writes soft distance-to-class scores (msf_negdist_*),
        # which is what makes a measured C1 AUC possible.
        cmd = [
            "python", "project/classify_medmnist.py",
            "--checkpoint", c.checkpoint_path, "--output_csv", str(out_csv),
            "--dataset", "pneumoniamnist", "--n_classes", "2",
            "--image_size", str(c.gen_image_size), "--beta", str(c.gen_beta),
            "--rgb_mask", "--solver", c.gen_solver, "--step_size", str(c.gen_step_size),
        ]
        env = dict(os.environ, PYTHONPATH=f"{c.medsymm_root}/src")
        res = subprocess.run(cmd, cwd=c.medsymm_root, env=env, capture_output=True, text=True)
        print(res.stdout.strip()[-800:])
        if res.returncode != 0:
            # Fall back to the legacy script: it yields no soft scores (so no measured C1
            # AUC) but still produces the hard predictions the fingerprint needs.
            print("classify_medmnist.py failed; falling back to classify_pneumoniamnist.py")
            print(res.stderr[-1500:])
            legacy = [
                "python", "project/classify_pneumoniamnist.py",
                "--checkpoint", c.checkpoint_path, "--output_csv", str(out_csv),
                "--dataset", "pneumoniamnist", "--n_classes", "2",
                "--image_size", str(c.gen_image_size), "--beta", str(c.gen_beta),
                "--rgb_mask", "--solver", c.gen_solver, "--step_size", str(c.gen_step_size),
                "--model_channels", "64", "--num_res_blocks", "2",
                "--channel_mult", "1", "2", "2", "2",
                "--num_heads", "4", "--num_head_channels", "64", "--attention_resolutions", "2",
            ]
            res = subprocess.run(legacy, cwd=c.medsymm_root, env=env,
                                 capture_output=True, text=True)
            print(res.stdout.strip()[-800:])
            if res.returncode != 0:
                print(res.stderr[-2500:])
                raise RuntimeError("MSF classification failed")
        return pd.read_csv(out_csv)

    def distillation_agreement(self, budget=None, seed=None):
        """Fingerprint: does a synthetic-trained ResNet (D1) copy MSF's predictions --
        especially MSF's *errors* -- more than a real-trained ResNet (B0)? Copying errors
        needs copying the function, which mere data-manifold coverage cannot explain."""
        msf = self.msf_test_predictions()
        ty, msf_pred = msf["true"].values, msf["msf_pred"].values

        budget = budget if budget is not None else (
            self.cfg.fingerprint_budget or max(self.cfg.budgets))
        seeds = self.cfg.seeds if seed is None else [seed]
        err = msf_pred != ty                       # images MSF gets wrong
        n_err = int(err.sum())

        def scores(model):
            """Agreement, plus a CALIBRATION-MATCHED variant.

            Raw agreement is confounded: D1 trains on a near-balanced synthetic set while
            B0 trains on 74%-pneumonia real data, so the two models sit at different
            operating points and their agreement differs for reasons that have nothing to
            do with copying a decision function. Matching each model's positive rate to
            MSF's removes that prior difference before comparing.
            """
            y, p = self.predict_probs(model, self.loader(self.test_set))
            assert np.array_equal(y, ty), "test order mismatch between MSF CSV and loader"
            raw = (p >= 0.5).astype(int)
            thr = np.quantile(p, 1.0 - (msf_pred == 1).mean())   # match MSF's positive rate
            matched = (p >= thr).astype(int)
            out = {}
            for name, pred in (("", raw), ("_matched", matched)):
                out["agree_with_MSF" + name] = float((pred == msf_pred).mean())
                out["agree_on_MSF_errors" + name] = (
                    float((pred[err] == msf_pred[err]).mean()) if n_err else np.nan)
            return out

        rows = []
        for s in seeds:
            real_model = self.baseline_model(budget, s, train_if_missing=False)
            syn_model = self.d1_models.get(s)
            if real_model is None or syn_model is None:
                continue
            rows.append({"seed": s, "model": "real-trained (B0)", **scores(real_model)})
            rows.append({"seed": s, "model": "synthetic-trained (D1)", **scores(syn_model)})
        if not rows:
            raise RuntimeError("no models available; run run_baselines() and run_diagnostic_d1()")

        per_seed = pd.DataFrame(rows)
        agg = (per_seed.drop(columns=["seed"]).groupby("model").agg(["mean", "std"]).round(3))
        print(f"MSF test accuracy: {(msf_pred == ty).mean():.3f}  (errors: {n_err}/{len(ty)}), "
              f"B0 budget={budget}, seeds={list(per_seed.seed.unique())}")
        print("Distillation fingerprint = HIGHER agree_on_MSF_errors for the synthetic-trained "
              "model. Read the *_matched columns: they remove the class-prior confound.")
        self._fingerprint_per_seed = per_seed
        return agg

    # ----------------------------------------------------- reference & summary
    def measure_c1(self):
        """Measure MSF's own classification on OUR test split instead of quoting the paper.

        The published 0.952 was produced on the authors' setup; our reproduction of the
        ResNet-18 baseline already lands well above the paper's (0.970 vs 0.944), so the
        published constant is not a like-for-like reference. `classify_medmnist.py` also
        writes soft distance-to-class scores, which is what makes an AUC possible here.
        """
        msf = self.msf_test_predictions()
        y, pred = msf["true"].values, msf["msf_pred"].values
        acc = float((pred == y).mean())
        auc = np.nan
        if "msf_negdist_1" in msf.columns:      # soft scores available -> real AUC
            auc = float(roc_auc_score(y, msf["msf_negdist_1"].values))
        elif "msf_negdist_0" in msf.columns:
            auc = float(roc_auc_score(y, -msf["msf_negdist_0"].values))
        print(f"C1 measured on our test split: ACC {acc:.4f}"
              + (f", AUC {auc:.4f}" if not np.isnan(auc) else
                 "  (no soft scores in the cached CSV -- regenerate with classify_medmnist.py for AUC)")
              + f"   [published: ACC {self.cfg.c1_acc}, AUC {self.cfg.c1_auc}]")
        if not np.isnan(auc):
            self.cfg.c1_auc_measured = auc
        self.cfg.c1_acc_measured = acc
        return {"c1_acc_measured": acc, "c1_auc_measured": auc,
                "c1_acc_published": self.cfg.c1_acc, "c1_auc_published": self.cfg.c1_auc}

    def record_c1(self, use_measured=False):
        auc = getattr(self.cfg, "c1_auc_measured", None) if use_measured else None
        acc = getattr(self.cfg, "c1_acc_measured", None) if use_measured else None
        auc = self.cfg.c1_auc if auc is None else auc
        acc = self.cfg.c1_acc if acc is None else acc
        for budget in self.cfg.budgets:
            if self.already_done("C1", budget, -1):
                continue
            self.add_result("C1", budget, -1,
                            {"test_auc": auc, "test_acc": acc,
                             "test_balacc": np.nan, "test_f1": np.nan}, np.nan)
        print(f"C1 MSF reference recorded: AUC {auc}, ACC {acc}"
              + ("  (measured)" if use_measured else "  (published)"))

    @staticmethod
    def _ci95(x):
        x = x.dropna().values
        if len(x) < 2:
            return np.nan
        return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

    def current_selection(self):
        """Rows in the ledger belonging to the current configuration."""
        df = self.ledger
        if not len(df):
            return df
        for col, val in (("dataset", "pneumoniamnist"), ("arch", self.cfg.arch),
                         ("run_tag", self.cfg.run_tag)):
            if col in df.columns:
                df = df[df[col].fillna("").astype(str) == str(val)]
        return df

    def summarize(self, select=True):
        # Read the LEDGER, never overwrite it: it is now a growing record and rewriting
        # it from the in-memory list would delete every previous run.
        res = self.current_selection() if select else self.ledger
        if not len(res):
            res = pd.DataFrame(self.results)
        summary = (res.groupby(["arm", "budget"])
                     .agg(auc_mean=("test_auc", "mean"),
                          auc_ci=("test_auc", self._ci95),
                          acc_mean=("test_acc", "mean"),
                          n_seeds=("test_auc", "count"))
                     .reset_index())

        BASE, SYN = ["B0", "B1", "B2"], ["S1", "S2", "S3"]
        # dropna=False keeps all-NaN CI columns (e.g. single-seed quick runs), so
        # `means` and `cis` stay column-aligned.
        means = summary.pivot_table(index="budget", columns="arm", values="auc_mean", dropna=False)
        cis = summary.pivot_table(index="budget", columns="arm", values="auc_ci", dropna=False)
        for col in BASE + SYN + ["C1"]:
            if col not in means.columns:
                means[col] = np.nan
            if col not in cis.columns:
                cis[col] = np.nan

        rows = []
        for b in means.index:
            if means.loc[b, BASE].isna().all():   # e.g. D1's sentinel budget 0
                continue
            best_base = means.loc[b, BASE].max()
            best_base_arm = means.loc[b, BASE].idxmax()
            for s in SYN:
                if np.isnan(means.loc[b, s]):
                    continue
                sig = ""
                if not np.isnan(cis.loc[b, s]) and not np.isnan(cis.loc[b, best_base_arm]):
                    lo_s = means.loc[b, s] - cis.loc[b, s]
                    hi_base = best_base + cis.loc[b, best_base_arm]
                    sig = "yes" if lo_s > hi_base else "no"
                rows.append({"budget": b, "arm": s, "auc": round(means.loc[b, s], 4),
                             "best_baseline": f"{best_base_arm} {best_base:.4f}",
                             "delta_vs_base": round(means.loc[b, s] - best_base, 4),
                             "CIs_separate": sig,
                             "beats_C1": "yes" if means.loc[b, s] > self.cfg.c1_auc else "no"})
        comparison = pd.DataFrame(rows)
        summary.to_csv(self.cfg.run_dir / "summary.csv", index=False)
        comparison.to_csv(self.cfg.run_dir / "comparison.csv", index=False)
        print(f"Ledger: {len(self.ledger)} rows ({len(res)} in this selection) -> {self.results_path}")
        return summary, comparison

    def paired_tests(self, alpha=0.05):
        """Paired seed-wise test vs the strongest baseline + BH correction.

        Runs on the current selection of the ledger, so rows from a different filter
        configuration cannot be silently averaged into the same (arm, budget, seed) cell.
        """
        res = self.current_selection()
        if not len(res):
            res = pd.DataFrame(self.results)
        return paired_tests_from_csv(res, alpha=alpha)

    def plot(self, summary):
        fig, ax = plt.subplots(figsize=(8, 5))
        palette = {"B0": "C0", "B1": "C1", "B2": "C2", "S1": "C3", "S2": "C4", "S3": "C5"}
        for arm in ["B0", "B1", "B2", "S1", "S2", "S3"]:
            d = summary[summary.arm == arm].sort_values("budget")
            if len(d):
                ls = "--" if arm.startswith("B") else "-"
                ax.errorbar(d.budget, d.auc_mean, yerr=d.auc_ci.fillna(0), marker="o",
                            capsize=3, ls=ls, color=palette[arm], label=arm)
        ax.axhline(self.cfg.c1_auc, color="gray", ls="--", label=f"C1 MSF ({self.cfg.c1_auc})")
        ax.axhline(0.944, color="black", ls=":", label="B0 target (0.944)")
        d1 = summary[summary.arm == "D1"]
        if len(d1):
            ax.axhline(d1.auc_mean.mean(), color="C6", ls="-.",
                       label=f"D1 synth-only ({d1.auc_mean.mean():.3f})")
        ax.set_xlabel("real training images"); ax.set_ylabel("test AUC")
        ax.set_title("Gain vs data budget (baselines dashed, synthetic solid)")
        ax.legend(ncol=2, fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout(); self._savefig("gain_vs_budget"); plt.show()
