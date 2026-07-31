"""PneumoniaMNIST + MedSymmFlow synthetic-augmentation experiment.

Imported by notebooks/pneumoniamnist_augmentation.ipynb so that logic fixes land
via `git pull` instead of a notebook re-upload. The notebook holds only config,
narrative, and calls; all behaviour lives here.

Protocol: pneumoniamnist_augmentation_protocol v2.0.
Arms: B0/B1/B2 (baselines), S1/S2/S3 (synthetic), C1 (MSF reference).
"""
import os
import shutil
import subprocess
import random
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

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class Config:
    """Experiment knobs. `quick=True` is a fast smoke test; False is the real run."""

    def __init__(self, quick=True, save_dir="/content/drive/MyDrive/MedSymmFlow_Project",
                 medsymm_root="/content/MedSymmFlow", image_size=28, gen_image_size=32,
                 use_amp=True, budgets=None, seeds=None, epochs=None, syn_per_class=None,
                 scratch_dir="/content", fig_dir=None):
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
        self.gen_chunk = 200       # images per class per subprocess call (reduce if OOM)
        self.c1_auc, self.c1_acc = 0.952, 0.880  # published MSF (28px) test reference


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
        self.synthetic_dir = c.save_dir / "synthetic_28"
        self.filtered_dir = c.save_dir / "synthetic_28_filtered"
        self.results_path = c.save_dir / "results.csv"
        self.scratch = Path(c.scratch_dir)
        self.fig_dir = Path(c.fig_dir) if c.fig_dir else None
        for d in (self.synthetic_dir, self.filtered_dir, self.scratch, self.fig_dir):
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
        self.filtered = None
        print("PyTorch:", torch.__version__, "| GPU:", torch.cuda.get_device_name(0))
        print("quick:", c.quick, "| budgets:", c.budgets, "| seeds:", c.seeds, "| epochs:", c.epochs)

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
    def build_resnet18(self, num_classes=2, pretrained=True):
        model = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 28px stem
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.to(self.device)

    def run_epoch(self, model, loader, criterion, optimizer=None, scaler=None):
        training = optimizer is not None
        model.train(training)
        losses = []
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.squeeze().long().to(self.device)
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
            ys.append(labels.squeeze().long().numpy())
        return np.concatenate(ys), np.concatenate(ps)

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

    def train_classifier(self, train_ds, train_labels, seed, epochs=None, lr=1e-4,
                         init_state=None, weighted=False, sampler=None, tag=""):
        epochs = epochs or self.cfg.epochs
        self.set_seed(seed)
        model = self.build_resnet18()
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
    def add_result(self, arm, budget, seed, metrics, val_auc):
        self.results.append({"arm": arm, "budget": budget, "seed": seed,
                             "val_auc": val_auc, **metrics})
        print(f"  -> {arm} n={budget} seed={seed}: test AUC {metrics['test_auc']:.4f} "
              f"acc {metrics['test_acc']:.4f}")

    def run_supervised(self, arm, train_ds, train_labels, budget, seed, **kw):
        model, val_auc = self.train_classifier(train_ds, train_labels, seed,
                                               tag=f"{arm} n={budget}", **kw)
        self.add_result(arm, budget, seed, self.evaluate_on_test(model, self.best_threshold_on_val(model)), val_auc)
        return model

    def oversampler(self, sub_labels):
        class_w = 1.0 / np.bincount(sub_labels, minlength=2).clip(min=1)
        return WeightedRandomSampler(class_w[sub_labels], num_samples=len(sub_labels), replacement=True)

    # ---------------------------------------------------------- baselines (3)
    def run_baselines(self):
        for budget in self.cfg.budgets:
            for seed in self.cfg.seeds:
                idx, sub_labels = self.stratified_subset(budget, seed)
                sub = Subset(self.train_set, idx)
                self.baseline_models[(budget, seed)] = self.run_supervised(
                    "B0", sub, sub_labels, budget, seed, weighted=False)
                self.run_supervised("B1", sub, sub_labels, budget, seed, weighted=True)
                self.run_supervised("B2", sub, sub_labels, budget, seed, weighted=False,
                                    sampler=self.oversampler(sub_labels))
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
    def filter_synthetic(self, mem_quantile=0.015, conf_thresh=0.60):
        enc = resnet18(weights=ResNet18_Weights.DEFAULT)
        enc.fc = nn.Identity()
        enc = enc.to(self.device).eval()

        @torch.no_grad()
        def embed(loader):
            out = []
            for x, _ in loader:
                out.append(nn.functional.normalize(enc(x.to(self.device)), dim=1).cpu())
            return torch.cat(out)

        real_dir = self.scratch / "_real_train_png"; real_dir.mkdir(parents=True, exist_ok=True)
        raw_train = PneumoniaMNIST(split="train", download=True, size=self.cfg.image_size)
        real_paths = []
        for i in range(len(raw_train)):
            p = real_dir / f"r_{i:05d}.png"
            if not p.exists():
                raw_train[i][0].convert("L").save(p)
            real_paths.append(str(p))

        real_emb = embed(self.loader(PathDataset(real_paths, [0] * len(real_paths), self.embed_tf), batch_size=128))
        syn_emb = embed(self.loader(PathDataset(self.synthetic_meta.image_path,
                                                self.synthetic_meta.label.tolist(), self.embed_tf), batch_size=128))
        nn_dist = 1.0 - (syn_emb @ real_emb.T).max(dim=1).values.numpy()
        cut = np.quantile(nn_dist, mem_quantile)
        keep_mem = nn_dist > cut
        plt.hist(nn_dist, bins=40); plt.axvline(cut, color="r", ls="--")
        plt.xlabel("nearest-real distance"); plt.ylabel("count"); plt.title("Memorisation screen")
        self._savefig("memorisation_screen"); plt.show()
        print(f"Memorisation discard: {int((~keep_mem).sum())}/{len(keep_mem)} ({(~keep_mem).mean()*100:.1f}%)")

        scorer = self.baseline_models[(max(self.cfg.budgets), self.cfg.seeds[0])]
        sy, sp = self.predict_probs(scorer, self.loader(
            PathDataset(self.synthetic_meta.image_path, self.synthetic_meta.label.tolist(), self.eval_tf)))
        labels = np.array(self.synthetic_meta.label)
        pred_label = (sp >= 0.5).astype(int)
        conf = np.where(labels == 1, sp, 1 - sp)
        keep_conf = (pred_label == labels) & (conf >= conf_thresh)

        keep = keep_mem & keep_conf
        self.filtered = self.synthetic_meta[keep].reset_index(drop=True)
        self.filtered.to_csv(self.filtered_dir / "metadata.csv", index=False)
        print(f"Confidence filter keeps {int(keep_conf.sum())}/{len(keep_conf)}")
        print(f"Combined kept: {len(self.filtered)}/{len(self.synthetic_meta)} "
              f"(normal {int((self.filtered.label == 0).sum())}, "
              f"pneumonia {int((self.filtered.label == 1).sum())})")
        return self.filtered

    # --------------------------------------------------- synthetic arms (S1-3)
    def _synth_ds(self, df):
        return PathDataset(df.image_path.tolist(), df.label.tolist(), self.train_tf)

    def run_synthetic(self):
        syn_all_labels = np.array(self.filtered.label)
        minority = int(np.argmin(np.bincount(self.train_labels_all, minlength=2)))
        syn_minority = self.filtered[self.filtered.label == minority].reset_index(drop=True)

        for seed in self.cfg.seeds:
            pre_model, _ = self.train_classifier(self._synth_ds(self.filtered), syn_all_labels, seed,
                                                 lr=1e-4, weighted=False, tag="S1-pretrain")
            pre_state = {k: v.detach().cpu().clone() for k, v in pre_model.state_dict().items()}

            for budget in self.cfg.budgets:
                idx, sub_labels = self.stratified_subset(budget, seed)
                real_sub = Subset(self.train_set, idx)

                self.run_supervised("S1", real_sub, sub_labels, budget, seed,
                                    lr=1e-5, init_state=pre_state)

                mix_ds = ConcatDataset([IntLabel(real_sub), self._synth_ds(self.filtered)])
                self.run_supervised("S2", mix_ds, np.concatenate([sub_labels, syn_all_labels]),
                                    budget, seed, weighted=False)

                n0, n1 = int((sub_labels == 0).sum()), int((sub_labels == 1).sum())
                add_df = syn_minority.iloc[:min(abs(n1 - n0), len(syn_minority))]
                s3_ds = ConcatDataset([IntLabel(real_sub), self._synth_ds(add_df)])
                self.run_supervised("S3", s3_ds, np.concatenate([sub_labels, np.array(add_df.label)]),
                                    budget, seed, weighted=False)

    # ---------------------------------------------- distillation diagnostics
    def run_diagnostic_d1(self):
        """D1: train on synthetic ONLY, test on real. If D1 recovers baseline/C1-level
        AUC, the synthetic set alone carries the decision function -> distillation."""
        self.d1_models = {}
        for seed in self.cfg.seeds:
            model, val_auc = self.train_classifier(
                self._synth_ds(self.filtered), np.array(self.filtered.label), seed,
                weighted=False, tag="D1")
            self.add_result("D1", 0, seed, self.evaluate_on_test(model, self.best_threshold_on_val(model)), val_auc)
            self.d1_models[seed] = model
        aucs = [r["test_auc"] for r in self.results if r["arm"] == "D1"]
        print(f"\nD1 (synthetic-only) mean test AUC: {np.mean(aucs):.4f}  "
              f"[C1 MSF ref {self.cfg.c1_auc}; real baselines ~0.94]")
        print("Near C1/baseline => the synthetic set transmits MSF's decision function (distillation-like).")

    def msf_test_predictions(self):
        """MSF's own reverse-flow classification of the real test split (subprocess), cached to Drive."""
        out_csv = self.cfg.save_dir / "msf_test_predictions.csv"
        if out_csv.exists():
            return pd.read_csv(out_csv)
        cmd = [
            "python", "project/classify_pneumoniamnist.py",
            "--checkpoint", self.cfg.checkpoint_path, "--output_csv", str(out_csv),
            "--dataset", "pneumoniamnist", "--n_classes", "2",
            "--image_size", str(self.cfg.gen_image_size), "--beta", str(self.cfg.gen_beta),
            "--rgb_mask", "--solver", "euler", "--step_size", str(self.cfg.gen_step_size),
            "--model_channels", "64", "--num_res_blocks", "2",
            "--channel_mult", "1", "2", "2", "2",
            "--num_heads", "4", "--num_head_channels", "64", "--attention_resolutions", "2",
        ]
        env = dict(os.environ, PYTHONPATH=f"{self.cfg.medsymm_root}/src")
        res = subprocess.run(cmd, cwd=self.cfg.medsymm_root, env=env, capture_output=True, text=True)
        print(res.stdout.strip()[-800:])
        if res.returncode != 0:
            print(res.stderr[-2500:])
            raise RuntimeError("MSF classification failed")
        return pd.read_csv(out_csv)

    def distillation_agreement(self):
        """Fingerprint: does a synthetic-trained ResNet (D1) copy MSF's predictions --
        especially MSF's *errors* -- more than a real-trained ResNet (B0)? Copying errors
        needs copying the function, which mere data-manifold coverage cannot explain."""
        msf = self.msf_test_predictions()
        ty, msf_pred = msf["true"].values, msf["msf_pred"].values

        real_model = self.baseline_models[(max(self.cfg.budgets), self.cfg.seeds[0])]
        syn_model = self.d1_models[self.cfg.seeds[0]]
        ry, rp = self.predict_probs(real_model, self.loader(self.test_set))
        _, sp = self.predict_probs(syn_model, self.loader(self.test_set))
        assert np.array_equal(ry, ty), "test order mismatch between MSF CSV and loader"
        rp = (rp >= 0.5).astype(int)
        sp = (sp >= 0.5).astype(int)

        err = msf_pred != ty  # images MSF gets wrong
        def agree(pred):
            on_err = float((pred[err] == msf_pred[err]).mean()) if err.any() else np.nan
            return round(float((pred == msf_pred).mean()), 3), round(on_err, 3)
        r_all, r_err = agree(rp)
        s_all, s_err = agree(sp)
        print(f"MSF test accuracy: {(msf_pred == ty).mean():.3f}  (MSF errors: {int(err.sum())}/{len(ty)})")
        print("Higher 'agree_on_MSF_errors' for the synthetic-trained model = distillation fingerprint.")
        return pd.DataFrame([
            {"model": "real-trained (B0)", "agree_with_MSF": r_all, "agree_on_MSF_errors": r_err},
            {"model": "synthetic-trained (D1)", "agree_with_MSF": s_all, "agree_on_MSF_errors": s_err},
        ])

    # ----------------------------------------------------- reference & summary
    def record_c1(self):
        for budget in self.cfg.budgets:
            self.results.append({"arm": "C1", "budget": budget, "seed": -1, "val_auc": np.nan,
                                 "test_auc": self.cfg.c1_auc, "test_acc": self.cfg.c1_acc,
                                 "test_balacc": np.nan, "test_f1": np.nan})
        print(f"C1 MSF reference recorded: AUC {self.cfg.c1_auc}, ACC {self.cfg.c1_acc}")

    @staticmethod
    def _ci95(x):
        x = x.dropna().values
        if len(x) < 2:
            return np.nan
        return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))

    def summarize(self):
        res = pd.DataFrame(self.results)
        res.to_csv(self.results_path, index=False)
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
        print("Saved results ->", self.results_path)
        return summary, comparison

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
