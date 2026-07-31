"""Record reproducibility fixtures from the CURRENT code, before refactoring.

`train_classifier` does `set_seed(seed)` and then constructs the model; `nn.Conv2d`
and `nn.Linear` consume the global torch RNG at construction time. So any change
that reorders module construction silently shifts every initialisation -- and the
resulting drift looks exactly like a real experimental effect.

This script captures the fingerprints that prove a later refactor is numerically
inert. RUN IT ONCE ON THE PRE-REFACTOR CODE, then keep the JSON:

    python project/record_fixtures.py --out /content/drive/MyDrive/MedSymmFlow_Project

Afterwards `Experiment.selftest_repro()` verifies against the same file.
"""
import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def sha1_state_dict(sd):
    """Order-independent-by-key, value-exact digest of a state dict."""
    h = hashlib.sha1()
    for k in sorted(sd):
        v = sd[k]
        h.update(k.encode())
        buf = io.BytesIO()
        import torch
        torch.save(v.detach().cpu().contiguous(), buf)
        h.update(buf.getvalue())
    return h.hexdigest()


def sha1_ints(values):
    return hashlib.sha1(",".join(str(int(v)) for v in values).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory to write fixtures.json into")
    ap.add_argument("--name", default="fixtures_prerefactor.json")
    args = ap.parse_args()

    from augmentation import Experiment, Config

    exp = Experiment(Config(quick=True, save_dir=args.out))
    exp.setup_data()

    fx = {"schema": 1}

    # 1. Model init: seed, then build, exactly as train_classifier does.
    exp.set_seed(0)
    model = exp.build_resnet18()
    fx["model_init_sha1_seed0"] = sha1_state_dict(model.state_dict())

    # 2. Subsampling: same RNG consumption, same indices.
    for n in (250, 500, 1000):
        idx, labels = exp.stratified_subset(n, 0)
        fx[f"subset_{n}_seed0_sha1"] = sha1_ints(idx)
        fx[f"subset_{n}_seed0_len"] = len(idx)
        fx[f"subset_{n}_seed0_pos"] = int((labels == 1).sum())

    # 3. Split sizes -- guards against a silent medmnist change.
    fx["split_sizes"] = [len(exp.train_set), len(exp.val_set), len(exp.test_set)]

    dest = Path(args.out) / args.name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(fx, indent=2))
    print(json.dumps(fx, indent=2))
    print("\nwrote", dest)


if __name__ == "__main__":
    main()
