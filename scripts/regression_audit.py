"""docs/regression_check.md Phase 2 (only if the check finds a regression): audit, no fix.

    uv run python scripts/regression_audit.py

For seeds 0-7 and both code paths, runs `scripts/train.py` with v0.1's arguments up to the
call of `train`, with `sokudan.train.loop.train` replaced by a recorder, in a fresh process
each time (`--device cpu`, CUDA hidden: construction happens on the CPU either way):

- **old**: `scripts/train.py` as of f75e3f2^ (build the model, then `train` seeds);
- **new**: the current `scripts/train.py` (`set_seed`, build, then `train` seeds again).

The recorder writes, before anything else: the head parameters' mean / SD / max |x| and
hash, every parameter's dtype, `torch.initial_seed()`; then it calls `set_seed(seed)` as
`train` does first, and records what the generators draw next (python `random`, numpy,
torch CPU) and the first 64 indices of the epoch-0 shuffle `train` would make. The old
path is run twice for seeds 0 and 1 to see whether it is reproducible across processes.
Writes `runs/regression/audit.json`. No training.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

OLD_SCRIPT = Path("runs/regression/train_pre_fix.py")
OUT = Path("runs/regression/audit.json")
HEADS = ("scorer.weight", "scorer.bias", "ordinal.cut.weight", "ordinal.cut.bias",
         "ordinal.location.weight", "ordinal.location.bias")

PROBE = r"""
import hashlib, json, random, runpy, sys
import numpy as np
import torch
import sokudan.train.loop as loop

script, seed, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]

def recorder(model, tokenizer, train_examples, val_examples, config, **_):
    params = dict(model.named_parameters())
    rec = {"initial_seed_at_train": torch.initial_seed(),
           "dtypes": sorted({str(p.dtype) for p in params.values()}),
           "heads": {}}
    for name in HEADS_PLACEHOLDER:
        if name in params:
            t = params[name].detach().double()
            rec["heads"][name] = {
                "mean": float(t.mean()), "sd": float(t.std()) if t.numel() > 1 else 0.0,
                "max_abs": float(t.abs().max()), "numel": t.numel(),
                "sha": hashlib.sha256(params[name].detach().numpy().tobytes()).hexdigest()[:16]}
    loop.set_seed(config.seed)
    rec["after_set_seed"] = {"python": random.random(), "numpy": float(np.random.rand()),
                             "torch": torch.rand(3).tolist()}
    order = list(range(len(train_examples)))
    random.Random(config.seed).shuffle(order)
    rec["shuffle_head"] = order[:64]
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rec, fh)
    raise SystemExit(0)

loop.train = recorder
sys.argv = [script, "--train", "data/train_v2b.jsonl", "--val", "data/val_v2.jsonl",
            "--encoding", "joint", "--epochs", "2", "--batch-size", "24", "--lr", "2e-5",
            "--head-lr", "2e-4", "--seed", str(seed), "--input-order", "question_first",
            "--device", "cpu", "--run-id", "audit_probe"]
runpy.run_path(script, run_name="__main__")
""".replace("HEADS_PLACEHOLDER", repr(HEADS))


def probe(script: str, seed: int, tag: str) -> dict:
    out = Path(f"runs/regression/audit_{tag}.json")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    code = subprocess.run(["uv", "run", "python", "-c", PROBE, script, str(seed), str(out)],
                          env=env, capture_output=True, text=True, encoding="utf-8",
                          errors="replace")
    if not out.exists():
        raise SystemExit(f"probe {tag} failed ({code.returncode}):\n{code.stderr[-3000:]}")
    return json.loads(out.read_text(encoding="utf-8"))


def main() -> int:
    old_source = subprocess.run(["git", "show", "f75e3f2^:scripts/train.py"],
                                capture_output=True, text=True, encoding="utf-8",
                                check=True).stdout
    OLD_SCRIPT.write_text(old_source, encoding="utf-8")
    res: dict = {"old_script_sha256": hashlib.sha256(old_source.encode()).hexdigest(),
                 "runs": {}}
    for seed in range(8):
        for flow, script in (("old", str(OLD_SCRIPT)), ("new", "scripts/train.py")):
            res["runs"][f"{flow}_seed{seed}"] = probe(script, seed, f"{flow}_seed{seed}")
            print(f"{flow} seed {seed} done", flush=True)
    for seed in (0, 1):
        res["runs"][f"old_seed{seed}_repeat"] = probe(str(OLD_SCRIPT), seed,
                                                     f"old_seed{seed}_repeat")
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
