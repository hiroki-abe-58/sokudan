"""Reference records for the performance work A (numbers must not change).

    uv run python scripts/perf_record.py batches --out runs/perf/batches.json         # CPU
    uv run python scripts/perf_record.py steps --out runs/perf/steps.json             # GPU
    uv run python scripts/perf_record.py preds --checkpoint runs/v01_seed0/model.pt \
        --out runs/perf/preds.npz                                                     # GPU

Run once from a worktree of the code *before* A (with `PYTHONPATH` pointing at it) and
once from the code with A; `tests/test_perf_identity.py` compares the two.

- `batches`: v0.1's two epochs of `length_bucketed_batches` (seed 0, batch 24), as the
  training loop draws them from one `random.Random(seed)`; each epoch's batch list as
  example indices, and its SHA-256.
- `steps`: v0.1's configuration, seed 0, deterministic mode
  (`torch.use_deterministic_algorithms(True, warn_only=True)`, cuDNN deterministic,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`), the first 20 training steps: each step's batch as
  example indices and its loss (`float.hex`, exact).
- `preds`: per-row predictions of one checkpoint on the frozen held-out set, the
  held-out rows of the long-document val set (P(true)) and `val_v2` (all options),
  batch 32 -- the scoring of `scripts/soup_eval.collect`.

`--data-root` is the repository whose `data/` is read (a worktree has no data).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

STEPS = 20


class StopRecording(Exception):
    pass


def batches(data: Path) -> dict:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.train.dataset import length_bucketed_batches, load_examples

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    examples = load_examples(data / "data" / "train_v2b.jsonl")
    index_of = {id(e): i for i, e in enumerate(examples)}
    rng = random.Random(0)
    out = {"n_examples": len(examples), "epochs": []}
    for _ in range(2):
        groups = length_bucketed_batches(examples, 24, tokenizer, rng=rng)
        indices = [[index_of[id(e)] for e in g] for g in groups]
        blob = json.dumps(indices).encode()
        out["epochs"].append({"n_batches": len(indices), "sha256": hashlib.sha256(blob).hexdigest(),
                              "first": indices[:3]})
    return out


def steps(data: Path) -> dict:
    import torch

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    from transformers import AutoTokenizer

    from scripts.train import build_model
    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.train.dataset import load_examples
    from sokudan.train.loop import TrainConfig, train

    args = argparse.Namespace(seed=0, encoding="joint", no_ordinal=False, local_attention=None,
                              input_order="question_first", head_layers=2)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    train_examples = load_examples(data / "data" / "train_v2b.jsonl")
    val_examples = load_examples(data / "data" / "val_v2.jsonl")
    index_of = {id(e): i for i, e in enumerate(train_examples)}
    # Seeded here, not by build_model (which no longer seeds, docs/regression_check.md §10),
    # so the record is the same model whichever version of build_model runs it.
    from sokudan.train.loop import set_seed

    set_seed(0)
    model = build_model(args)
    config = TrainConfig(seed=0, epochs=2, batch_size=24, learning_rate=2e-5,
                         head_learning_rate=2e-4, device="cuda", encoding="joint",
                         input_order="question_first")
    record: list[dict] = []

    def hook(epoch, batch_index, group, loss):
        record.append({"epoch": epoch, "batch": batch_index,
                       "indices": [index_of[id(e)] for e in group],
                       "loss": float(loss).hex()})
        if len(record) >= STEPS:
            raise StopRecording

    try:
        train(model, tokenizer, train_examples, val_examples, config, step_hook=hook)
    except StopRecording:
        pass
    return {"steps": record, "deterministic": True, "torch": torch.__version__}


def preds(data: Path, checkpoint: str, out: Path) -> None:
    import numpy as np
    import torch
    from transformers import AutoTokenizer

    from scripts.ens_collect import val_probs
    from scripts.eval_heldout import load_checkpoint, load_rows
    from scripts.eval_local_attention import p_true
    from scripts.eval_long_states import rows_from_documents
    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.train.dataset import load_examples
    from sokudan.train.loop import TrainConfig, build_collator

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    frozen = load_rows(data / "data" / "v2" / "heldout_val_v2.jsonl")
    long_rows = [r for r in rows_from_documents(data / "data" / "docs_long_val.jsonl")
                 if r["held_out"]]
    val = load_examples(data / "data" / "val_v2.jsonl")
    model, encoding, device = load_checkpoint(checkpoint)
    config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                         input_order=model.input_order)
    collator = build_collator(tokenizer, config)
    f = p_true(model, frozen, collator, config)
    g = p_true(model, long_rows, collator, config)
    v, n = val_probs(model, val, collator, config)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, frozen=f, long=g, val=v, val_n=n)
    del model
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("batches", "steps", "preds"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--checkpoint", default="runs/v01_seed0/model.pt")
    parser.add_argument("--data-root", default=str(Path(__file__).resolve().parents[1]))
    args = parser.parse_args()
    data = Path(args.data_root)
    out = Path(args.out)
    if args.mode == "preds":
        preds(data, args.checkpoint, out)
        return 0
    result = batches(data) if args.mode == "batches" else steps(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
