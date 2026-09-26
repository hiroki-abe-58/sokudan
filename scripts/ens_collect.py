"""Per-row predictions for docs/ensemble.md, one file per checkpoint.

    uv run python scripts/ens_collect.py                 # v0.1 x8, io_sf x8, soup
    uv run python scripts/ens_collect.py --only soup     # just the soup

For every checkpoint, writes `runs/ens/probs_<name>.npz` with, in a fixed row order:

- `frozen`: P(true) on every frozen held-out row (`data/v2/heldout_val_v2.jsonl`);
- `long`: P(true) on every held-out row of `data/docs_long_val.jsonl`;
- `val`: the option probabilities of every `data/val_v2.jsonl` row (padded with NaN),
  with `val_n` options per row.

Scoring is the evaluation's own: `scripts/eval_local_attention.p_true` for the
held-out rows, and for val the same bucketing, batch size (32) and autocast as
`sokudan.train.loop.evaluate`, only keyed by row instead of by kind. Each model is
fed its own saved input order.

The soup is the plain average of all eight v0.1 state dicts (floating tensors
averaged, integer tensors taken from seed 0), saved as `runs/ens/soup_v01/model.pt`
with seed 0's config.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from scripts.eval_local_attention import p_true
from scripts.eval_long_states import rows_from_documents
from sokudan.train.dataset import length_bucketed_batches, load_examples
from sokudan.train.loop import TrainConfig, _amp_dtype, build_collator, run_model

OUT = Path("runs/ens")
CHECKPOINTS = {**{f"v01_seed{s}": f"runs/v01_seed{s}/model.pt" for s in range(8)},
               **{f"io_sf_seed{s}": f"runs/io_sf_seed{s}/model.pt" for s in range(8)}}
SOUP = OUT / "soup_v01" / "model.pt"


def make_soup() -> Path:
    blobs = [torch.load(f"runs/v01_seed{s}/model.pt", map_location="cpu", weights_only=False)
             for s in range(8)]
    averaged = {}
    for key, first in blobs[0]["state_dict"].items():
        if torch.is_floating_point(first):
            averaged[key] = torch.stack([b["state_dict"][key].float() for b in blobs]).mean(
                0).to(first.dtype)
        else:
            averaged[key] = first.clone()
    SOUP.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": averaged, "config": dict(blobs[0]["config"])}, SOUP)
    return SOUP


@torch.no_grad()
def val_probs(model, examples, collator, config) -> tuple[np.ndarray, np.ndarray]:
    """`loop.evaluate`'s batching and autocast, but each row's probabilities kept in place."""
    device = torch.device(config.device)
    amp = _amp_dtype(config.amp_dtype)
    index_of = {id(e): i for i, e in enumerate(examples)}
    width = 16
    probs = np.full((len(examples), width), np.nan)
    n_opts = np.zeros(len(examples), dtype=int)
    model.eval()
    for group in length_bucketed_batches(examples, config.batch_size, collator.tokenizer,
                                         rng=random.Random(0), shuffle=False):
        batch = collator(group).to(device)
        with torch.autocast(device_type=device.type, dtype=amp, enabled=amp != torch.float32):
            out = run_model(model, batch)
        p = out.probs.float().cpu().numpy()
        for row, example in enumerate(group):
            n = int(batch.marker_mask[row].sum())
            i = index_of[id(example)]
            probs[i, :n] = p[row, :n]
            n_opts[i] = n
    return probs, n_opts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    frozen = load_rows(Path("data/v2/heldout_val_v2.jsonl"))
    long_rows = [r for r in rows_from_documents(Path("data/docs_long_val.jsonl")) if r["held_out"]]
    val = load_examples("data/val_v2.jsonl")
    todo = dict(CHECKPOINTS)
    todo["soup_v01"] = str(SOUP)
    if args.only:
        todo = {k: v for k, v in todo.items() if k in args.only}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, path in todo.items():
        if name == "soup_v01" and not SOUP.exists():
            make_soup()
        model, encoding, device = load_checkpoint(path)
        config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                             input_order=model.input_order)
        collator = build_collator(tokenizer, config)
        f = p_true(model, frozen, collator, config)
        g = p_true(model, long_rows, collator, config)
        v, n = val_probs(model, val, collator, config)
        np.savez(OUT / f"probs_{name}.npz", frozen=f, long=g, val=v, val_n=n,
                 input_order=np.array(model.input_order))
        print(f"{name}: {model.input_order}, saved", flush=True)
        del model
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
