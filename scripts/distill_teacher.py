"""docs/distill.md Phase 3: the teacher's distribution on every training row.

    uv run python scripts/distill_teacher.py --teacher E_mix16

The teacher is the mean of its members' probabilities (docs/ensemble.md: bool P(true),
choice option probabilities, score level probabilities). Every member is loaded with
`load_checkpoint`, which restores its own input order, and the collator is built from
that order -- checked here per member (test (d) of docs/distill.md).

Written to `data/cache/distill/teacher_<name>.npz` (gitignored): `probs` (rows x 16,
NaN-padded), `n_options`, `row_sha256` (one per line of `data/train_v2b.jsonl`, in file
order) and the member list with their input orders. Before any inference it asserts
test (c): the rows are the training file's lines one to one, and no row shares an
attribute with v0.1's held-out set, a document with any evaluation set, or a state with
any evaluation set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.ens_collect import val_probs
from scripts.eval_heldout import load_checkpoint, load_rows
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator

TRAIN = Path("data/train_v2b.jsonl")
MEMBERS = {
    "E_qf8": [f"runs/v01_seed{s}/model.pt" for s in range(8)],
    "E_mix16": [f"runs/v01_seed{s}/model.pt" for s in range(8)]
    + [f"runs/io_sf_seed{s}/model.pt" for s in range(8)],
}
EVAL_SETS = ["data/v2/heldout_val_v2.jsonl", "data/heldout_val_v2.jsonl", "data/val_v2.jsonl",
             "data/v4/heldout_val_v4.jsonl"]
V01_HELD_OUT = {"implies_running_out_of_patience", "implies_declining", "implies_escalation",
                "requests_owner_change", "ends_with_question"}
OUT_DIR = Path("data/cache/distill")


def row_hash(line: str) -> str:
    return hashlib.sha256(line.strip().encode("utf-8")).hexdigest()


def doc_key(doc_id: str) -> tuple[str, str]:
    """(corpus, id). Built sets prefix the corpus ("v3-i2-000006" is docs_v3's i2-000006);
    unprefixed ids are docs_v2's, the corpus `train_v2b` and `val_v2` were built from."""
    for corpus in ("v2", "v3", "v4", "v5"):
        if doc_id.startswith(corpus + "-"):
            return corpus, doc_id[len(corpus) + 1:]
    return "v2", doc_id


def check_training_rows(lines: list[str]) -> list[dict]:
    """Test (c): no held-out attribute, evaluation document or evaluation state."""
    rows = [json.loads(line) for line in lines]
    eval_rows = [r for p in EVAL_SETS for r in load_rows(Path(p))]

    def docs(path: str) -> list[dict]:
        return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]

    eval_docs = docs("data/docs_long_val.jsonl") + [
        d for d in docs("data/docs_all_v5.jsonl") if d.get("split") == "val"]
    eval_states = {r["state"] for r in eval_rows} | {d["state"] for d in eval_docs}
    eval_doc_ids = {doc_key(r["doc_id"]) for r in eval_rows}
    assert not {r["attribute"] for r in rows} & V01_HELD_OUT, "a held-out attribute is in training"
    train_docs = {doc_key(r["doc_id"]) for r in rows}
    assert not train_docs & eval_doc_ids, "an evaluation document is in training"
    assert not {r["state"] for r in rows} & eval_states, "an evaluation state is in training"
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=sorted(MEMBERS), required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    lines = [line for line in TRAIN.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = check_training_rows(lines)
    examples = [Example.from_row(r) for r in rows]
    print(f"{len(rows)} training rows checked (no held-out attribute, eval doc or eval state)",
          flush=True)

    total = None
    n_ref = None
    orders = []
    for path in MEMBERS[args.teacher]:
        model, encoding, device = load_checkpoint(path)
        config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                             input_order=model.input_order)
        collator = build_collator(tokenizer, config)
        assert collator.input_order == model.input_order  # (d): each model its own order
        probs, n_opts = val_probs(model, examples, collator, config)
        if total is None:
            total, n_ref = np.nan_to_num(probs), n_opts
        else:
            assert (n_opts == n_ref).all()
            total += np.nan_to_num(probs)
        orders.append((path, model.input_order))
        print(f"{path}: {model.input_order}", flush=True)
        del model
        torch.cuda.empty_cache()
    mean = total / len(orders)
    width = int(n_ref.max())
    mask = np.arange(mean.shape[1])[None, :] >= n_ref[:, None]
    mean[mask] = np.nan
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"teacher_{args.teacher}.npz"
    np.savez(out, probs=mean[:, :max(width, 2)], n_options=n_ref,
             row_sha256=np.array([row_hash(line) for line in lines]),
             members=np.array([p for p, _ in orders]), orders=np.array([o for _, o in orders]))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    sums = np.nansum(mean, axis=1)
    print(f"-> {out} rows {len(rows)} sha256 {digest} row-sum min {sums.min():.6f} "
          f"max {sums.max():.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
