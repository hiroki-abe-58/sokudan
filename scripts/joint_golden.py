"""Fingerprint the joint encoding, so a change to it can be checked byte for byte.

    uv run python scripts/joint_golden.py --write tests/fixtures/joint_golden.json
    uv run python scripts/joint_golden.py --check tests/fixtures/joint_golden.json

`--write` was run on the code *before* `input_order` existed (commit 709abea), so the
fixture is the pre-change output. It stores, for a few rows of each kind, the
`encode_joint` fields and the collated tensors, plus a sha256 over every row of
each data file -- `input_ids`, `attention_mask` and `marker_positions` in the default
order. `--check` recomputes and compares.

`--input-order state_first` does the same for the other order;
`tests/fixtures/joint_golden_state_first.json` was written on commit ca7a601, before the
`sandwich` order was added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import sokudan.config  # noqa: F401
from sokudan.encoding.question import encode_joint
from sokudan.train.dataset import JointCollator, load_examples

FILES = ["data/val_v2.jsonl", "data/v2/heldout_val_v2.jsonl", "data/train_v2b.jsonl",
         "data/v4/heldout_val_v4.jsonl"]
SAMPLE_FILE = "data/val_v2.jsonl"
PER_KIND = 3


def fingerprint(tokenizer, input_order: str = "question_first") -> dict:
    out: dict = {"samples": [], "file_sha256": {}}
    examples = load_examples(SAMPLE_FILE)
    picked = []
    for kind in ("choice", "score", "bool"):
        picked += [e for e in examples if e.kind == kind][:PER_KIND]
    for e in picked:
        enc = encode_joint(e.question, e.state, tokenizer, input_order=input_order)
        # Force truncation: 20 state tokens fewer than the pair needs.
        limit = len(enc.input_ids) - min(20, enc.n_state_tokens - 1)
        small = encode_joint(e.question, e.state, tokenizer, max_tokens=limit,
                             input_order=input_order)
        out["samples"].append({
            "kind": e.kind, "state": e.state,
            "question": e.question.model_dump(mode="json"),
            "input_ids": enc.input_ids, "attention_mask": enc.attention_mask,
            "marker_positions": enc.marker_positions,
            "truncated_minus20": {"input_ids": small.input_ids,
                             "marker_positions": small.marker_positions,
                             "truncated": small.truncated},
        })
    batch = JointCollator(tokenizer, input_order=input_order)(picked)
    out["collated"] = {"input_ids": batch.input_ids.tolist(),
                       "attention_mask": batch.attention_mask.tolist(),
                       "marker_positions": batch.marker_positions.tolist(),
                       "marker_mask": batch.marker_mask.tolist()}
    for path in FILES:
        h = hashlib.sha256()
        for e in load_examples(path):
            enc = encode_joint(e.question, e.state, tokenizer, input_order=input_order)
            h.update(json.dumps([enc.input_ids, enc.attention_mask,
                                 enc.marker_positions]).encode())
        out["file_sha256"][path] = h.hexdigest()
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write")
    group.add_argument("--check")
    parser.add_argument("--input-order", default="question_first")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    current = fingerprint(tokenizer, args.input_order)
    if args.write:
        Path(args.write).write_text(json.dumps(current, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(current["file_sha256"], indent=2))
        return 0
    stored = json.loads(Path(args.check).read_text(encoding="utf-8"))
    ok = True
    for path, digest in stored["file_sha256"].items():
        same = current["file_sha256"][path] == digest
        ok &= same
        print(f"{'same' if same else 'DIFFERENT'}  {path}")
    same = current["samples"] == stored["samples"] and current["collated"] == stored["collated"]
    ok &= same
    print(f"{'same' if same else 'DIFFERENT'}  samples + collated batch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
