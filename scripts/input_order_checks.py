"""Phase 1 checks for docs/input_order.md, over every row the experiment touches.

    uv run python scripts/input_order_checks.py

For each data set, both block orders:

- rows that get truncated (the experiment needs 0: if the order decided which state
  tokens survive, the two arms would not be reading the same text);
- rows whose `state_first` token multiset differs from `question_first`'s (needs 0);
- rows whose `sandwich` tokens are not `question_first`'s plus one question block
  (needs 0);
- rows where a marker position does not land on `<mask>`, or where the number of
  markers differs from the number of options / levels (needs 0).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_rows
from scripts.eval_local_attention import opening_rows
from scripts.eval_long_states import rows_from_documents
from sokudan.encoding.question import INPUT_ORDERS, encode_joint
from sokudan.train.dataset import Example


def data_sets() -> dict[str, list[dict]]:
    return {
        "train_v2b": load_rows(Path("data/train_v2b.jsonl")),
        "val_v2": load_rows(Path("data/val_v2.jsonl")),
        "frozen_heldout_v2": load_rows(Path("data/v2/heldout_val_v2.jsonl")),
        "v4_heldout": load_rows(Path("data/v4/heldout_val_v4.jsonl")),
        "docs_long_val_heldout": [r for r in rows_from_documents(Path("data/docs_long_val.jsonl"))
                                  if r["held_out"]],
        "includes_greeting_val (M4)": opening_rows(),
    }


def check(rows: list[dict], tokenizer: Any) -> dict[str, Any]:
    mask_id = tokenizer.mask_token_id
    out: dict[str, Any] = {"rows": len(rows)}
    truncated = Counter()
    max_len = Counter()
    multiset_diff = 0
    sandwich_extra_differs = 0
    bad_mask = Counter()
    bad_count = Counter()
    for row in rows:
        example = Example.from_row(row)
        n_options = len(example.question.labels)
        encoded = {}
        for order in INPUT_ORDERS:
            enc = encode_joint(example.question, example.state, tokenizer, input_order=order)
            encoded[order] = enc
            truncated[order] += int(enc.truncated)
            max_len[order] = max(max_len[order], len(enc.input_ids))
            positions = enc.marker_positions + (enc.marker_positions_back or [])
            if any(enc.input_ids[p] != mask_id for p in positions):
                bad_mask[order] += 1
            if len(enc.marker_positions) != n_options or (
                    enc.marker_positions_back is not None
                    and len(enc.marker_positions_back) != n_options):
                bad_count[order] += 1
        a, b = encoded["question_first"].input_ids, encoded["state_first"].input_ids
        multiset_diff += int(Counter(a) != Counter(b) or len(a) != len(b))
        # sandwich = question_first + one more question block (and its separator).
        qf, sw = encoded["question_first"], encoded["sandwich"]
        extra = Counter(sw.input_ids) - Counter(qf.input_ids)
        sandwich_extra_differs += int(
            sum(extra.values()) != sw.n_question_tokens - qf.n_question_tokens
            or Counter(qf.input_ids) - Counter(sw.input_ids) != Counter())
    for order in INPUT_ORDERS:
        out[order] = {"truncated": truncated[order], "max_joint_tokens": max_len[order],
                      "marker_not_on_mask": bad_mask[order],
                      "marker_count_mismatch": bad_count[order]}
    out["multiset_differs"] = multiset_diff
    out["sandwich_extra_not_one_block"] = sandwich_extra_differs
    return out


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    results = {}
    for name, rows in data_sets().items():
        results[name] = check(rows, tokenizer)
        print(name, json.dumps(results[name], ensure_ascii=False), flush=True)
    path = Path("runs/diagnostics/input_order_checks.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
