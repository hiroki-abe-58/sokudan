"""Token-length distribution of every set the local_attention_1024 comparison uses.

    uv run python scripts/la_lengths.py

Two lengths per row. `state` counts the state alone with special tokens, the way
`scripts/eval_long_states.py` bins it (and the 400-800 band in the pre-registration).
`joint` is what the backbone actually sees in the joint arm -- instructions, options,
markers, state, separators -- untruncated, so rows past `MAX_JOINT_TOKENS` show up.

The thresholds that matter: 65 (below it 128 already sees everything), 513 (below it
1024 sees everything, since 1024 means +-512), and 1024 (the joint truncation limit).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401
from scripts.eval_long_states import rows_from_documents
from sokudan.data import intent_attributes as ia
from sokudan.encoding.question import encode_joint
from sokudan.train.dataset import Example

SETS = {
    "frozen_heldout (data/v2/heldout_val_v2.jsonl)": "data/v2/heldout_val_v2.jsonl",
    "v4_heldout uses_bullet_points (data/v4/heldout_val_v4.jsonl)": "data/v4/heldout_val_v4.jsonl",
    "docs_long_val held-out rows (data/docs_long_val.jsonl)": "data/docs_long_val.jsonl",
    "val_v2 (data/val_v2.jsonl)": "data/val_v2.jsonl",
    "train_v2b (data/train_v2b.jsonl)": "data/train_v2b.jsonl",
}


def load(name: str, path: str) -> list[dict]:
    if path.endswith("docs_long_val.jsonl"):
        return [r for r in rows_from_documents(Path(path)) if r["held_out"]]
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    if "uses_bullet_points" in name:
        rows = [r for r in rows if r["attribute"] == "uses_bullet_points"]
    return rows


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    out = {}
    for name, path in SETS.items():
        rows = load(name, path)
        state_cache: dict[str, int] = {}
        state_len, joint_len = [], []
        for row in rows:
            if row["state"] not in state_cache:
                state_cache[row["state"]] = len(
                    tokenizer(row["state"], add_special_tokens=True)["input_ids"])
            state_len.append(state_cache[row["state"]])
            example = Example.from_row(row)
            joint_len.append(len(encode_joint(example.question, example.state, tokenizer,
                                              max_tokens=10**6).input_ids))
        entry = {"n_rows": len(rows)}
        for label, values in (("state", state_len), ("joint", joint_len)):
            a = np.array(values)
            entry[label] = {
                "mean": round(float(a.mean()), 1), "median": float(np.median(a)),
                "p95": float(np.percentile(a, 95)), "max": int(a.max()),
                "n_ge_65": int((a >= 65).sum()), "n_gt_513": int((a > 513).sum()),
                "n_gt_1024": int((a > 1024).sum()),
                "n_400_800": int(((a >= 400) & (a < 800)).sum()),
            }
        if rows and "attribute" in rows[0]:
            entry["n_400_800_state_by_attribute"] = {}
            for attr in sorted({r["attribute"] for r in rows}):
                idx = [i for i, r in enumerate(rows) if r["attribute"] == attr]
                entry["n_400_800_state_by_attribute"][attr] = int(sum(
                    400 <= state_len[i] < 800 for i in idx))
        entry["held_out_attrs_only"] = all(r.get("attribute") in ia.HELD_OUT
                                           for r in rows if r.get("kind") == "bool")
        out[name] = entry
        print(name, json.dumps(entry, ensure_ascii=False))
    path = Path("runs/diagnostics/la_lengths.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
