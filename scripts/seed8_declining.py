"""docs/seed8.md §6.1: implies_declining rows that flip between base12 and long12.

    uv run python scripts/seed8_declining.py

Same rules as docs/length_2x2.md §11 (`scripts/l2x2_declining.py`: the cue list, the
first / last position, the negation flip), with the two groups being the 8 base12 and
the 8 long12 checkpoints: rows correct for all 8 base12 seeds and wrong for all 8
long12 seeds, and the reverse. Counts only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from scripts.eval_local_attention import p_true
from scripts.l2x2_declining import classify
from sokudan.train.loop import TrainConfig, build_collator

BASE = [f"runs/seed8_base12_seed{s}/model.pt" for s in range(8)]
LONG = ([f"runs/l2x2_qf_long_seed{s}/model.pt" for s in range(3)]
        + [f"runs/seed8_long12_seed{s}/model.pt" for s in range(3, 8)])


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    rows = [r for r in load_rows(Path("data/v2/heldout_val_v2.jsonl"))
            if r["attribute"] == "implies_declining"]
    labels = np.array([int(r["label"]) for r in rows])

    def correct(paths: list[str]) -> np.ndarray:
        out = []
        for path in paths:
            model, encoding, device = load_checkpoint(path)
            config = TrainConfig(device=device, batch_size=32, encoding=encoding,
                                 input_order=model.input_order)
            p = p_true(model, rows, build_collator(tokenizer, config), config)
            out.append((p > 0.5).astype(int) == labels)
            del model
            torch.cuda.empty_cache()
        return np.array(out)

    base, long_ = correct(BASE), correct(LONG)
    to_wrong = base.all(axis=0) & (~long_).all(axis=0)
    to_right = (~base).all(axis=0) & long_.all(axis=0)

    def truth(r: dict) -> int:
        return 1 - int(r["label"]) if r.get("negation") else int(r["label"])

    flipped = [r for r, f in zip(rows, to_wrong, strict=True) if f]
    fixed = [r for r, f in zip(rows, to_right, strict=True) if f]
    report = {
        "rows": len(rows),
        "base12_accuracy_per_seed": base.mean(axis=1).tolist(),
        "long12_accuracy_per_seed": long_.mean(axis=1).tolist(),
        "correct_all_base12_wrong_all_long12": len(flipped),
        "wrong_all_base12_correct_all_long12": len(fixed),
        "flipped_by_doc_truth": dict(Counter(truth(r) for r in flipped)),
        "fixed_by_doc_truth": dict(Counter(truth(r) for r in fixed)),
        "positions_flipped_implying": classify([r for r in flipped if truth(r) == 1]),
        "positions_fixed_implying": classify([r for r in fixed if truth(r) == 1]),
        "positions_all_implying": classify([r for r in rows if truth(r) == 1]),
    }
    Path("runs/seed8/declining.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
