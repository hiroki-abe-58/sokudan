"""Phase 6.2 of docs/length_2x2.md: where do implies_declining errors move?

    uv run python scripts/l2x2_declining.py

Rules fixed in docs/length_2x2.md §11 before this ran: frozen `implies_declining` rows,
qf-short and sf-short 3 seeds each, rows correct for all three qf-short seeds and
wrong for all three sf-short seeds, and the first / last position of a fixed list of
"can't take this on" expressions in the state. Counts only.
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
from sokudan.train.loop import TrainConfig, build_collator

QF = [f"runs/v01_seed{s}/model.pt" for s in range(3)]
SF = [f"runs/io_sf_seed{s}/model.pt" for s in range(3)]
CUES = ("難しい 難しく 厳しい 厳しく 困難 余裕がな 手が回ら 手が空か 立て込 都合がつか 都合が合わ "
        "調整が難 見合わ 折り合 条件が合わ 現状では 当面 しかね かねます できかね 対応できな "
        "できない できません 控え 今回は").split()


def positions(state: str) -> tuple[float, float] | None:
    hits = [i for cue in CUES for i in range(len(state)) if state.startswith(cue, i)]
    if not hits:
        return None
    return min(hits) / len(state), max(hits) / len(state)


def classify(rows: list[dict]) -> dict:
    first, last = Counter(), Counter()
    for r in rows:
        pos = positions(r["state"])
        if pos is None:
            first["該当なし"] += 1
            last["該当なし"] += 1
            continue
        first["前半" if pos[0] < 0.5 else "後半"] += 1
        last["前半" if pos[1] < 0.5 else "後半"] += 1
    return {"n": len(rows), "first_cue": dict(first), "last_cue": dict(last)}


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

    qf, sf = correct(QF), correct(SF)
    to_wrong = qf.all(axis=0) & (~sf).all(axis=0)
    to_right = (~qf).all(axis=0) & sf.all(axis=0)

    def truth(r: dict) -> int:  # does the document imply declining?
        return 1 - int(r["label"]) if r.get("negation") else int(r["label"])

    flipped = [r for r, f in zip(rows, to_wrong, strict=True) if f]
    fixed = [r for r, f in zip(rows, to_right, strict=True) if f]
    report = {
        "rows": len(rows),
        "qf_short_accuracy_per_seed": qf.mean(axis=1).tolist(),
        "sf_short_accuracy_per_seed": sf.mean(axis=1).tolist(),
        "correct_all_qf_wrong_all_sf": len(flipped),
        "wrong_all_qf_correct_all_sf": len(fixed),
        "flipped_by_doc_truth": dict(Counter(truth(r) for r in flipped)),
        "flipped_by_negation": dict(Counter(bool(r.get("negation")) for r in flipped)),
        "fixed_by_doc_truth": dict(Counter(truth(r) for r in fixed)),
        "positions_flipped_implying": classify([r for r in flipped if truth(r) == 1]),
        "positions_fixed_implying": classify([r for r in fixed if truth(r) == 1]),
        "positions_all_implying": classify([r for r in rows if truth(r) == 1]),
        "flipped_doc_ids": sorted({r["doc_id"] for r in flipped}),
    }
    out = Path("runs/l2x2/declining.json")
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "flipped_doc_ids"},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
