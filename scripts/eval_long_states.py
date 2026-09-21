"""§6: does accuracy fall as the state gets longer?

    uv run python scripts/eval_long_states.py --checkpoint runs/v01_seed0/model.pt

The backbone runs `local_attention: 128` with global attention every third layer.
`docs/benchmarks.md` records that the training states average 134 tokens -- about the
width of that window -- so the mechanism §6.1 originally worried about never had a
chance to fire. These documents are generated at 600-2200 characters to push past it.

Scored by *token* bin rather than character bin, because the window is counted in
tokens. Held-out attributes and trained attributes are reported separately: a drop on
both is a length effect, a drop on held-out only would be something else.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint
from sokudan.calibration.metrics import auroc
from sokudan.data import intent_attributes as ia
from sokudan.data.schema_aug import LabelledQuestion
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, run_model

BINS = [(0, 200), (200, 400), (400, 600), (600, 10**9)]


def rows_from_documents(path: Path) -> list[dict[str, Any]]:
    """One row per (document, kept attribute), using the canonical question form."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        doc = json.loads(line)
        verdicts = doc.get("verdicts") or {}
        for attribute_id, gold in doc["intent_labels"].items():
            answer = verdicts.get(attribute_id)
            if answer is None or answer == "判断できない":
                continue
            if (answer == "はい") != bool(gold):
                continue  # same discard rule as training
            attribute = ia.BY_ID[attribute_id]
            item = LabelledQuestion(attribute.question(0), gold)
            out.append({
                "doc_id": doc["doc_id"], "domain": doc["domain"],
                "attribute": attribute_id, "tier": attribute.tier,
                "held_out": attribute_id in ia.HELD_OUT,
                "kind": "bool", "state": doc["state"],
                "question": item.question.model_dump(mode="json"),
                "label": item.label, "n_options": 2,
            })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/v01_seed0/model.pt")
    parser.add_argument("--docs", default="data/docs_long.jsonl")
    parser.add_argument("--baseline-docs", default="data/docs_v2.jsonl")
    parser.add_argument("--out", default="runs/diagnostics/long_states.json")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    model, encoding, device = load_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    config = TrainConfig(device=device, batch_size=args.batch_size, encoding=encoding)
    collator = build_collator(tokenizer, config)

    rows = rows_from_documents(Path(args.docs))
    rows += rows_from_documents(Path(args.baseline_docs))
    print(f"{len(rows)} rows from {len({r['doc_id'] for r in rows})} documents")

    import torch

    buckets: dict[tuple[str, str], dict[str, list]] = defaultdict(
        lambda: {"p": [], "g": []}
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            batch = collator([Example.from_row(r) for r in chunk]).to(device)
            with torch.autocast(device_type=torch.device(device).type,
                                dtype=torch.bfloat16):
                out = run_model(model, batch)
            probs = out.probs.float().cpu().numpy()
            for index, row in enumerate(chunk):
                n_tokens = len(tokenizer(row["state"], add_special_tokens=True)["input_ids"])
                label = next(f"{lo}-{hi if hi < 10**8 else ''}"
                             for lo, hi in BINS if lo <= n_tokens < hi)
                group = "held-out" if row["held_out"] else "trained"
                buckets[(group, label)]["p"].append(float(probs[index, 1]))
                buckets[(group, label)]["g"].append(int(row["label"]))

    print(f"\n{'group':10s} {'tokens':>10s} {'n':>6s} {'AUROC':>8s} {'acc':>8s}")
    results = []
    for group in ("trained", "held-out"):
        for lo, hi in BINS:
            label = f"{lo}-{hi if hi < 10**8 else ''}"
            entry = buckets.get((group, label))
            if not entry or len(entry["g"]) < 20:
                continue
            p = np.array(entry["p"])
            g = np.array(entry["g"])
            value = auroc(p, g) if 0 < g.sum() < len(g) else float("nan")
            acc = float(((p > 0.5).astype(int) == g).mean())
            results.append({"group": group, "bin": label, "n": int(len(g)),
                            "auroc": float(value), "accuracy": acc})
            print(f"{group:10s} {label:>10s} {len(g):6d} {value:8.4f} {acc:8.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
