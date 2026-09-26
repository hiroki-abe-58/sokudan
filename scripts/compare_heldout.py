"""Compare checkpoints on a held-out set, split into the attributes each model has seen.

    uv run python scripts/compare_heldout.py
        --checkpoints runs/v01_seed0/model.pt runs/v03_seed0/model.pt

Day 3 widened the held-out set from 5 attributes to 12, which makes a single pooled
number ambiguous: a model can move because it reads intent better, or because seven of
the twelve attributes are new and happen to suit it. So the pool is reported three ways
-- all twelve, the five that were held out for v0.1, and the seven added on Day 3 -- and
the five are the ones that answer "did this beat v0.1", because they are the only
attributes measured on both models under the same definition.

`scripts/eval_heldout.py` decides the gate. This only regroups its per-attribute output,
so the two cannot disagree about what was measured.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from sokudan.calibration.metrics import auroc
from sokudan.data import intent_attributes as ia
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, length_bucketed_batches, run_model

# The five attributes v0.1 was measured on. Anything else in HELD_OUT arrived on Day 3.
V01_HELD_OUT = frozenset({
    "implies_running_out_of_patience", "implies_declining", "implies_escalation",
    "requests_owner_change", "ends_with_question",
})


@torch.no_grad()
def score(model: Any, rows: list[dict], collator: Any, config: TrainConfig
          ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    import random as _random

    examples = [Example.from_row(r) for r in rows]
    by_example = {id(e): r for e, r in zip(examples, rows, strict=True)}
    buckets: dict[str, dict[str, list]] = defaultdict(lambda: {"p": [], "g": []})

    device = torch.device(config.device)
    model.eval()
    for group in length_bucketed_batches(
        examples, config.batch_size, collator.tokenizer,
        rng=_random.Random(0), shuffle=False,
    ):
        batch = collator(group).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = run_model(model, batch)
        probs = out.probs.float().cpu().numpy()
        labels = batch.labels.cpu().numpy()
        for index, example in enumerate(group):
            row = by_example[id(example)]
            attribute = row["attribute"]
            tier = row.get("tier") or ia.BY_ID[attribute].tier
            era = "v0.1 の 5 属性" if attribute in V01_HELD_OUT else "Day 3 で追加"
            keys = [f"{era}/all", f"{era}/{tier}", "全体/all", f"全体/{tier}",
                    f"attr:{attribute}"]
            for key in keys:
                buckets[key]["p"].append(float(probs[index, 1]))
                buckets[key]["g"].append(int(labels[index]))
    return {k: (np.array(v["p"]), np.array(v["g"])) for k, v in buckets.items()}


def summarise(scored: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    out = {}
    for key, (p, g) in scored.items():
        if len(g) < 20 or not (0 < g.sum() < len(g)):
            continue
        out[key] = {"n": int(len(g)), "auroc": float(auroc(p, g)),
                    "accuracy": float(((p > 0.5).astype(int) == g).mean())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default="data/v4/heldout_val_v4.jsonl")
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="runs/diagnostics/heldout_compare.json")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    rows = load_rows(Path(args.heldout))
    attributes = sorted({r["attribute"] for r in rows})
    print(f"{len(rows)} views over {len(attributes)} attributes "
          f"from {len({r['doc_id'] for r in rows})} documents")
    print(f"  v0.1 の 5 属性: {sorted(a for a in attributes if a in V01_HELD_OUT)}")
    print(f"  Day 3 で追加  : {sorted(a for a in attributes if a not in V01_HELD_OUT)}\n")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    results: dict[str, Any] = {}
    for checkpoint in args.checkpoints:
        model, encoding, device = load_checkpoint(checkpoint)
        config = TrainConfig(device=device, batch_size=args.batch_size, encoding=encoding,
                             input_order=getattr(model, "input_order", "question_first"))
        collator = build_collator(tokenizer, config)
        results[checkpoint] = summarise(score(model, rows, collator, config))
        del model
        torch.cuda.empty_cache()
        print(f"scored {checkpoint}", flush=True)

    names = {c: Path(c).parent.name for c in args.checkpoints}
    groups = [k for k in sorted({k for r in results.values() for k in r})
              if not k.startswith("attr:")]
    print(f"\n{'group':18s}" + "".join(f"{names[c]:>20s}" for c in args.checkpoints))
    print("-" * (18 + 20 * len(args.checkpoints)))
    for key in groups:
        line = f"{key:18s}"
        for checkpoint in args.checkpoints:
            entry = results[checkpoint].get(key)
            line += f"{entry['auroc']:>13.4f}(n={entry['n']})" if entry else f"{'—':>20s}"
        print(line)

    if len(args.checkpoints) > 1:
        base = args.checkpoints[0]
        for other in args.checkpoints[1:]:
            print(f"\n差分 ({names[other]} − {names[base]}):")
            for key in groups:
                a, b = results[base].get(key), results[other].get(key)
                if a and b:
                    print(f"  {key:18s} {b['auroc'] - a['auroc']:+.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"heldout": args.heldout, "by_checkpoint": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
