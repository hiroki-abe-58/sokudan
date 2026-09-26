"""Split the held-out set by which corpus its documents came from.

    uv run python scripts/diagnose_v3_domains.py

v0.2 trains on 2.6x the views of v0.1 and scores *lower* on held-out. Two candidate
explanations were confounded in that comparison: the extra data, and the drop from two
epochs to one. This script addresses only the data half.

The merged corpus namespaces document ids by source (`v2-...`, `v3-...`), so the
held-out rows can be split into the twenty-one original domains and the nine added
for v0.2 and scored separately. If both models fall on the v3 slice and hold on the
v2 slice, the new documents are harder or noisier; if both models are flat across the
split, the data is not the problem and the epoch change is left holding it.

Per-attribute discard rates from the two generation runs are printed beside it,
because a label that the generator and the verifier disagreed about more often in v3
than in v2 is the mechanism that would produce the first case.
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


@torch.no_grad()
def score_rows(model: Any, rows: list[dict], collator: Any, config: TrainConfig
               ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return `{group: (p_true, gold)}` keyed by `source/tier`."""
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
            source = row["doc_id"].split("-", 1)[0]
            tier = row.get("tier") or ia.BY_ID[row["attribute"]].tier
            for key in (f"{source}/{tier}", f"{source}/all", f"ALL/{tier}"):
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
    parser.add_argument("--heldout", default="data/v2/heldout_val_v2.jsonl")
    parser.add_argument("--checkpoints", nargs="+",
                        default=["runs/v01_seed0/model.pt", "runs/v02_seed0/model.pt"])
    parser.add_argument("--out", default="runs/diagnostics/v3_domain_split.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    rows = load_rows(Path(args.heldout))
    sources = {r["doc_id"].split("-", 1)[0] for r in rows}
    print(f"{len(rows)} held-out views; sources {sorted(sources)}")
    for source in sorted(sources):
        subset = [r for r in rows if r["doc_id"].startswith(source + "-")]
        docs = len({r["doc_id"] for r in subset})
        print(f"  {source}: {len(subset)} views from {docs} documents")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    results: dict[str, Any] = {}
    for checkpoint in args.checkpoints:
        model, encoding, device = load_checkpoint(checkpoint)
        config = TrainConfig(device=device, batch_size=args.batch_size, encoding=encoding,
                             input_order=getattr(model, "input_order", "question_first"))
        collator = build_collator(tokenizer, config)
        results[checkpoint] = summarise(score_rows(model, rows, collator, config))
        del model
        torch.cuda.empty_cache()

    keys = sorted({k for r in results.values() for k in r})
    name = {c: Path(c).parent.name for c in args.checkpoints}
    print(f"\n{'group':14s}" + "".join(f"{name[c]:>22s}" for c in args.checkpoints))
    print("-" * (14 + 22 * len(args.checkpoints)))
    for key in keys:
        line = f"{key:14s}"
        for checkpoint in args.checkpoints:
            entry = results[checkpoint].get(key)
            line += (f"{entry['auroc']:>14.4f} (n={entry['n']})" if entry
                     else f"{'—':>22s}")
        print(line)

    print("\n差分 (v0.2 − v0.1):")
    a, b = args.checkpoints[0], args.checkpoints[1]
    for key in keys:
        ea, eb = results[a].get(key), results[b].get(key)
        if ea and eb:
            print(f"  {key:14s} {eb['auroc'] - ea['auroc']:+.4f}")

    # Per-attribute discard rates from the two generation runs.
    print("\n検算の破棄率 (v2 対 v3):")
    v2 = json.loads(Path("data/docs_v2.manifest.json").read_text(encoding="utf-8"))
    v3 = json.loads(Path("data/docs_v3.manifest.json").read_text(encoding="utf-8"))
    d2 = {r["attribute"]: r for r in v2["verification"]["per_attribute"]}
    d3 = {r["attribute"]: r for r in v3["verification"]["per_attribute"]}
    deltas = []
    for attribute in sorted(set(d2) | set(d3)):
        r2, r3 = d2.get(attribute), d3.get(attribute)
        if not r2 or not r3:
            continue
        deltas.append((r3["discard_rate"] - r2["discard_rate"], attribute,
                       r2["discard_rate"], r3["discard_rate"], r2["tier"]))
    deltas.sort(reverse=True)
    print(f"  {'attribute':34s} {'tier':5s} {'v2':>7s} {'v3':>7s} {'差':>8s}")
    for delta, attribute, a2, a3, tier in deltas:
        flag = "  <<<" if abs(delta) >= 0.05 else ""
        print(f"  {attribute:34s} {tier:5s} {a2:7.3f} {a3:7.3f} {delta:+8.3f}{flag}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "heldout": args.heldout,
        "by_checkpoint": results,
        "discard_delta": [
            {"attribute": a, "tier": t, "v2": x, "v3": y, "delta": d}
            for d, a, x, y, t in deltas
        ],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
