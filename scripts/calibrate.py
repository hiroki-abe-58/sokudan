"""Stage 2: temperature calibration (SOKUDAN_SPEC.md §8 Stage 2).

    uv run python scripts/calibrate.py --checkpoint runs/s0/model.pt

One temperature per `(question type, option count)` bucket, fitted by minimising NLL
on the **validation** split. §8 requires the ECE at this point to be recorded, because
it is the number that decides whether Stage 3 (RLCD) is worth building at all -- and
§14.3 already deferred Stage 3, so this is where the sprint's calibration story ends.

`bench_ja` is never touched here. Fitting a temperature on the test set and then
reporting the test ECE would make the calibration numbers meaningless, which is the
same trap the cross-fitted Laya row in `docs/baseline_ja.md` was built to avoid.
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
from sokudan.calibration.metrics import brier, ece, nll
from sokudan.calibration.temperature import apply_temperature, fit_temperature
from sokudan.model.sokudan import SokudanModel
from sokudan.train.dataset import Collator, load_examples
from sokudan.train.loop import length_bucketed_batches


@torch.no_grad()
def collect_probabilities(
    model: SokudanModel, examples: list, collator: Collator, device: str, batch_size: int
) -> dict[tuple[str, int], dict[str, list]]:
    """Group validation probabilities by (question type, option count)."""
    import random

    buckets: dict[tuple[str, int], dict[str, list]] = defaultdict(
        lambda: {"probs": [], "labels": []}
    )
    batches = length_bucketed_batches(
        examples, batch_size, collator.tokenizer, rng=random.Random(0), shuffle=False
    )
    for group in batches:
        batch = collator(group).to(device)
        with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16):
            out = model(
                batch.state_input_ids, batch.state_attention_mask,
                batch.question_input_ids, batch.question_attention_mask,
                batch.marker_positions, batch.marker_mask, batch.ordered,
            )
        probs = out.probs.float().cpu().numpy()
        labels = batch.labels.cpu().numpy()
        for row, example in enumerate(group):
            n_options = int(batch.marker_mask[row].sum())
            key = (example.kind, n_options)
            buckets[key]["probs"].append(probs[row, :n_options])
            buckets[key]["labels"].append(int(labels[row]))
    return buckets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/s0/model.pt")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--out", default=None,
                        help="defaults to <checkpoint dir>/temperatures.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-bucket", type=int, default=25,
                        help="buckets smaller than this keep T=1.0")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = blob.get("config", {})
    model = SokudanModel.from_pretrained_backbone(
        config.get("backbone", BACKBONE_MODEL_ID),
        n_head_layers=config.get("n_head_layers", 2),
    )
    model.load_state_dict(blob["state_dict"])
    model.to(args.device).eval()

    tokenizer = AutoTokenizer.from_pretrained(config.get("backbone", BACKBONE_MODEL_ID))
    collator = Collator(tokenizer, max_state_tokens=1024)
    examples = load_examples(args.val)

    buckets = collect_probabilities(model, examples, collator, args.device, args.batch_size)

    print(f"validation: {len(examples)} examples in {len(buckets)} "
          f"(kind, option count) buckets\n")
    header = f"{'bucket':>16} {'n':>6} {'T':>7} {'ECE before':>11} {'ECE after':>10} " \
             f"{'NLL before':>11} {'NLL after':>10}"
    print(header)
    print("-" * len(header))

    temperatures: dict[str, float] = {}
    rows: list[dict[str, Any]] = []
    for (kind, n_options), data in sorted(buckets.items()):
        probs = np.vstack(data["probs"])
        labels = np.asarray(data["labels"])
        n = len(labels)

        before = {"ece": ece(probs, labels), "nll": nll(probs, labels),
                  "brier": brier(probs, labels)}

        if n < args.min_bucket:
            temperature = 1.0
            after = before
            note = "bucket too small to fit"
        else:
            temperature = fit_temperature(probs, labels)
            scaled = apply_temperature(probs, temperature)
            after = {"ece": ece(scaled, labels), "nll": nll(scaled, labels),
                     "brier": brier(scaled, labels)}
            note = ""

        temperatures[f"{kind}/{n_options}"] = temperature
        rows.append({"kind": kind, "n_options": n_options, "n": n,
                     "temperature": temperature, "before": before, "after": after,
                     "note": note})
        print(f"{kind + '/' + str(n_options):>16} {n:>6} {temperature:>7.3f} "
              f"{before['ece']:>11.4f} {after['ece']:>10.4f} "
              f"{before['nll']:>11.4f} {after['nll']:>10.4f}  {note}")

    total = sum(r["n"] for r in rows)
    weighted = {
        stage: sum(r[stage]["ece"] * r["n"] for r in rows) / total
        for stage in ("before", "after")
    }
    print(f"\nweighted mean ECE: {weighted['before']:.4f} -> {weighted['after']:.4f}")

    out_path = Path(args.out) if args.out else Path(args.checkpoint).parent / "temperatures.json"
    out_path.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "val": str(args.val),
                "min_bucket": args.min_bucket,
                "temperatures": temperatures,
                "buckets": rows,
                "weighted_mean_ece": weighted,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"temperatures -> {out_path}")
    print("\nNote: fitted on the validation split only. bench_ja was not used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
