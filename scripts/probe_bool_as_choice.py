"""Does routing `bool` through the `choice` path fix it? (bench_ja, no retraining)

    uv run python scripts/probe_bool_as_choice.py --checkpoint runs/s1/model.pt

`bool` is internally a two-slot question already, but its two slots carry generic
yes/no labels. Asking the same thing as a two-option `choice` gives each slot the
*content* of the proposition instead, which is the one thing the marker
representation never sees in the boolean form.

Both option orders are measured. If the answer depends on which option is listed
first, the routing is not a fix, it is a coin flip with extra steps.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from sokudan.calibration.metrics import auroc, ece, rate_matched_accuracy
from sokudan.eval.baselines import floor_and_renormalise
from sokudan.eval.bench_ja import BenchItem
from sokudan.schema.question import ChoiceQuestion
from sokudan.train.dataset import Collator, Example

TRUE_OPTION = "解約を示唆している"
FALSE_OPTION = "解約を示唆していない"
INSTRUCTIONS = "送信者の意図はどちらか"


@torch.no_grad()
def run(model, tokenizer, items, question, true_index, device, batch_size=24):
    collator = Collator(tokenizer, max_state_tokens=1024)
    rows = []
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        examples = [
            Example(state=item.state, question=question, label=0, ordered=False,
                    doc_id=item.item_id, domain="bench_ja", attribute="churn",
                    kind="choice")
            for item in chunk
        ]
        batch = collator(examples).to(device)
        out = model(
            batch.state_input_ids, batch.state_attention_mask,
            batch.question_input_ids, batch.question_attention_mask,
            batch.marker_positions, batch.marker_mask, batch.ordered,
        )
        rows.append(out.probs.float().cpu().numpy()[:, :2])
    probs = floor_and_renormalise(np.vstack(rows))
    return probs[:, true_index]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/s1/model.pt")
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--out", default="runs/diagnostics/bool_as_choice.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.model.sokudan import SokudanModel

    items = [BenchItem(**json.loads(line))
             for line in Path(args.bench).read_text(encoding="utf-8").splitlines()
             if line.strip()]
    gold = np.array([int(i.churn) for i in items])

    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = blob.get("config", {})
    model = SokudanModel.from_pretrained_backbone(
        config.get("backbone", BACKBONE_MODEL_ID),
        n_head_layers=config.get("n_head_layers", 2))
    model.load_state_dict(blob["state_dict"])
    model.to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(config.get("backbone", BACKBONE_MODEL_ID))

    variants = {
        "true を先に": (ChoiceQuestion(instructions=INSTRUCTIONS,
                                    criteria={TRUE_OPTION: "", FALSE_OPTION: ""}), 0),
        "false を先に": (ChoiceQuestion(instructions=INSTRUCTIONS,
                                     criteria={FALSE_OPTION: "", TRUE_OPTION: ""}), 1),
    }

    print(f"\n== bool を 2択 choice 経路で評価  n={len(items)} ==")
    print(f"   選択肢: 「{TRUE_OPTION}」/「{FALSE_OPTION}」")
    print(f"   gold 正例率 {gold.mean():.3f}  (多数決 {max(gold.mean(), 1-gold.mean()):.3f})\n")
    header = (f"{'条件':<14} {'acc':>7} {'ECE':>7} {'AUROC':>7} "
              f"{'mean P(true)':>13} {'acc@率一致':>11}")
    print(header)
    print("-" * len(header))

    results = {}
    for name, (question, true_index) in variants.items():
        p_true = run(model, tokenizer, items, question, true_index, args.device)
        pred = (p_true >= 0.5).astype(int)
        two_col = np.stack([1 - p_true, p_true], axis=1)
        entry = {
            "accuracy": float((pred == gold).mean()),
            "ece": ece(two_col, gold),
            "auroc": auroc(p_true, gold),
            "mean_p_true": float(p_true.mean()),
            "accuracy_rate_matched": rate_matched_accuracy(p_true, gold)[0],
        }
        results[name] = entry
        print(f"{name:<14} {entry['accuracy']:>7.3f} {entry['ece']:>7.3f} "
              f"{entry['auroc']:>7.3f} {entry['mean_p_true']:>13.3f} "
              f"{entry['accuracy_rate_matched']:>11.3f}")

    accs = [e["accuracy"] for e in results.values()]
    aurocs = [e["auroc"] for e in results.values()]
    print(f"選択肢順による差: acc {abs(accs[0] - accs[1]):.3f}  "
          f"AUROC {abs(aurocs[0] - aurocs[1]):.3f}")

    passed = min(aurocs) > 0.65 and min(accs) > 0.703
    verdict = ("PASS: both orders clear AUROC > 0.65 and acc > 0.703."
               if passed else
               f"FAIL: worst order gives AUROC {min(aurocs):.3f} and acc {min(accs):.3f}; "
               "the thresholds are AUROC > 0.65 and acc > 0.703.")
    print(f"判定: {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"checkpoint": args.checkpoint, "n": len(items), "gold_rate": float(gold.mean()),
         "variants": results, "passed": passed, "verdict": verdict},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
