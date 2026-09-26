"""Diagnostic 2: can the pipeline memorise 100 examples?

    uv run python scripts/diagnose_overfit.py --n 100 --epochs 20

If training accuracy on a tiny fixed set does not approach 1.0, something between
encoding and loss has the marker-to-label correspondence wrong, and no amount of
data or tuning would fix it. If it does approach 1.0, the pipeline is sound and the
problem is generalisation.

Evaluated on the *training* set on purpose. Memorisation is the whole point.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sokudan.config  # noqa: F401
from sokudan.config import BACKBONE_MODEL_ID
from sokudan.model.sokudan import SokudanModel
from sokudan.train.dataset import Collator, load_examples
from sokudan.train.loop import TrainConfig, evaluate, train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--out", default="runs/diagnostics/overfit.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    everything = load_examples(args.train)
    # Sample across domains and primitives rather than taking a contiguous slice,
    # which would be dominated by whichever documents happen to sort first.
    rng = random.Random(0)
    subset = rng.sample(everything, args.n)
    kinds = {}
    for example in subset:
        kinds[example.kind] = kinds.get(example.kind, 0) + 1
    print(f"subset of {len(subset)}: {kinds}")

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    model = SokudanModel.from_pretrained_backbone(n_head_layers=2)

    config = TrainConfig(
        seed=0, epochs=args.epochs, batch_size=args.batch_size,
        device=args.device, log_every=10_000,
    )
    # Train on the subset and evaluate on the same subset: can it memorise?
    result = train(model, tokenizer, subset, subset, config)

    collator = Collator(tokenizer, max_state_tokens=config.max_state_tokens)
    final = evaluate(model, subset, collator, config)

    print("\n== 診断2: 100件・20epoch の過学習テスト（train セットで評価）==\n")
    print(f"{'primitive':<10} {'n':>5} {'acc before':>11} {'acc after':>10}")
    print("-" * 40)
    reached = True
    for kind in ("choice", "score", "bool"):
        before = result["before"].get(kind)
        after = final.get(kind)
        if not after:
            continue
        print(f"{kind:<10} {int(after['n']):>5} "
              f"{before['accuracy'] if before else float('nan'):>11.4f} "
              f"{after['accuracy']:>10.4f}")
        if after["accuracy"] < 0.90:
            reached = False

    overall = sum(final[k]["accuracy"] * final[k]["n"] for k in final if k != "n_examples")
    overall /= sum(final[k]["n"] for k in final if k != "n_examples")
    print(f"\n加重平均 train accuracy: {overall:.4f}")

    if overall >= 0.95 and reached:
        verdict = ("PIPELINE OK: the model memorises the subset, so encoding, marker "
                   "positions, labels and loss line up. The problem is generalisation.")
    elif overall >= 0.80:
        verdict = (f"PARTIAL: weighted train accuracy {overall:.4f}. It learns but does "
                   "not fully memorise; look for a primitive that lags.")
    else:
        verdict = (f"PIPELINE BUG: weighted train accuracy only {overall:.4f} after "
                   f"{args.epochs} epochs on {args.n} examples. Something between "
                   "encoding and loss has the marker-to-label correspondence wrong.")
    print(f"\n判定: {verdict}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"n": args.n, "epochs": args.epochs, "kinds": kinds,
                    "before": result["before"], "after": final,
                    "weighted_train_accuracy": overall, "verdict": verdict},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path}")
    return 0 if overall >= 0.80 else 1


if __name__ == "__main__":
    raise SystemExit(main())
