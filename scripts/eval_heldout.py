"""The Day 2 gate: held-out boolean AUROC, reported per tier.

    uv run python scripts/eval_heldout.py --checkpoint runs/s2a/model.pt

Three conditions, all of which must hold before a second and third seed are worth
running:

    held-out overall AUROC  > 0.75
    tier I (implicit) pool  > 0.70
    tier S (surface) pool   > 0.85

One pooled number would not do. Tier S is decidable from the words on the page, so a
model that learned nothing about intent can still carry a combined figure over 0.75
on the strength of tier S alone -- which is Day 1's failure wearing a different hat
(`docs/benchmarks.md` §8). Splitting them makes the three outcomes distinct:

    S low                 -- the wiring is broken, not the data. Look at training.
    S high, I low         -- surface features transferred, intent did not. Stop.
    S high, I high        -- the thing this corpus was built to teach took.

The attributes scored here never appear in training in any form: not the question,
not its three paraphrases, not the negation template, not the option strings
(`sokudan.data.intent_attributes.HELD_OUT`). The documents are from the validation
split, so this is an unseen schema on an unseen document -- the same shift `bench_ja`
measures, which is what makes it usable as a proxy for it.

It is a proxy and not a substitute. Both sides come from one generator, and
`bench_ja` does not. Passing here is a licence to spend GPU time on two more seeds,
nothing more.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

import sokudan.config  # noqa: F401
from sokudan.data import intent_attributes as ia
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, evaluate

GATES = {
    "overall": 0.75,
    "I": 0.70,
    "S": 0.85,
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_checkpoint(path: str) -> tuple[Any, str, str]:
    """Load either arm. The checkpoint says which one it is; nothing is guessed.

    `scripts/train.py` records `encoding` in the checkpoint config, so a joint
    checkpoint cannot be scored through the separate collator (or the reverse) by
    forgetting a flag -- which would silently produce a number for an architecture
    that was never trained.
    """
    from sokudan.config import BACKBONE_MODEL_ID

    blob = torch.load(path, map_location="cpu", weights_only=False)
    stored = blob.get("config", {})
    backbone_id = stored.get("backbone", BACKBONE_MODEL_ID)
    encoding = stored.get("encoding", "separate")

    if encoding == "joint":
        from sokudan.model.joint import SokudanJointModel

        model = SokudanJointModel.from_pretrained_backbone(backbone_id)
    else:
        from sokudan.model.sokudan import SokudanModel

        model = SokudanModel.from_pretrained_backbone(
            backbone_id, n_head_layers=stored.get("n_head_layers", 2)
        )
    model.load_state_dict(blob["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    return model, encoding, device


def apply_state_mode(
    rows: list[dict[str, Any]], mode: str, seed: int = 0
) -> list[dict[str, Any]]:
    """Diagnostic 1 (`docs/benchmarks.md` §3), on whichever arm is loaded.

    Day 1 found `bool` scoring *higher* with an empty state than with the real one.
    The two arms have to be checked the same way or the comparison says nothing, so
    the substitution happens on the rows -- before either collator sees them --
    rather than inside a model-specific path.
    """
    if mode == "real":
        return rows
    if mode == "empty":
        return [{**r, "state": ""} for r in rows]
    if mode == "shuffled":
        rng = random.Random(seed)
        states = [r["state"] for r in rows]
        shifted = states[1:] + states[:1]
        # A different *real* document is the sharper test: an empty state is out of
        # distribution and could degrade for unrelated reasons.
        rng.shuffle(shifted)
        return [{**r, "state": s} for r, s in zip(rows, shifted, strict=True)]
    raise ValueError(f"unknown state mode {mode!r}")


def score(
    rows: list[dict[str, Any]], model: Any, collator: Any, config: TrainConfig
) -> dict[str, Any]:
    if not rows:
        return {}
    examples = [Example.from_row(row) for row in rows]
    result = evaluate(model, examples, collator, config)
    boolean = result.get("bool", {})
    return {
        "n": len(rows),
        "auroc": boolean.get("auroc"),
        "accuracy": boolean.get("accuracy"),
        "ece": boolean.get("ece"),
        "mean_p_true": boolean.get("mean_p_true"),
        "gold_true_rate": boolean.get("gold_true_rate"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/s2a/model.pt")
    parser.add_argument("--heldout", default="data/heldout_val_v2.jsonl")
    parser.add_argument("--out", default="runs/heldout_gate.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--state-mode", choices=("real", "empty", "shuffled"),
                        default="real",
                        help="diagnostic 1: replace the state to test whether it is read")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    model, encoding, device = load_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    config = TrainConfig(device=device, batch_size=args.batch_size, encoding=encoding)
    collator = build_collator(tokenizer, config)

    rows = load_rows(Path(args.heldout))
    rows = apply_state_mode(rows, args.state_mode)
    print(f"{len(rows)} held-out views from {len({r['doc_id'] for r in rows})} documents"
          f"  [encoding={encoding}, state={args.state_mode}]")

    by_tier: dict[str, list] = defaultdict(list)
    by_attribute: dict[str, list] = defaultdict(list)
    for row in rows:
        tier = row.get("tier") or ia.BY_ID[row["attribute"]].tier
        by_tier[tier].append(row)
        by_attribute[row["attribute"]].append(row)

    results: dict[str, Any] = {
        "checkpoint": args.checkpoint,
        "encoding": encoding,
        "state_mode": args.state_mode,
        "overall": score(rows, model, collator, config),
        "by_tier": {
            tier: score(group, model, collator, config)
            for tier, group in sorted(by_tier.items())
        },
        "by_attribute": {
            name: {**score(group, model, collator, config),
                   "tier": ia.BY_ID[name].tier}
            for name, group in sorted(by_attribute.items())
        },
    }

    def check(name: str, value: float | None, threshold: float) -> bool:
        ok = value is not None and value > threshold
        mark = "PASS" if ok else "FAIL"
        shown = "  n/a" if value is None else f"{value:.4f}"
        print(f"  {name:22s} {shown} > {threshold:.2f}   {mark}")
        return ok

    print("\nゲート（3条件すべて）")
    passed = {
        "overall": check("held-out 全体", results["overall"].get("auroc"), GATES["overall"]),
        "I": check("I 段（非明示の意図）",
                   results["by_tier"].get("I", {}).get("auroc"), GATES["I"]),
        "S": check("S 段（表層）",
                   results["by_tier"].get("S", {}).get("auroc"), GATES["S"]),
    }
    results["gates"] = {
        name: {"threshold": GATES[name], "passed": ok} for name, ok in passed.items()
    }
    results["all_passed"] = all(passed.values())

    print("\n属性ごと")
    print(f"  {'attribute':34s} {'tier':5s} {'n':>5s} {'AUROC':>7s} {'acc':>7s} {'P(true)':>8s}")
    for name, entry in results["by_attribute"].items():
        auroc_text = "    n/a" if entry["auroc"] is None else f"{entry['auroc']:7.4f}"
        print(f"  {name:34s} {entry['tier']:5s} {entry['n']:5d} {auroc_text} "
              f"{entry['accuracy']:7.4f} {entry['mean_p_true']:8.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")

    if not results["all_passed"]:
        failed = [name for name, ok in passed.items() if not ok]
        if failed == ["I"]:
            print("\nI 段のみ未達。表層は転移したが意図は転移していない。"
                  "3 シードには進まず、ここで報告する。")
        else:
            print(f"\n未達: {failed}")
        return 2

    print("\n3 条件通過。3 シードへ進んでよい。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
