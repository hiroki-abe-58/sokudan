"""Two diagnostics that need no training: the `bool` threshold, and K.

    uv run python scripts/diagnose_bias_and_k.py --checkpoint runs/v01_seed0/model.pt

**§4 -- the boolean threshold.** `bench_ja` shows mean P(true) at 0.125 against a gold
rate of 0.297, so the head ranks well (AUROC 0.789) and decides badly. A temperature
cannot fix that: dividing a logit gap by a constant leaves the argmax where it was.
A *bias* can. One scalar is fitted on validation -- added to the true-side logit --
and the held-out numbers are recomputed with it.

This is a diagnostic, not a proposal. Fitting it on validation and reporting it on
held-out keeps it honest, but the fix belongs in whatever prior the caller actually
has, which is why the README tells them to choose a threshold rather than shipping
one.

**§5 -- the K=4 drop.** `bench_ja`'s four-level condition scores 0.427 against
0.697-0.788 for the three-level ones. Two explanations fit: the model is bad at K=4
specifically, or it degrades as K grows. Held-out `score` split by K separates them.
The training K distribution is printed alongside, because "we barely trained on K=4"
would be a third explanation and it needs ruling out.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sokudan.config  # noqa: F401
from scripts.eval_heldout import load_checkpoint, load_rows
from sokudan.calibration.metrics import auroc, ece, rps
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, length_bucketed_batches, run_model


@torch.no_grad()
def collect(model: Any, rows: list[dict], collator: Any, config: TrainConfig,
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Return `(p_true, gold, probs_padded, n_options)` over the rows given."""
    import random as _random

    examples = [Example.from_row(r) for r in rows]
    order: list[int] = []
    p_true, gold, widths = [], [], []
    all_probs: list[np.ndarray] = []
    index = {id(e): i for i, e in enumerate(examples)}

    device = torch.device(config.device)
    batches = length_bucketed_batches(
        examples, config.batch_size, collator.tokenizer,
        rng=_random.Random(0), shuffle=False,
    )
    model.eval()
    for group in batches:
        batch = collator(group).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            out = run_model(model, batch)
        probs = out.probs.float().cpu().numpy()
        labels = batch.labels.cpu().numpy()
        for row, example in enumerate(group):
            k = int(batch.marker_mask[row].sum())
            all_probs.append(probs[row, :k])
            widths.append(k)
            gold.append(int(labels[row]))
            p_true.append(float(probs[row, 1]) if k == 2 else float("nan"))
            order.append(index[id(example)])
    return (np.array(p_true), np.array(gold),
            np.array(all_probs, dtype=object), widths)


def fit_bias(p_true: np.ndarray, gold: np.ndarray) -> float:
    """The scalar logit shift that makes mean P(true) match the gold rate.

    Solved by bisection on the realised mean rather than by maximising likelihood:
    the stated problem is that the *rate* is wrong, so the fitted quantity should be
    the one that fixes the rate.
    """
    eps = 1e-6
    logit = np.log(np.clip(p_true, eps, 1 - eps) / np.clip(1 - p_true, eps, 1 - eps))
    target = float(gold.mean())
    low, high = -10.0, 10.0
    for _ in range(80):
        mid = (low + high) / 2
        if float((1 / (1 + np.exp(-(logit + mid)))).mean()) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def apply_bias(p_true: np.ndarray, bias: float) -> np.ndarray:
    eps = 1e-6
    logit = np.log(np.clip(p_true, eps, 1 - eps) / np.clip(1 - p_true, eps, 1 - eps))
    return 1 / (1 + np.exp(-(logit + bias)))


def bool_metrics(p_true: np.ndarray, gold: np.ndarray) -> dict[str, float]:
    probs = np.column_stack([1 - p_true, p_true])
    entry = {
        "accuracy": float((probs.argmax(1) == gold).mean()),
        "ece": float(ece(probs, gold)),
        "mean_p_true": float(p_true.mean()),
        "gold_true_rate": float(gold.mean()),
    }
    if 0 < gold.sum() < len(gold):
        entry["auroc"] = float(auroc(p_true, gold))
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/v01_seed0/model.pt")
    parser.add_argument("--val", default="data/val_v2.jsonl")
    parser.add_argument("--heldout", default="data/heldout_val_v2.jsonl")
    parser.add_argument("--train", default="data/train_v2b.jsonl")
    parser.add_argument("--out", default="runs/diagnostics/bias_and_k.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    model, encoding, device = load_checkpoint(args.checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    config = TrainConfig(device=device, batch_size=args.batch_size, encoding=encoding)
    collator = build_collator(tokenizer, config)

    # ---------------- §4: the boolean threshold ----------------
    val_bool = [r for r in load_rows(Path(args.val)) if r["kind"] == "bool"]
    held = load_rows(Path(args.heldout))
    held_bool = [r for r in held if r["kind"] == "bool"]

    vp, vg, _, _ = collect(model, val_bool, collator, config)
    hp, hg, _, _ = collect(model, held_bool, collator, config)
    bias = fit_bias(vp, vg)

    before = bool_metrics(hp, hg)
    after = bool_metrics(apply_bias(hp, bias), hg)

    print("== §4 bool の閾値（bias を val でフィット、held-out で評価）==")
    print(f"  fitted bias (logit shift): {bias:+.4f}   (val n={len(vp)})")
    print(f"  {'':10s} {'acc':>8s} {'ECE':>8s} {'AUROC':>8s} {'P(true)':>9s} {'gold':>8s}")
    for label, entry in (("before", before), ("after", after)):
        print(f"  {label:10s} {entry['accuracy']:8.4f} {entry['ece']:8.4f} "
              f"{entry.get('auroc', float('nan')):8.4f} {entry['mean_p_true']:9.4f} "
              f"{entry['gold_true_rate']:8.4f}")
    auroc_moved = abs(after.get("auroc", 0) - before.get("auroc", 0))
    print(f"  AUROC の変化: {auroc_moved:.6f}"
          f"  {'(不変。期待どおり)' if auroc_moved < 1e-6 else '(動いた — 要調査)'}")

    # ---------------- §5: K ----------------
    train_k = Counter(
        r["n_options"] for r in load_rows(Path(args.train)) if r["kind"] == "score"
    )
    held_score = [r for r in load_rows(Path(args.val)) if r["kind"] == "score"]
    _, sg, sprobs, widths = collect(model, held_score, collator, config)

    by_k: dict[int, dict[str, list]] = defaultdict(lambda: {"probs": [], "gold": []})
    for probs, gold_index, k in zip(sprobs, sg, widths, strict=True):
        by_k[k]["probs"].append(np.asarray(probs, dtype=float))
        by_k[k]["gold"].append(int(gold_index))

    print("\n== §5 score を K 別に分解（val、未知文書）==")
    print(f"  {'K':>3s} {'学習ビュー':>9s} {'n':>6s} {'acc':>8s} {'RPS↓':>8s}")
    k_rows = []
    for k in sorted(by_k):
        probs = np.vstack(by_k[k]["probs"])
        gold = np.array(by_k[k]["gold"])
        entry = {
            "k": k,
            "train_views": int(train_k.get(k, 0)),
            "n": int(len(gold)),
            "accuracy": float((probs.argmax(1) == gold).mean()),
            "rps": float(rps(probs, gold)),
        }
        k_rows.append(entry)
        print(f"  {k:3d} {entry['train_views']:9d} {entry['n']:6d} "
              f"{entry['accuracy']:8.4f} {entry['rps']:8.4f}")

    accs = [r["accuracy"] for r in k_rows]
    ks = [r["k"] for r in k_rows]
    slope = float(np.polyfit(ks, accs, 1)[0]) if len(ks) > 1 else 0.0
    k4 = next((r for r in k_rows if r["k"] == 4), None)
    others = [r["accuracy"] for r in k_rows if r["k"] != 4]
    verdict = (
        f"K=4 単独の落ち込みではなく、K が増えるほど下がる傾向（傾き {slope:+.4f}/K）"
        if k4 and k4["accuracy"] >= min(others) else
        f"K=4 が単独で落ちている（K=4 {k4['accuracy']:.3f} 対 他の最小 {min(others):.3f}）"
        if k4 else "K=4 が val に存在しない"
    )
    print(f"  結論: {verdict}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint,
        "bool_bias": {
            "fitted_bias": bias, "val_n": int(len(vp)), "heldout_n": int(len(hp)),
            "before": before, "after": after, "auroc_delta": auroc_moved,
        },
        "score_by_k": k_rows,
        "train_score_views_by_k": {str(k): v for k, v in sorted(train_k.items())},
        "k_slope_per_level": slope,
        "k_verdict": verdict,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
