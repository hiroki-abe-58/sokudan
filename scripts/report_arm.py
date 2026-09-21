"""One report for one training arm, so s2b and s2c are read the same way.

    uv run python scripts/report_arm.py --run runs/s2b

Prints four things, in the order they answer questions:

1. **The gate.** Held-out AUROC overall and per tier against the three thresholds.
2. **Trained attributes on unseen documents.** Per tier and per attribute. This is
   what separates "learned nothing" from "learned, but does not transfer to an unseen
   question" -- s2a turned out to be the first, which is not what Day 1 was.
3. **`choice` and `score` against s1.** The maintenance gate: `score` RPS must not
   regress past s1's 0.168 on `bench_ja`, and validation is the early warning.
4. **Diagnostic 1.** Held-out AUROC with the state emptied and with it swapped for
   another document. Day 1's `bool` scored *higher* with no state at all; if that
   comes back, the number above it means nothing.

Everything is read from files the run already wrote, except the ablations, which are
forward passes over the held-out split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

S1_VAL = {"choice_acc": 0.792, "score_rps": 0.149, "score_acc": 0.487,
          "bool_acc": 0.820, "bool_auroc": 0.907}


def run_gate(checkpoint: Path, heldout: Path, out: Path, state_mode: str) -> dict[str, Any]:
    subprocess.run(
        [sys.executable, "-m", "scripts.eval_heldout",
         "--checkpoint", str(checkpoint), "--heldout", str(heldout),
         "--out", str(out), "--state-mode", state_mode],
        check=False, capture_output=True, text=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def trained_attribute_subset(val_path: Path, scratch: Path) -> Path:
    """Intent attributes that *were* trained, on documents that were not."""
    rows = [
        json.loads(line)
        for line in val_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    keep = [r for r in rows if r["kind"] == "bool" and r["source"] == "intent"]
    scratch.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep), encoding="utf-8"
    )
    return scratch


def fmt(value: float | None, width: int = 7) -> str:
    return " " * (width - 3) + "n/a" if value is None else f"{value:{width}.4f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--heldout", default="data/heldout_val_v2.jsonl")
    parser.add_argument("--val", default="data/val_v2.jsonl")
    parser.add_argument("--scratch", default="runs/report_tmp")
    args = parser.parse_args()

    run_dir = Path(args.run)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / "model.pt"

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    after, before = metrics["after"], metrics["before"]
    config = metrics["config"]

    print(f"== {run_dir.name} ==")
    print(f"encoding {config.get('encoding', 'separate')}  epochs {config.get('epochs')}  "
          f"batch {config.get('batch_size')}  head_lr {config.get('head_learning_rate')}  "
          f"views {metrics['n_train']}")

    gate = run_gate(checkpoint, Path(args.heldout), scratch / "gate.json", "real")
    print("\n-- ゲート（held-out、未知スキーマ × 未知文書）--")
    for name, key, threshold in (("held-out 全体", None, 0.75),
                                 ("I 段", "I", 0.70), ("S 段", "S", 0.85)):
        entry = gate["overall"] if key is None else gate["by_tier"].get(key, {})
        value = entry.get("auroc")
        mark = "PASS" if value is not None and value > threshold else "FAIL"
        print(f"  {name:14s} {fmt(value)} > {threshold:.2f}  {mark}  (n={entry.get('n', 0)})")

    subset = trained_attribute_subset(Path(args.val), scratch / "trained_intent.jsonl")
    trained = run_gate(checkpoint, subset, scratch / "trained.json", "real")
    print("\n-- 学習済み属性 × 未知文書 --")
    for tier in ("S", "E", "I"):
        entry = trained["by_tier"].get(tier)
        if entry:
            print(f"  {tier} 段  {fmt(entry.get('auroc'))}  acc {fmt(entry.get('accuracy'))}  "
                  f"n={entry['n']}")
    ranked = sorted(
        trained["by_attribute"].items(),
        key=lambda kv: kv[1].get("auroc") or 0.0, reverse=True,
    )
    print(f"  {'attribute':32s} {'tier':5s} {'n':>5s} {'AUROC':>8s} {'acc':>8s}")
    for name, entry in ranked:
        print(f"  {name:32s} {entry['tier']:5s} {entry['n']:5d} "
              f"{fmt(entry.get('auroc'), 8)} {fmt(entry.get('accuracy'), 8)}")

    print("\n-- val（s1 と並べる）--")
    print(f"  {'':12s} {'before':>8s} {'after':>8s} {'s1':>8s}")
    rows = [
        ("choice acc", before["choice"]["accuracy"], after["choice"]["accuracy"],
         S1_VAL["choice_acc"]),
        ("score RPS", before["score"]["rps"], after["score"]["rps"], S1_VAL["score_rps"]),
        ("score acc", before["score"]["accuracy"], after["score"]["accuracy"],
         S1_VAL["score_acc"]),
        ("bool acc", before["bool"]["accuracy"], after["bool"]["accuracy"],
         S1_VAL["bool_acc"]),
        ("bool AUROC", before["bool"].get("auroc"), after["bool"].get("auroc"),
         S1_VAL["bool_auroc"]),
    ]
    for label, first, last, reference in rows:
        print(f"  {label:12s} {fmt(first, 8)} {fmt(last, 8)} {reference:8.3f}")
    print("  ※ s1 の val は派生 bool 中心の別セット。choice / score は比較可、bool は不可")

    print("\n-- 診断1: state 差し替え（held-out 全体 AUROC）--")
    ablation = {"real": gate["overall"].get("auroc")}
    for mode in ("empty", "shuffled"):
        result = run_gate(checkpoint, Path(args.heldout), scratch / f"{mode}.json", mode)
        ablation[mode] = result["overall"].get("auroc")
    for mode in ("real", "empty", "shuffled"):
        print(f"  {mode:9s} {fmt(ablation[mode])}")
    if ablation["empty"] is not None and ablation["real"] is not None:
        if ablation["empty"] >= ablation["real"]:
            print("  ⚠ state を空にしても落ちない。Day 1 と同じ症状")

    summary = {"run": run_dir.name, "config": config, "gate": gate["gates"],
               "gate_auroc": {"overall": gate["overall"].get("auroc"),
                              **{t: v.get("auroc") for t, v in gate["by_tier"].items()}},
               "trained_by_tier": {t: v.get("auroc") for t, v in trained["by_tier"].items()},
               "trained_by_attribute": {
                   k: v.get("auroc") for k, v in trained["by_attribute"].items()
               },
               "val_after": {k: after[k] for k in ("choice", "score", "bool")},
               "state_ablation": ablation}
    out = run_dir / "arm_report.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
