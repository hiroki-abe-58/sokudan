"""Collect the ablation runs into one table.

    uv run python scripts/report_ablations.py

Every run is scored the same way: held-out AUROC overall and per tier (unseen schema
on an unseen document), plus the validation figures for the three primitives. The
baseline is `runs/v01_seed0`, which is the published configuration, so each row
answers "what happens if I change this one thing".

`bench_ja` is not touched. It was measured once, and an ablation sweep is exactly the
kind of thing that would turn it into a validation set by repeated use.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

RUNS: list[tuple[str, str, str]] = [
    ("baseline (epoch 2)", "runs/v01_seed0", "published configuration"),
    ("(a) epoch 1", "runs/abl_ep1", "half the training"),
    ("(a) epoch 3", "runs/abl_ep3", "half again more"),
    ("(b) no schema randomisation", "runs/abl_noaug",
     "option order, surface forms and paraphrases fixed"),
    ("(c) derived booleans 1:2", "runs/abl_derived", "derived restored, intent cut to match"),
    ("(d) no ordinal head", "runs/abl_noord", "score through the plain softmax"),
]


def gate(checkpoint: Path, heldout: Path, out: Path) -> dict[str, Any]:
    subprocess.run(
        [sys.executable, "-m", "scripts.eval_heldout", "--checkpoint", str(checkpoint),
         "--heldout", str(heldout), "--out", str(out)],
        check=False, capture_output=True, text=True,
    )
    return json.loads(out.read_text(encoding="utf-8"))


def fmt(value: float | None, width: int = 6) -> str:
    return " " * (width - 3) + "—" if value is None else f"{value:{width}.3f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", default="data/heldout_val_v2.jsonl")
    parser.add_argument("--scratch", default="runs/abl_reports")
    parser.add_argument("--out", default="docs/ablations.md")
    args = parser.parse_args()

    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for label, run_dir, note in RUNS:
        directory = Path(run_dir)
        checkpoint = directory / "model.pt"
        metrics_path = directory / "metrics.json"
        if not checkpoint.exists() or not metrics_path.exists():
            print(f"skip {label}: {run_dir} incomplete")
            rows.append({"label": label, "note": note, "missing": True})
            continue

        print(f"scoring {label} ...", flush=True)
        result = gate(checkpoint, Path(args.heldout), scratch / f"{directory.name}.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        after = metrics["after"]
        seconds = sum(e["seconds"] for e in metrics["epochs"])
        rows.append({
            "label": label, "note": note, "missing": False,
            "run": run_dir,
            "epochs": metrics["config"]["epochs"],
            "views": metrics["n_train"],
            "seconds": round(seconds),
            "heldout_all": result["overall"].get("auroc"),
            "heldout_I": result["by_tier"].get("I", {}).get("auroc"),
            "heldout_E": result["by_tier"].get("E", {}).get("auroc"),
            "heldout_S": result["by_tier"].get("S", {}).get("auroc"),
            "val_choice_acc": after["choice"]["accuracy"],
            "val_score_rps": after["score"]["rps"],
            "val_score_acc": after["score"]["accuracy"],
            "val_bool_acc": after["bool"]["accuracy"],
            "val_bool_auroc": after["bool"].get("auroc"),
        })

    header = (f"{'ablation':32s} {'views':>7s} {'ep':>3s} {'sec':>5s} "
              f"{'held.all':>8s} {'held.I':>7s} {'held.E':>7s} {'held.S':>7s} "
              f"{'choice':>7s} {'RPS↓':>7s} {'bool':>7s} {'boolAUC':>8s}")
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        if row["missing"]:
            print(f"{row['label']:32s} {'(未完了)':>7s}")
            continue
        print(f"{row['label']:32s} {row['views']:7d} {row['epochs']:3d} {row['seconds']:5d} "
              f"{fmt(row['heldout_all'], 8)} {fmt(row['heldout_I'], 7)} "
              f"{fmt(row['heldout_E'], 7)} {fmt(row['heldout_S'], 7)} "
              f"{fmt(row['val_choice_acc'], 7)} {fmt(row['val_score_rps'], 7)} "
              f"{fmt(row['val_bool_acc'], 7)} {fmt(row['val_bool_auroc'], 8)}")

    done = [r for r in rows if not r["missing"]]
    lines = [
        "# アブレーション（v0.1、joint、各 1 シード）",
        "",
        "> **すべて held-out val（未知スキーマ × 未知文書）で評価しています。**",
        "> `bench_ja` には一切触れていません——アブレーションの掃引で繰り返し当てれば、",
        "> それは held-out テストセットではなく検証セットになってしまうためです。",
        "> ベースラインは公開している構成（`runs/v01_seed0`、seed 0、epoch 2）です。",
        "",
        "| アブレーション | ビュー | ep | 秒 | held-out 全体 | I 段 | E 段 | S 段 "
        "| val choice | val RPS↓ | val bool | val bool AUROC |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in done:
        lines.append(
            f"| {row['label']} | {row['views']:,} | {row['epochs']} | {row['seconds']} "
            f"| {fmt(row['heldout_all']).strip()} | {fmt(row['heldout_I']).strip()} "
            f"| {fmt(row['heldout_E']).strip()} | {fmt(row['heldout_S']).strip()} "
            f"| {fmt(row['val_choice_acc']).strip()} | {fmt(row['val_score_rps']).strip()} "
            f"| {fmt(row['val_bool_acc']).strip()} | {fmt(row['val_bool_auroc']).strip()} |"
        )
    lines += ["", "各行の意味:", ""]
    lines += [f"- **{r['label']}** — {r['note']}" for r in rows]
    lines += ["", f"生データ: `{args.scratch}/*.json`、`runs/abl_*/metrics.json`", ""]

    out = Path(args.out)
    out.write_text("\n".join(lines), encoding="utf-8")
    (Path(args.scratch) / "summary.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
