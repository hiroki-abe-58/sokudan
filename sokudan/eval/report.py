"""Turn baseline outputs into tables and a reliability diagram (§4.2, §9)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from sokudan.calibration.metrics import (
    binary_to_probs,
    reliability_bins,
    summarize_choice,
    summarize_ordinal,
)
from sokudan.eval.baselines import BaselineOutput, floor_and_renormalise
from sokudan.eval.bench_ja import DEPARTMENTS, BenchItem


def gold_arrays(items: list[BenchItem]) -> dict[str, np.ndarray]:
    dept_keys = list(DEPARTMENTS)
    return {
        "choice": np.array([dept_keys.index(i.department) for i in items]),
        "score": np.array([i.urgency for i in items]),
        "bool": np.array([int(i.churn) for i in items]),
    }


def score_baseline(output: BaselineOutput, gold: dict[str, np.ndarray]) -> dict[str, Any]:
    """Apply the identical probability floor to every baseline, then score."""
    choice = floor_and_renormalise(output.choice_probs)
    score = floor_and_renormalise(output.score_probs)
    boolean = floor_and_renormalise(binary_to_probs(output.bool_p_true))

    latencies = np.array(output.per_item_latency_s) if output.per_item_latency_s else None
    latency: dict[str, float] | None = None
    if latencies is not None and latencies.size:
        latency = {
            "p50_ms": float(np.percentile(latencies, 50) * 1000),
            "p95_ms": float(np.percentile(latencies, 95) * 1000),
            "mean_ms": float(latencies.mean() * 1000),
            "questions_per_sec": float(3 * len(latencies) / latencies.sum()),
        }

    return {
        "name": output.name,
        "choice": summarize_choice(choice, gold["choice"]),
        "score": summarize_ordinal(score, gold["score"]),
        "bool": summarize_choice(boolean, gold["bool"]),
        "latency": latency,
        "parse_failures": output.parse_failures,
        "parse_attempts": output.parse_attempts,
        "parse_failure_rate": output.parse_failure_rate,
        "notes": output.notes,
    }


def markdown_accuracy_table(rows: list[dict[str, Any]]) -> str:
    head = (
        "| 対象 | choice acc | choice ECE | choice Brier | score RPS | score MAE(argmax) "
        "| score acc | bool acc | bool ECE | パース失敗率 |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for r in rows:
        rate = r["parse_failure_rate"]
        rate_text = "—" if rate is None else f"{rate:.1%}"
        lines.append(
            f"| {r['name']} | {r['choice']['accuracy']:.3f} | {r['choice']['ece']:.3f} "
            f"| {r['choice']['brier']:.3f} | {r['score']['rps']:.3f} "
            f"| {r['score']['mae_argmax']:.3f} | {r['score']['accuracy']:.3f} "
            f"| {r['bool']['accuracy']:.3f} | {r['bool']['ece']:.3f} | {rate_text} |"
        )
    return head + "\n".join(lines)


def markdown_latency_table(rows: list[dict[str, Any]]) -> str:
    head = ("| 対象 | p50 (ms/件) | p95 (ms/件) | mean (ms/件) | questions/sec |\n"
            "|---|---|---|---|---|\n")
    lines = []
    for r in rows:
        lat = r["latency"]
        if lat is None:
            lines.append(f"| {r['name']} | — | — | — | — |")
            continue
        lines.append(
            f"| {r['name']} | {lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} "
            f"| {lat['mean_ms']:.1f} | {lat['questions_per_sec']:.1f} |"
        )
    return head + "\n".join(lines)


def save_reliability_diagram(
    per_baseline: dict[str, np.ndarray],
    gold: np.ndarray,
    out_path: Path,
    *,
    title: str,
    n_bins: int = 10,
    min_count: int = 5,
) -> Path:
    """Reliability diagram: mean confidence vs mean accuracy, per bin.

    Bins holding fewer than `min_count` items are dropped from the curves. A bin with
    one item reads as 0.0 or 1.0 accuracy and draws a dramatic line that means nothing;
    the counts panel below still shows every bin, so nothing is hidden.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Labels are Japanese; without a CJK font matplotlib silently renders tofu.
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Meiryo", "Yu Gothic", "BIZ UDGothic", "MS Gothic", "Noto Sans CJK JP"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax, ax_hist) = plt.subplots(
        2, 1, figsize=(7.0, 7.4), height_ratios=[3, 1], sharex=True
    )
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="#888888", label="完全較正")

    for name, probs in per_baseline.items():
        bins = reliability_bins(probs, gold, n_bins=n_bins)
        mask = bins.counts >= min_count
        ax.plot(
            bins.mean_confidence[mask],
            bins.mean_accuracy[mask],
            marker="o",
            markersize=4,
            linewidth=1.4,
            label=name,
        )
        centres = (bins.edges[:-1] + bins.edges[1:]) / 2
        shown = bins.counts > 0
        ax_hist.plot(centres[shown], bins.counts[shown], marker=".", linewidth=1.0, label=name)

    ax.set_ylabel("実測正答率")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.text(0.98, 0.02, f"{min_count}件未満のビンは非表示", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#666666")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax_hist.set_xlabel("予測確信度 (top-1)")
    ax_hist.set_ylabel("件数")
    ax_hist.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
