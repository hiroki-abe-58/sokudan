"""Pre-training sanity checks on the synthetic corpus.

    uv run python scripts/check_data_health.py

Three things that are cheap to check now and expensive to discover after training:

1. **State length.** If the synthetic documents and `bench_ja` sit at different
   lengths, the model trains on one distribution and is tested on another, and any
   generalisation gap would be partly a length artefact rather than a schema one.
2. **Boolean skew direction.** `docs/baseline_ja.md` §6.3 measured a model leaning to
   one side regardless of the text. If every bool attribute in the catalogue leaned
   the same way, training would install that habit instead of curing it.
3. **`score` level coverage.** The dynamic-K head (§6.3) is the differentiator, so
   every K from 2 to 7 needs at least one attribute behind it.

Exits non-zero when a check fails, so it can gate the training run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.data.builders.synthetic import CATALOG

MEDIAN_RATIO_LIMIT = 2.0
REQUIRED_SCORE_KS = set(range(2, 8))


def token_lengths(texts: list[str], tokenizer: Any) -> np.ndarray:
    from sokudan.encoding.state import encode_state

    return np.array([encode_state(t, tokenizer).n_tokens for t in texts])


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "n": int(values.size),
        "min": int(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
        "mean": float(values.mean()),
    }


def plot_histograms(
    synthetic: np.ndarray, bench: np.ndarray, out_path: Path
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Meiryo", "Yu Gothic", "BIZ UDGothic", "MS Gothic"):
        if candidate in available:
            plt.rcParams["font.family"] = candidate
            break
    plt.rcParams["axes.unicode_minus"] = False

    upper = float(np.percentile(np.concatenate([synthetic, bench]), 99))
    bins = np.linspace(0, upper, 45)

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.hist(synthetic, bins=bins, alpha=0.55, density=True,
            label=f"合成（学習）n={synthetic.size}")
    ax.hist(bench, bins=bins, alpha=0.55, density=True,
            label=f"bench_ja（評価）n={bench.size}")
    ax.axvline(np.median(synthetic), linestyle="--", linewidth=1.2,
               label=f"合成の中央値 {np.median(synthetic):.0f}")
    ax.axvline(np.median(bench), linestyle=":", linewidth=1.4,
               label=f"bench_ja の中央値 {np.median(bench):.0f}")
    ax.set_xlabel("state のトークン数")
    ax.set_ylabel("密度")
    ax.set_title("state 長の分布: 合成学習データ vs bench_ja")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def check_bool_skew() -> tuple[bool, list[dict[str, Any]], dict[str, int]]:
    """Every bool attribute's prior, and which way it leans."""
    rows = []
    directions: Counter = Counter()
    for domain in CATALOG:
        for attribute in domain.attributes:
            if attribute.kind != "bool":
                continue
            p_true = attribute.weights[1] / sum(attribute.weights)
            direction = ("true寄り" if p_true > 0.55 else
                         "false寄り" if p_true < 0.45 else "ほぼ均衡")
            directions[direction] += 1
            rows.append({
                "domain": domain.name,
                "attribute": attribute.name,
                "p_true": round(p_true, 3),
                "direction": direction,
            })
    leaning = {k: v for k, v in directions.items() if k != "ほぼ均衡"}
    mixed = len(leaning) >= 2 or (len(rows) and directions["ほぼ均衡"] == len(rows))
    return mixed, rows, dict(directions)


def check_score_coverage() -> tuple[bool, dict[int, list[str]]]:
    by_k: dict[int, list[str]] = defaultdict(list)
    for domain in CATALOG:
        for attribute in domain.attributes:
            if attribute.kind == "score":
                by_k[len(attribute.labels)].append(f"{domain.name}.{attribute.name}")
    return REQUIRED_SCORE_KS.issubset(by_k.keys()), dict(sorted(by_k.items()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--out", default="docs/data/data_health.json")
    parser.add_argument("--figure", default="docs/img/state_length.png")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    failures: list[str] = []
    report: dict[str, Any] = {}

    # ---------------------------------------------------------------- 3. score K
    ok_k, by_k = check_score_coverage()
    print("== score の K 被覆（K=2..7 に最低1属性）==")
    for k in sorted(set(by_k) | REQUIRED_SCORE_KS):
        holders = by_k.get(k, [])
        mark = "OK " if holders else "MISSING"
        print(f"  K={k}: {mark} {len(holders)} 属性  {holders}")
    if not ok_k:
        failures.append(f"score の K が欠けている: {sorted(REQUIRED_SCORE_KS - set(by_k))}")
    report["score_k_coverage"] = {str(k): v for k, v in by_k.items()}

    # ------------------------------------------------------------- 2. bool skew
    mixed, bool_rows, directions = check_bool_skew()
    print("\n== bool のラベル偏りの向き ==")
    for row in sorted(bool_rows, key=lambda r: r["p_true"]):
        print(f"  {row['domain']:<20} {row['attribute']:<22} "
              f"P(true)={row['p_true']:.3f}  {row['direction']}")
    print(f"  内訳: {directions}")
    if not mixed:
        failures.append(f"bool の偏りが一方向に揃っている: {directions}")
    report["bool_skew"] = {"attributes": bool_rows, "directions": directions,
                           "mixed": mixed}

    # ----------------------------------------------------------- 1. state length
    train_path, bench_path = Path(args.train), Path(args.bench)
    if not train_path.exists():
        print(f"\n(skipped) {train_path} はまだありません。生成完了後に再実行してください。")
        report["state_length"] = None
    else:
        rows = [json.loads(line) for line in
                train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        # One state per document; the expansion repeats each state many times.
        seen: dict[str, str] = {}
        for row in rows:
            seen.setdefault(row["doc_id"], row["state"])
        synthetic_texts = list(seen.values())
        bench_texts = [
            json.loads(line)["state"]
            for line in bench_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]

        synthetic = token_lengths(synthetic_texts, tokenizer)
        bench = token_lengths(bench_texts, tokenizer)
        ratio = max(np.median(synthetic), np.median(bench)) / max(
            min(np.median(synthetic), np.median(bench)), 1e-9
        )

        print("\n== state 長（トークン数）==")
        print(f"{'':>12} {'n':>6} {'min':>5} {'p25':>7} {'median':>7} {'p75':>7} "
              f"{'p95':>7} {'max':>6} {'mean':>7}")
        for name, values in (("合成(学習)", synthetic), ("bench_ja", bench)):
            d = describe(values)
            print(f"{name:>12} {d['n']:>6} {d['min']:>5} {d['p25']:>7.0f} "
                  f"{d['median']:>7.0f} {d['p75']:>7.0f} {d['p95']:>7.0f} "
                  f"{d['max']:>6} {d['mean']:>7.1f}")
        print(f"  中央値の比: {ratio:.2f}x  (限度 {MEDIAN_RATIO_LIMIT}x)")

        figure = plot_histograms(synthetic, bench, Path(args.figure))
        print(f"  ヒストグラム -> {figure}")

        report["state_length"] = {
            "synthetic": describe(synthetic),
            "bench_ja": describe(bench),
            "median_ratio": float(ratio),
            "limit": MEDIAN_RATIO_LIMIT,
            "figure": str(figure),
        }
        if ratio >= MEDIAN_RATIO_LIMIT:
            failures.append(
                f"state 長の中央値が {ratio:.2f}x ずれている "
                f"(合成 {np.median(synthetic):.0f} vs bench_ja {np.median(bench):.0f})"
            )

    report["failures"] = failures
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nreport -> {out_path}")

    if failures:
        print("\n報告が必要な項目:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nすべて基準内。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
