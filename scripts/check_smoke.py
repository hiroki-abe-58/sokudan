"""Judge a smoke-test corpus against the four Day 3 pass criteria.

    uv run python scripts/check_smoke.py data/smoke_v4.manifest.json

The criteria are about the *catalogue*, not the model: they ask whether 36 newly
written attribute definitions can actually be carried out by the generator and read
back by the verifier. A definition that fails here would otherwise reach training as
label noise, and no amount of data fixes that.

Criterion 4 is the one the 30% discard ceiling cannot see. A tier-I attribute set to
false whose text the verifier nevertheless reads as true is an *implication* failure --
some other attribute on the document entails it -- and the fix is an exclusion pair,
not a reworded definition. Lumping it in with plain instruction misses would send the
wrong repair.

Small-sample honesty: per-attribute counts are printed with their n, and an attribute
whose n is below --min-n is reported as UNDECIDED rather than passed. With 200
documents and MAX_IMPLICIT_PER_DOC = 1, the tier-I attributes get single-digit samples
each, so the tier-level numbers decide and the per-attribute column only flags gross
breakage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DISCARD_CEILING = 0.30
IMPLICATION_CEILING = 0.20
CORRELATION_CEILING = 0.30
BALANCE_BAND = (0.40, 0.60)


def judge(manifest: dict[str, Any], min_n: int) -> tuple[bool, list[str]]:
    v = manifest["verification"]
    rows = v["per_attribute"]
    tiers = v["by_tier"]
    notes: list[str] = []
    failed = False

    print(f"{'attribute':34s}{'tier':5s}{'new':4s}{'n':>6s}{'discard':>9s}"
          f"{'f->t':>8s}{'t->f':>8s}{'kept+':>8s}")
    print("-" * 82)
    known = set(manifest.get("previously_validated") or [])
    for row in sorted(rows, key=lambda r: (r["tier"], -r["discard_rate"])):
        n = row["total"]
        flags = ""
        if n >= min_n and row["discard_rate"] >= DISCARD_CEILING:
            flags += " DISCARD"
        if (row["tier"] == "I" and n >= min_n
                and row["false_to_true_rate"] >= IMPLICATION_CEILING):
            flags += " IMPLIES"
        if n < min_n:
            flags += " (n low)"
        print(f"{row['attribute']:34s}{row['tier']:5s}"
              f"{'' if row['attribute'] in known else '*':4s}{n:6d}"
              f"{row['discard_rate']:9.3f}{row['false_to_true_rate']:8.3f}"
              f"{row['true_to_false_rate']:8.3f}{row['kept_positive_rate']:8.3f}{flags}")

    # 1. Discard ceiling, decided per tier and flagged per attribute.
    print("\n[1] 検算不一致による破棄率 (< 0.30)")
    for tier, entry in sorted(tiers.items()):
        ok = entry["discard_rate"] < DISCARD_CEILING
        failed |= not ok
        print(f"    {tier}: {entry['discard_rate']:.3f} (n={entry['total']}) "
              f"{'OK' if ok else 'FAIL'}")
    over = [r["attribute"] for r in rows
            if r["total"] >= min_n and r["discard_rate"] >= DISCARD_CEILING]
    if over:
        notes.append(f"破棄率 30% 超（n>={min_n}）: {', '.join(over)}")
    undecided = [r["attribute"] for r in rows if r["total"] < min_n]
    if undecided:
        notes.append(f"n<{min_n} で判定保留: {len(undecided)} 属性")

    # 2. Can the rebalance reach 40-60% on what survived?
    print(f"\n[2] 破棄後の再バランスが {BALANCE_BAND[0]:.0%}〜{BALANCE_BAND[1]:.0%} に入るか")
    unbalanceable, low = [], []
    for row in rows:
        kept, rate = row["kept_total"], row["kept_positive_rate"]
        positives = round(kept * rate)
        negatives = kept - positives
        # The rebalance downsamples the majority side, so it lands inside the band
        # whenever both sides survive at all; what it cannot fix is an empty side.
        if kept and min(positives, negatives) == 0:
            # An empty side on a handful of draws says nothing about the definition --
            # at n=4 one side is empty 12% of the time from a fair coin alone. Only
            # attributes with enough draws to distinguish the two can fail here.
            (unbalanceable if row["total"] >= min_n else low).append(
                (row["attribute"], positives, negatives, row["total"]))
    for attribute, p, n, total in low:
        print(f"    — {attribute}: 残り +{p} / -{n}（n={total} で判定保留）")
    if unbalanceable:
        failed = True
        for attribute, p, n, total in unbalanceable:
            print(f"    FAIL {attribute}: 残り +{p} / -{n}（n={total}、片側が空）")
    else:
        decided = [r for r in rows if r["total"] >= min_n] or rows
        worst = min(decided, key=lambda r: min(r["kept_positive_rate"],
                                               1 - r["kept_positive_rate"]))
        print(f"    OK n>={min_n} の全属性で両側が残存（最も偏るのは {worst['attribute']} の "
              f"{worst['kept_positive_rate']:.3f}、再バランスで 0.50 に戻る）")

    # 3. Length vs true-count correlation.
    print(f"\n[3] 文長 × true 個数の相関 (|r| < {CORRELATION_CEILING})")
    for split, entry in sorted(v["length_vs_true_count"].items()):
        r = entry.get("r")
        if r is None:
            print(f"    {split}: n/a")
            continue
        n = entry.get("n") or 0
        # Fisher's z standard error, so a split that lands near the ceiling is not
        # read as a verdict when it is a sample size.
        se = 1 / max(n - 3, 1) ** 0.5
        ok = abs(r) < CORRELATION_CEILING
        # Only the whole-corpus figure decides; the splits are printed for comparison.
        if split == "all":
            failed |= not ok
        print(f"    {split}: r={r:+.4f} ±{se:.3f} (n={n}) "
              f"{'OK' if ok else 'FAIL'}{'' if split == 'all' else '（参考）'}")

    # 4. Implication violations, tier I.
    print(f"\n[4] 含意違反（false を検算が true と読む）I 段 < {IMPLICATION_CEILING}")
    entry = tiers.get("I")
    if entry:
        rate = entry["false_to_true_rate"]
        ok = rate < IMPLICATION_CEILING
        failed |= not ok
        print(f"    I: {rate:.3f} {'OK' if ok else 'FAIL'}")
    worst = sorted((r for r in rows if r["tier"] == "I"),
                   key=lambda r: -r["false_to_true_rate"])[:5]
    for row in worst:
        print(f"      {row['attribute']:34s} {row['false_to_true_rate']:.3f} "
              f"(n_false={row['total'] - round(row['positive_rate'] * row['total'])})")

    return not failed, notes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--min-n", type=int, default=20,
                        help="below this, an attribute is undecided rather than passed")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    passed, notes = judge(manifest, args.min_n)

    print("\n" + "=" * 60)
    print("PASS" if passed else "FAIL")
    for note in notes:
        print(f"  note: {note}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
