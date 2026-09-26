"""Night 3 §2 table: can a second generator write the ten retired attributes?

    uv run python scripts/report_v5.py data/docs_v5.manifest.json

Reads the manifest `build_intent_corpus.py` wrote for the cross-model run and prints:

1. per-attribute discard rates, the ten retired attributes first, beside the same
   attribute's figures from the second smoke test (qwen3 wrote *and* verified);
2. document length against the number of true attributes;
3. tier-I implication violations (gold=false read as true), retired and active apart;
4. the positive rate before and after the length-stratified rebalance.

The comparison in (1) holds the verifier fixed. Smoke 2 and v5 were both verified by
qwen3 with the same prompts; only the generator differs. So a retired attribute whose
discard rate falls here fell because mistral wrote it differently -- not because a
second verifier was more lenient.

§3's rule reads the discard rate: an attribute passes below 30%.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.build_intent_train import rebalance
from sokudan.data import intent_attributes as ia

YES, UNKNOWN = "はい", "判断できない"
CEILING = 0.30


def load_rows(manifest: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return {r["attribute"]: r for r in data["verification"]["per_attribute"]}, data


def kept_pairs(corpus: Path) -> list[dict[str, Any]]:
    """Verifier-confirmed (document, attribute) pairs, in `partition()`'s record shape."""
    out = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        document = json.loads(line)
        verdicts = document.get("verdicts") or {}
        for attribute_id, gold in document["intent_labels"].items():
            answer = verdicts.get(attribute_id)
            if answer is None or answer == UNKNOWN or (answer == YES) != bool(gold):
                continue
            out.append({"attribute": attribute_id, "split": document["split"],
                        "label": gold, "state": document["state"]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", nargs="?", default="data/docs_v5.manifest.json")
    parser.add_argument("--baseline", default="data/smoke2_v4.manifest.json",
                        help="same-model run to compare the retired attributes against")
    parser.add_argument("--out", default="runs/diagnostics/v5_report.json")
    args = parser.parse_args()

    manifest = Path(args.manifest)
    rows, data = load_rows(manifest)
    base_rows, base = load_rows(Path(args.baseline))
    # All ten, re-admitted or not: RETIRED shrinks as attributes are put back.
    retired = [a.id for a in ia.RETIRED_DEFINITIONS]

    print(f"generator {data.get('generator_model')}  verifier {data.get('verifier_model')}  "
          f"cross-model={data.get('cross_model_verification')}")
    print(f"baseline  {base.get('generator_model')} (generated and verified by the same model)")
    print(f"documents kept {data['documents_kept']}/{data['documents_requested']}  "
          f"rejections {data.get('rejections')}\n")

    # 1. Retired attributes first.
    print(f"[1] 除外 10 属性（判定は false->true < {ia.READMIT_MAX_FALSE_TO_TRUE}、"
          f"旧基準の破棄率 < {CEILING} を括弧内に併記）")
    header = (f"{'attribute':32s}{'n':>5s}{'discard':>9s}{'f->t':>7s}{'t->f':>7s}"
              f"{'kept+':>7s}  |{'smoke2 n':>9s}{'discard':>9s}{'f->t':>7s}  {'判定':>6s}")
    print(header)
    print("-" * len(header))
    passed, table = [], []
    for name in retired:
        r, b = rows.get(name), base_rows.get(name)
        if r is None:
            print(f"{name:32s}  (not in this corpus)")
            continue
        ok = r["false_to_true_rate"] < ia.READMIT_MAX_FALSE_TO_TRUE
        ok_old = r["discard_rate"] < CEILING
        passed += [name] if ok else []
        table.append({"attribute": name, "v5": r, "smoke2": b, "passed": ok,
                      "passed_old_discard_rule": ok_old})
        baseline = (
            f"{b['total']:9d}{b['discard_rate']:9.3f}"
            f"{b.get('false_to_true_rate', float('nan')):7.3f}"
            if b else f"{'—':>9s}{'—':>9s}{'—':>7s}"
        )
        print(f"{name:32s}{r['total']:5d}{r['discard_rate']:9.3f}"
              f"{r['false_to_true_rate']:7.3f}{r['true_to_false_rate']:7.3f}"
              f"{r['kept_positive_rate']:7.3f}  |{baseline}  {'PASS' if ok else 'FAIL':>6s}"
              f"  (旧 {'PASS' if ok_old else 'FAIL'})")
    n_old = sum(t["passed_old_discard_rule"] for t in table)
    print(f"\n通過（false->true 基準）: {len(passed)} / {len(retired)}   "
          f"旧基準（破棄率）での通過: {n_old} / {len(retired)}")

    print("\n    active attributes (for reference)")
    for tier in ia.TIERS:
        names = sorted(n for n, r in rows.items() if r["tier"] == tier and n not in retired)
        worst = sorted(names, key=lambda n: -rows[n]["discard_rate"])[:4]
        print(f"    {tier}: " + ", ".join(f"{n} {rows[n]['discard_rate']:.3f}" for n in worst))

    # 2. Length correlation.
    print("\n[2] 文長 × true 個数の相関（|r| < 0.3）")
    for split, entry in sorted(data["verification"]["length_vs_true_count"].items()):
        n = entry.get("n") or 0
        se = 1 / max(n - 3, 1) ** 0.5
        print(f"    {split:5s} r={entry.get('r', 0):+.4f} ±{se:.3f} (n={n})")

    # 3. Implication violations, tier I, retired and active apart.
    print("\n[3] I 段の含意違反率（gold=false を検算が true と読む、< 0.20）")
    groups: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for name, r in rows.items():
        if r["tier"] != "I":
            continue
        negatives = r["total"] - round(r["positive_rate"] * r["total"])
        key = "retired" if name in retired else "active"
        groups[key][0] += r["false_to_true"]
        groups[key][1] += negatives
        groups["all"][0] += r["false_to_true"]
        groups["all"][1] += negatives
    for key in ("all", "active", "retired"):
        v, n = groups[key]
        print(f"    {key:8s} {v}/{n} = {v / max(n, 1):.3f}")

    # 4. Rebalance.
    print("\n[4] 再バランス後の陽性率（長さ 5 層、40〜60%）")
    pairs = kept_pairs(manifest.with_suffix("").with_suffix(".jsonl"))
    before: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in pairs:
        before[p["attribute"]][0] += p["label"]
        before[p["attribute"]][1] += 1
    balanced, report = rebalance(pairs, random.Random(0))
    after: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for p in balanced:
        after[p["attribute"]][0] += p["label"]
        after[p["attribute"]][1] += 1
    outside = []
    for name in retired + sorted(n for n in after if n not in retired):
        if name not in after:
            continue
        rate = after[name][0] / after[name][1]
        if not 0.40 <= rate <= 0.60:
            outside.append((name, rate))
        if name in retired:
            b = before[name]
            print(f"    {name:32s} {b[0]/b[1]:.3f} (n={b[1]}) -> {rate:.3f} (n={after[name][1]})")
    print(f"    40〜60% の外: {len(outside)} 属性 {outside}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "manifest": str(manifest), "baseline": args.baseline,
        "retired": table, "passed": passed, "n_passed": len(passed),
        "length_vs_true_count": data["verification"]["length_vs_true_count"],
        "implication_violation": {k: {"violations": v, "negatives": n}
                                  for k, (v, n) in groups.items()},
        "rebalance_outside_band": outside,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
