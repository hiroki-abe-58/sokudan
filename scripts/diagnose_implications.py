"""Find the exclusion pairs the implication graph missed, from measured violations.

    uv run python scripts/diagnose_implications.py data/smoke2_v4.jsonl

A tier-I attribute set to false whose document the verifier nevertheless reads as true
is an implication failure. `docs/day3_attribute_expansion.md` §3 derived 26 exclusion
pairs by reasoning about what entails what; the 1,163 tier-I observations in the smoke
corpus say which ones that reasoning missed.

For each (tier-I attribute A set false, other attribute B set true) that co-occur, this
compares the violation rate when B is present against when it is absent. A B that lifts
the rate well above the attribute's own baseline is entailing A, and the pair belongs in
`EXCLUSIVE_PAIRS`.

Rates on a handful of documents mean nothing, so a pair is only reported when both the
present and absent groups clear `--min-n`. The output is a ranked list to read, not a
patch to apply: a lift can also mean the two attributes simply share a domain.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from sokudan.data import intent_attributes as ia

YES = "はい"


def load(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", default="data/smoke2_v4.jsonl")
    parser.add_argument("--min-n", type=int, default=5,
                        help="minimum documents in each of the present/absent groups")
    parser.add_argument("--min-lift", type=float, default=0.25)
    parser.add_argument("--out", default="runs/diagnostics/implication_pairs.json")
    args = parser.parse_args()

    documents = load(Path(args.corpus))
    existing = {frozenset(p) for p in ia.EXCLUSIVE_PAIRS}

    # For each (implicit attribute set false, co-occurring attribute), how often did the
    # verifier read the implicit attribute as true?
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    baseline: dict[str, list[int]] = defaultdict(list)
    for document in documents:
        labels = document["intent_labels"]
        verdicts = document.get("verdicts") or {}
        present = {k for k, v in labels.items() if v}
        for attribute_id, gold in labels.items():
            if gold or attribute_id not in ia.BY_ID:
                continue
            if ia.BY_ID[attribute_id].tier != "I":
                continue
            answer = verdicts.get(attribute_id)
            if answer is None:
                continue
            violated = int(answer == YES)
            baseline[attribute_id].append(violated)
            for other in present:
                if other in ia.BY_ID:
                    cells[(attribute_id, other)].append(violated)

    rows = []
    for (attribute_id, other), values in cells.items():
        absent = [v for v in baseline[attribute_id]]
        n_present = len(values)
        # The absent group is the attribute's other false observations minus these.
        n_absent = len(absent) - n_present
        if n_present < args.min_n or n_absent < args.min_n:
            continue
        rate_present = sum(values) / n_present
        rate_absent = (sum(absent) - sum(values)) / n_absent
        lift = rate_present - rate_absent
        if lift < args.min_lift:
            continue
        rows.append({
            "implicit": attribute_id, "other": other,
            "other_tier": ia.BY_ID[other].tier,
            "n_present": n_present, "n_absent": n_absent,
            "rate_present": round(rate_present, 3), "rate_absent": round(rate_absent, 3),
            "lift": round(lift, 3),
            "already_excluded": frozenset((attribute_id, other)) in existing,
        })

    rows.sort(key=lambda r: -r["lift"])
    print(f"{len(documents)} documents; {len(rows)} pairs with lift >= {args.min_lift} "
          f"and >= {args.min_n} documents on each side\n")
    print(f"{'implicit (false)':32s}{'co-occurring (true)':32s}{'t':2s}"
          f"{'n+':>5s}{'n-':>5s}{'rate+':>7s}{'rate-':>7s}{'lift':>7s}")
    print("-" * 104)
    for row in rows:
        mark = "  (already excluded)" if row["already_excluded"] else ""
        print(f"{row['implicit']:32s}{row['other']:32s}{row['other_tier']:2s}"
              f"{row['n_present']:5d}{row['n_absent']:5d}{row['rate_present']:7.3f}"
              f"{row['rate_absent']:7.3f}{row['lift']:+7.3f}{mark}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"corpus": args.corpus, "pairs": rows},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
