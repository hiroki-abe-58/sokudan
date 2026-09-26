"""Summarise `probe_stock_phrases.py` runs, one column per verifier.

    uv run python scripts/report_phrase_probe.py \
        qwen3=data/reverify/phrases_qwen3.jsonl gemma=data/reverify/phrases_gemma.jsonl

For each formula: the 「はい」 rate on documents written without the attribute, as
written and with the formula appended; the paired flips; and an exact McNemar p on the
flips. The effect of the formula on a verifier is the rise from the first rate to the
second.
"""

from __future__ import annotations

import argparse
import json
from math import comb
from pathlib import Path
from typing import Any


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if not n:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for family in sorted({r["family"] for r in rows}):
        pairs = [r for r in rows if r["family"] == family
                 and r["plain"] in ("はい", "いいえ", "判断できない")
                 and r["with_formula"] in ("はい", "いいえ", "判断できない")]
        n = len(pairs)
        plain = sum(r["plain"] == "はい" for r in pairs)
        withf = sum(r["with_formula"] == "はい" for r in pairs)
        up = sum(r["plain"] != "はい" and r["with_formula"] == "はい" for r in pairs)
        down = sum(r["plain"] == "はい" and r["with_formula"] != "はい" for r in pairs)
        out[family] = {"attribute": pairs[0]["attribute"] if pairs else None, "n": n,
                       "yes_plain": plain / max(n, 1), "yes_with_formula": withf / max(n, 1),
                       "rise": (withf - plain) / max(n, 1), "flips_up": up,
                       "flips_down": down, "mcnemar_p": mcnemar(up, down)}
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", help="label=path")
    parser.add_argument("--out", default="runs/diagnostics/phrase_probe.json")
    args = parser.parse_args()

    results = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        results[label] = summarise(rows)

    families = sorted({f for r in results.values() for f in r})
    for family in families:
        attribute = next(r[family]["attribute"] for r in results.values() if family in r)
        print(f"== {family}  -> {attribute} ==")
        print(f"  {'verifier':10s}{'n':>5s}{'はい 素':>10s}{'はい +定型句':>14s}{'上昇':>9s}"
              f"{'いいえ→はい':>12s}{'はい→いいえ':>12s}{'McNemar p':>12s}")
        for label, r in results.items():
            e = r.get(family)
            if not e:
                continue
            print(f"  {label:10s}{e['n']:5d}{e['yes_plain']:10.3f}{e['yes_with_formula']:14.3f}"
                  f"{e['rise']:+9.3f}{e['flips_up']:12d}{e['flips_down']:12d}"
                  f"{e['mcnemar_p']:12.2e}")
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
