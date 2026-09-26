"""Compare two verifiers' readings of the same documents (night 4, item 2).

    uv run python scripts/compare_verifiers.py \
        data/reverify/smoke2_gemma.jsonl data/reverify/v5_gemma.jsonl

Each input row carries the original verdicts (`verdicts_ref`, qwen3) and the new ones
(`verdicts`). Reported, per corpus:

1. **false->true on the ten retired attributes** under each verifier: how often a
   document written *without* the attribute is read as having it.
2. **Stock phrases.** For each family of business-register formula, restricted to
   documents whose gold label for the matching attribute is false: the false->true rate
   when the formula is present and when it is absent, under each verifier. A verifier
   that reads the formula as the intent shows a large present-minus-absent gap.
3. **Agreement** between the two verifiers on the retired attributes (Cohen's kappa).
4. **Discard on every attribute** of the re-verified documents under each verifier --
   whether a verifier that did not write the document is stricter than one that did.

Nothing here feeds back into training data.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from sokudan.data import intent_attributes as ia

YES, NO, UNKNOWN = "はい", "いいえ", "判断できない"
RETIRED = [a.id for a in ia.RETIRED_DEFINITIONS]

# Business-register formulas, each paired with the retired attribute a reader might take
# it for. Taken from the documents read by hand (night 3 §6.7, day 3 §8.1).
FAMILIES: dict[str, tuple[str, re.Pattern[str]]] = {
    "検討のお願い": ("implies_testing_the_waters",
                 re.compile(r"ご検討(いただけ|頂け|のほど|くださ|願)|もしよろしければ|よろしければ")),
    "理解のお願い": ("implies_seeking_validation",
                 re.compile(r"ご理解(のほど|いただ|頂|賜)|ご承知(おき|のほど)|ご了承(のほど|いただ|くださ)")),
    "締めの挨拶": ("implies_closing_the_topic",
               re.compile(r"今後とも|引き続き.{0,6}よろしく|改めて.{0,12}(感謝|御礼|お礼)")),
    "事情を伏せる定型": ("implies_partial_disclosure",
                  re.compile(r"諸事情|諸般の事情|一身上の")),
}


def answer(verdicts: dict | None, attribute: str) -> str | None:
    if not verdicts:
        return None
    return verdicts.get(attribute)


def rate(hits: int, n: int) -> str:
    return f"{hits / n:.3f} ({hits}/{n})" if n else "—"


def kappa(pairs: list[tuple[bool, bool]]) -> float | None:
    n = len(pairs)
    if not n:
        return None
    agree = sum(a == b for a, b in pairs) / n
    pa = sum(a for a, _ in pairs) / n
    pb = sum(b for _, b in pairs) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected >= 1 else (agree - expected) / (1 - expected)


def analyse(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"retired": {}, "families": {}, "discard": {}}

    # 1. false->true on the retired ten.
    print("  [1] 除外 10 属性の false->true（gold=false を「はい」と読む率）")
    print(f"      {'attribute':32s}{'n_false':>8s}{'qwen3':>18s}{'new':>18s}")
    for name in RETIRED:
        cells = {"ref": [0, 0], "new": [0, 0]}
        for row in rows:
            if row["intent_labels"].get(name, 1) != 0:
                continue
            for key, field in (("ref", "verdicts_ref"), ("new", "verdicts")):
                a = answer(row.get(field), name)
                if a is None:
                    continue
                cells[key][1] += 1
                cells[key][0] += int(a == YES)
        if not cells["ref"][1]:
            continue
        result["retired"][name] = cells
        print(f"      {name:32s}{cells['ref'][1]:8d}{rate(*cells['ref']):>18s}"
              f"{rate(*cells['new']):>18s}")
    pooled = {k: [sum(c[k][0] for c in result["retired"].values()),
                  sum(c[k][1] for c in result["retired"].values())] for k in ("ref", "new")}
    result["retired_pooled"] = pooled
    print(f"      {'pooled':32s}{pooled['ref'][1]:8d}{rate(*pooled['ref']):>18s}"
          f"{rate(*pooled['new']):>18s}")

    # 2. Stock phrases.
    print("\n  [2] 定型句: gold=false の文書で、定型句あり / なし の false->true")
    print(f"      {'family':14s}{'attribute':30s}{'':>4s}{'n':>5s}{'qwen3':>16s}{'new':>16s}")
    for family, (attribute, pattern) in FAMILIES.items():
        entry = {}
        for present in (True, False):
            cells = {"ref": [0, 0], "new": [0, 0]}
            for row in rows:
                if row["intent_labels"].get(attribute, 1) != 0:
                    continue
                if bool(pattern.search(row["state"])) != present:
                    continue
                for key, field in (("ref", "verdicts_ref"), ("new", "verdicts")):
                    a = answer(row.get(field), attribute)
                    if a is None:
                        continue
                    cells[key][1] += 1
                    cells[key][0] += int(a == YES)
            entry["present" if present else "absent"] = cells
            label = "あり" if present else "なし"
            n = cells["ref"][1]
            print(f"      {family:14s}{attribute:30s}{label:>4s}{n:5d}"
                  f"{rate(*cells['ref']):>16s}{rate(*cells['new']):>16s}")
        result["families"][family] = {"attribute": attribute, **entry}

    # 3. Agreement on the retired attributes.
    pairs = []
    for row in rows:
        for name in RETIRED:
            if name not in row["intent_labels"]:
                continue
            a, b = answer(row.get("verdicts_ref"), name), answer(row.get("verdicts"), name)
            if a in (YES, NO) and b in (YES, NO):
                pairs.append((a == YES, b == YES))
    agree = sum(a == b for a, b in pairs) / max(len(pairs), 1)
    k = kappa(pairs)
    result["agreement"] = {"n": len(pairs), "raw": agree, "kappa": k}
    print(f"\n  [3] 除外属性での 2 検算器の一致: {agree:.3f}  kappa "
          f"{'—' if k is None else f'{k:.3f}'}  (n={len(pairs)})")

    # 4. Discard on every attribute, by tier.
    by_tier: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"ref": [0, 0], "new": [0, 0]})
    for row in rows:
        for name, gold in row["intent_labels"].items():
            tier = "retired" if name in RETIRED else ia.ALL_BY_ID[name].tier
            for key, field in (("ref", "verdicts_ref"), ("new", "verdicts")):
                a = answer(row.get(field), name)
                kept = a in (YES, NO) and (a == YES) == bool(gold)
                by_tier[tier][key][1] += 1
                by_tier[tier][key][0] += int(not kept)
    result["discard"] = {t: v for t, v in by_tier.items()}
    print("\n  [4] 破棄率（不一致 + 判断できない + 欠落）")
    for tier in ("S", "E", "I", "retired"):
        if tier in by_tier:
            v = by_tier[tier]
            print(f"      {tier:8s} qwen3 {rate(*v['ref']):>18s}   new {rate(*v['new']):>18s}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--out", default="runs/diagnostics/verifier_compare.json")
    args = parser.parse_args()

    results = {}
    for path in args.inputs:
        rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        meta_path = Path(path).with_suffix(".meta.json")
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        print(f"== {path}: {len(rows)} documents, generated by {meta.get('corpus_generator')}, "
              f"reference verifier {meta.get('reference_verifier')}, "
              f"new verifier {meta.get('verifier_model')} ==")
        results[path] = {"meta": meta, **analyse(rows)}
        print()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
