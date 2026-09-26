"""How many documents a Latin-intrusion filter would drop, per corpus and threshold.

    uv run python scripts/latin_filter_report.py

Report only: nothing is removed from any corpus. The filter itself lives in
`sokudan/data/latin.py` and is applied to *new* generations in
`build_intent_corpus.validate()`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from sokudan.data.latin import latin_intrusion_rate, latin_intrusions

CORPORA = [("v2", "data/docs_v2.jsonl"), ("v3", "data/docs_v3.jsonl"),
           ("v4", "data/docs_v4.jsonl"), ("v5", "data/docs_v5.jsonl")]
# (label, predicate on (intrusion words, rate))
THRESHOLDS = [
    ("1 語以上", lambda words, rate: len(words) >= 1),
    ("2 語以上", lambda words, rate: len(words) >= 2),
    ("率 > 0.5%", lambda words, rate: rate > 0.005),
    ("率 > 1%", lambda words, rate: rate > 0.01),
    ("率 > 2%", lambda words, rate: rate > 0.02),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/diagnostics/latin_filter.json")
    args = parser.parse_args()

    result = {}
    header = f"{'corpus':8s}{'docs':>7s}" + "".join(f"{label:>18s}" for label, _ in THRESHOLDS)
    print(header)
    print("-" * len(header))
    for name, path in CORPORA:
        docs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        scored = [(latin_intrusions(d["state"]), latin_intrusion_rate(d["state"])) for d in docs]
        row = {"documents": len(docs), "drops": {}, "top_words": Counter(
            w for words, _ in scored for w in words).most_common(15)}
        cells = []
        for label, predicate in THRESHOLDS:
            dropped = sum(predicate(words, rate) for words, rate in scored)
            row["drops"][label] = dropped
            cells.append(f"{dropped:>6d} ({dropped / len(docs):5.1%})")
        result[name] = row
        print(f"{name:8s}{len(docs):7d}" + "".join(f"{c:>18s}" for c in cells))

    print()
    for name, row in result.items():
        print(f"{name} top words: {row['top_words'][:10]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
