"""Merge generated corpora, re-namespacing the document ids.

    uv run python scripts/merge_corpora.py --out data/docs_v2plus3.jsonl \
        v2=data/docs_v2.jsonl v3=data/docs_v3.jsonl

`build_intent_corpus.py` numbers documents `i2-000000` upward from zero on every run,
so two corpora built with it share ids -- 4,687 of them between docs_v2 and docs_v3.
That matters because everything downstream keys on `doc_id`: the train/validation
split is by document, `rebalance` groups by document, and the "no document in both
splits" check would compare two unrelated documents that happen to share a number.
Merging without renaming would silently violate the one property the split exists to
guarantee.

Each document keeps the split it was generated into, so validation documents keep
carrying the held-out attributes they were conditioned on.

`--labels-only-from v5=a,b` keeps attributes a and b only in the documents of source
v5 and strips them from every other source. Night 3 needs it: an attribute re-admitted
because a second generator could write it should be trained on that generator's
documents, not on the older corpus whose failure to write it got it retired in the
first place.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", nargs="+", help="prefix=path, e.g. v2=data/docs_v2.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--labels-only-from", default=None,
                        help="PREFIX=attr1,attr2: keep these intent labels only in PREFIX")
    args = parser.parse_args()

    only_source, only_attrs = None, set()
    if args.labels_only_from:
        only_source, _, names = args.labels_only_from.partition("=")
        only_attrs = {n.strip() for n in names.split(",") if n.strip()}
        if not only_source or not only_attrs:
            raise SystemExit("expected --labels-only-from PREFIX=attr1,attr2")
    stripped: Counter = Counter()

    seen: set[str] = set()
    by_source: Counter = Counter()
    splits: Counter = Counter()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as fh:
        for source in args.sources:
            prefix, _, path = source.partition("=")
            if not path:
                raise SystemExit(f"expected prefix=path, got {source!r}")
            for line in Path(path).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                doc = json.loads(line)
                if only_attrs and prefix != only_source:
                    for name in only_attrs & set(doc["intent_labels"]):
                        del doc["intent_labels"][name]
                        (doc.get("verdicts") or {}).pop(name, None)
                        stripped[(prefix, name)] += 1
                doc["doc_id"] = f"{prefix}-{doc['doc_id']}"
                doc["source_corpus"] = prefix
                if doc["doc_id"] in seen:
                    raise SystemExit(f"duplicate after prefixing: {doc['doc_id']}")
                seen.add(doc["doc_id"])
                by_source[prefix] += 1
                splits[(prefix, doc["split"])] += 1
                fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"{len(seen)} documents -> {out}")
    for prefix, count in by_source.items():
        print(f"  {prefix}: {count}")
    for (prefix, split), count in sorted(splits.items()):
        print(f"  {prefix}/{split}: {count}")
    if only_attrs:
        print(f"labels kept only in {only_source}: {sorted(only_attrs)}")
        for (prefix, name), count in sorted(stripped.items()):
            print(f"  stripped {count} x {name} from {prefix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
