"""Phase 6.1 of docs/length_2x2.md: the data-amount control ("replace").

    uv run python scripts/l2x2_replace.py --seed 20260930

Takes v0.1's training set, removes as many *documents* as the long set has (chosen
with a fixed seed, every row of a removed document goes), and adds every long row. The
document count is then v0.1's, so "long" and "more data" come apart: if replace still
beats short on M2, the length effect does not need the extra volume to appear.

Writes `data/l2x2/train_replace.jsonl` and `runs/l2x2/replace.json` (counts, removed
document ids, type ratios, sha256).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    base = load(Path("data/train_v2b.jsonl"))
    new = load(Path("data/l2x2/expand/train_v2.jsonl"))
    long_docs = sorted({r["doc_id"] for r in new})
    base_docs = sorted({r["doc_id"] for r in base})
    removed = set(random.Random(args.seed).sample(base_docs, len(long_docs)))
    kept = [r for r in base if r["doc_id"] not in removed]
    rows = kept + new
    out = Path("data/l2x2/train_replace.jsonl")
    with out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter(r["kind"] for r in rows)
    report = {
        "seed": args.seed, "long_docs": len(long_docs), "removed_docs": len(removed),
        "docs_total": len({r["doc_id"] for r in rows}), "v01_docs": len(base_docs),
        "rows": len(rows), "rows_removed": len(base) - len(kept), "rows_added": len(new),
        "kinds": {k: {"n": kinds[k], "share": round(kinds[k] / len(rows), 4)}
                  for k in ("choice", "score", "bool")},
        "removed_doc_ids": sorted(removed),
        "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
    }
    Path("runs/l2x2/replace.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "removed_doc_ids"},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
