"""Sample verifier disagreements for reading by hand (night 3 §3, the <5 branch).

    uv run python scripts/sample_mismatches.py data/rejected_v5.jsonl --n 20

The discard rate cannot say *why* the generator and verifier disagreed. Two readings
predict the same number:

* **not written** -- told to carry the attribute (or to avoid it), the generator did
  not; the verifier read the text correctly;
* **written but refused** -- the text does what the instruction asked, and the verifier
  reads it the other way, because the attribute is ambient in the register or its
  criteria are ambiguous.

Only reading the documents separates them. This picks the documents to read: evenly
across the failing attributes, and within each attribute evenly across the two
directions, so a direction that dominates the count does not also dominate the sample.
Each is printed with both criteria the verifier was shown.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from sokudan.data import intent_attributes as ia


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rejected", nargs="?", default="data/rejected_v5.jsonl")
    parser.add_argument("--attributes", nargs="*", default=None,
                        help="restrict to these; default = every retired attribute")
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--out", default="runs/diagnostics/v5_mismatch_sample.jsonl")
    args = parser.parse_args()

    wanted = set(args.attributes or [a.id for a in ia.RETIRED])
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in Path(args.rejected).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record["attribute"] not in wanted or record["reason"] != "mismatch":
            continue
        direction = "false->true" if not record["label"] else "true->false"
        buckets[(record["attribute"], direction)].append(record)

    rng = random.Random(args.seed)
    for items in buckets.values():
        rng.shuffle(items)
    # Round-robin over (attribute, direction) so every cell contributes before any
    # contributes twice.
    keys = sorted(buckets)
    picked: list[dict] = []
    while len(picked) < args.n and any(buckets[k] for k in keys):
        for key in keys:
            if buckets[key] and len(picked) < args.n:
                picked.append({**buckets[key].pop(), "direction": key[1]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for index, record in enumerate(picked, start=1):
            attribute = ia.ALL_BY_ID[record["attribute"]]
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"### {index}. {record['attribute']}  gold={record['label']}  "
                  f"verdict={record['verdict']}  ({record['direction']})  {record['doc_id']}")
            print(f"    はい の条件: {attribute.true_behaviour}")
            print(f"    いいえの条件: {attribute.false_behaviour}")
            print(record["state"])
            print()
    print(f"-> {out}  ({len(picked)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
