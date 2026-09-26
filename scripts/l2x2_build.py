"""Phase 3 of docs/length_2x2.md: fix the long documents and build the long training set.

    uv run python scripts/l2x2_build.py --dir <snapshot>

1. Merge the trial and every batch, renaming ids to `l2x-b<kk>-<index>` (each batch
   numbered its documents from 0).
2. Keep documents whose state is 450-800 tokens (M2's counting).
3. Drop any whose character 5-gram Jaccard with *any* evaluation state is >= 0.5:
   frozen held-out, all of `docs_long_val` (M2's source documents), v4 held-out, every
   val-split document of `docs_all_v5` (M4's source), and `val_v2`.
4. Expand the survivors with v0.1's own expansion (`l2x2_pipeline.py expand`,
   `--bool-share 0.63`) and write `data/l2x2/train_long.jsonl` = every row of
   `data/train_v2b.jsonl` followed by the new rows.

Counts, lengths, type ratios and sha256 go to `runs/l2x2/build.json`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401

DATA = Path("data/l2x2")
TARGET = (450, 800)
JACCARD = 0.5
N = 5


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grams(text: str) -> frozenset[str]:
    return frozenset(text[i:i + N] for i in range(max(len(text) - N + 1, 1)))


def eval_states() -> list[str]:
    states: set[str] = set()
    for path in ("data/v2/heldout_val_v2.jsonl", "data/v4/heldout_val_v4.jsonl",
                 "data/val_v2.jsonl"):
        states |= {r["state"] for r in load(Path(path))}
    states |= {d["state"] for d in load(Path("data/docs_long_val.jsonl"))}
    states |= {d["state"] for d in load(Path("data/docs_all_v5.jsonl")) if d.get("split") == "val"}
    return sorted(states)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    report: dict = {}

    sources = [DATA / "trial_docs.jsonl"] + sorted(DATA.glob("batch_*.jsonl"))
    merged = []
    for k, path in enumerate(sources):
        for d in load(path):
            index = d["doc_id"].split("-")[-1]
            merged.append({**d, "doc_id": f"l2x-b{k:02d}-{index}", "source_file": path.name})
    report["generated_docs_kept"] = len(merged)
    report["source_files"] = {p.name: {"docs": len(load(p)), "sha256": sha256(p)} for p in sources}

    tokens = [len(tokenizer(d["state"], add_special_tokens=True)["input_ids"]) for d in merged]
    on_target = [d for d, t in zip(merged, tokens, strict=True) if TARGET[0] <= t <= TARGET[1]]
    report["on_target"] = len(on_target)
    report["off_target"] = {"below": sum(t < TARGET[0] for t in tokens),
                            "above": sum(t > TARGET[1] for t in tokens)}

    evals = [grams(s) for s in eval_states()]
    report["eval_states_compared"] = len(evals)
    kept, excluded = [], []
    for d in on_target:
        g = grams(d["state"])
        best = 0.0
        for e in evals:
            small, large = sorted((len(g), len(e)))
            if small < JACCARD * large:  # Jaccard <= small / large < 0.5
                continue
            inter = len(g & e)
            best = max(best, inter / (len(g) + len(e) - inter))
            if best >= JACCARD:
                break
        (excluded if best >= JACCARD else kept).append(d)
    report["dedup_excluded"] = len(excluded)
    report["dedup_excluded_ids"] = [d["doc_id"] for d in excluded]
    report["long_docs"] = len(kept)

    long_path = DATA / "long_docs.jsonl"
    with long_path.open("w", encoding="utf-8") as fh:
        for d in kept:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    t = np.array([len(tokenizer(d["state"], add_special_tokens=True)["input_ids"]) for d in kept])
    report["long_docs_tokens"] = {"mean": round(float(t.mean()), 1), "median": float(np.median(t)),
                                  "p95": float(np.percentile(t, 95)), "min": int(t.min()),
                                  "max": int(t.max())}
    report["long_docs_domains"] = dict(sorted(Counter(d["domain"] for d in kept).items()))

    expand_dir = DATA / "expand"
    code = subprocess.run([sys.executable, "scripts/l2x2_pipeline.py", "expand", "--dir",
                           args.dir, "--docs-file", str(long_path), "--out-dir",
                           str(expand_dir)]).returncode
    if code != 0:
        raise SystemExit(f"expansion failed: {code}")
    new_rows = load(expand_dir / "train_v2.jsonl")
    base_rows = load(Path("data/train_v2b.jsonl"))
    out = DATA / "train_long.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for r in base_rows + new_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def kinds(rows: list[dict]) -> dict:
        c = Counter(r["kind"] for r in rows)
        return {k: {"n": c[k], "share": round(c[k] / len(rows), 4)}
                for k in ("choice", "score", "bool")}

    report["rows"] = {"v0.1 (train_v2b)": {"n": len(base_rows), **kinds(base_rows)},
                      "new long rows": {"n": len(new_rows), **kinds(new_rows)},
                      "train_long": {"n": len(base_rows) + len(new_rows),
                                     **kinds(base_rows + new_rows)}}
    report["docs"] = {"v0.1 train docs": len({r["doc_id"] for r in base_rows}),
                      "long docs with rows": len({r["doc_id"] for r in new_rows})}
    report["sha256"] = {str(p): sha256(p) for p in (long_path, expand_dir / "train_v2.jsonl", out)}
    Path("runs/l2x2/build.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    brief = {k: v for k, v in report.items() if k != "dedup_excluded_ids"}
    print(json.dumps(brief, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
