"""Statistics of a generated document corpus, for the gates in docs/length_2x2.md.

    uv run python scripts/l2x2_docstats.py data/l2x2/trial_docs.jsonl [--baseline]

- throughput: documents kept per hour, generation + verification wall time
  (from the corpus manifest the generator writes);
- target-length hit rate: share of kept documents whose state is 450-800 tokens,
  counted as M2 counts (modernbert-ja tokenizer, special tokens included);
- verification pass rate: intent (document, attribute) pairs whose blind verdict
  agrees with the gold label, over all intent pairs -- the rule that decides which
  pairs survive (`build_intent_train.partition`);
- endings: share of states whose last character closes a sentence, the check that
  showed the 1000-token cap cutting documents off.

`--baseline` also prints the same pass rate for v0.1's training documents
(`data/docs_v2.jsonl`, split == train), the record the trial is compared against.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401

TARGET = (450, 800)
CLOSERS = "。！？」』）)!?."


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def pass_rate(docs: list[dict]) -> dict:
    total = agree = 0
    for d in docs:
        verdicts = d.get("verdicts") or {}
        for attribute, gold in d["intent_labels"].items():
            total += 1
            answer = verdicts.get(attribute)
            agree += int(answer is not None and answer != "判断できない"
                         and (answer == "はい") == bool(gold))
    return {"pairs": total, "agree": agree, "rate": agree / max(total, 1)}


def stats(path: Path, tokenizer) -> dict:
    docs = load(path)
    tokens = np.array([len(tokenizer(d["state"], add_special_tokens=True)["input_ids"])
                       for d in docs])
    out = {"file": str(path), "docs_kept": len(docs),
           "tokens": {"mean": round(float(tokens.mean()), 1),
                      "median": float(np.median(tokens)), "min": int(tokens.min()),
                      "max": int(tokens.max())},
           "hit": int(((tokens >= TARGET[0]) & (tokens <= TARGET[1])).sum()),
           "below": int((tokens < TARGET[0]).sum()), "above": int((tokens > TARGET[1]).sum()),
           "sentence_final_endings": sum(d["state"].rstrip()[-1:] in CLOSERS for d in docs),
           "pass": pass_rate(docs)}
    out["hit_rate"] = out["hit"] / max(len(docs), 1)
    manifest_path = path.with_suffix(".manifest.json")
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        seconds = m["generation_elapsed_s"] + m["verification_elapsed_s"]
        out.update({"requested": m["documents_requested"],
                    "generator_model": m["generator_model"],
                    "seconds_gen_plus_verify": seconds,
                    "docs_per_hour": round(len(docs) / seconds * 3600, 1),
                    "rejections": m["rejections"]})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--baseline", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    for f in args.files:
        print(json.dumps(stats(Path(f), tokenizer), ensure_ascii=False))
    if args.baseline:
        v01 = [d for d in load(Path("data/docs_v2.jsonl")) if d["split"] == "train"]
        print(json.dumps({"v0.1 train docs (data/docs_v2.jsonl)": len(v01),
                          "pass": pass_rate(v01)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
