"""Re-verify an existing corpus with a different model, prompt for prompt.

    uv run python scripts/reverify.py data/smoke2_v4.jsonl --verifier-model gemma3:27b \
        --out data/reverify/smoke2_gemma.jsonl

The question this answers is about the *verifier*, so everything else is held fixed.
Each document is asked exactly what the original verifier was asked: the same question
forms, the same criteria, the same order -- the order comes from the seeded shuffle in
`build_intent_corpus.verify_one`, reproduced here from the corpus's own manifest seed
and the document's position. Only the model reading the prompt changes.

By default only documents carrying a retired attribute are re-verified -- those are the
ones night 3 is about. Every attribute on such a document is re-asked, because the
verifier sees them together and an answer can depend on its neighbours.

**Analysis only.** The new verdicts are written next to the original ones
(`verdicts_ref`) and are not used to select, relabel or re-admit any training data.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.build_intent_corpus import verify_one
from sokudan.config import LocalLLMConfig
from sokudan.data import intent_attributes as ia
from sokudan.data.local_llm import LocalLLM

RETIRED_IDS = {a.id for a in ia.RETIRED_DEFINITIONS}


async def run(args: argparse.Namespace) -> int:
    corpus = Path(args.corpus)
    manifest = json.loads(corpus.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    seed = int(manifest["seed"])
    documents = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()
                 if line.strip()]

    # Position in the file is the index the original run seeded the shuffle with:
    # documents were written in the order they were verified.
    wanted: set[str] | None = None
    if args.doc_ids:
        wanted = {line.strip() for line in Path(args.doc_ids).read_text(encoding="utf-8")
                  .splitlines() if line.strip()}
    chosen = [
        (index, d) for index, d in enumerate(documents)
        if (wanted is not None and d["doc_id"] in wanted
            or wanted is None and (not args.only_retired
                                   or RETIRED_IDS & set(d["intent_labels"])))
        and set(d["intent_labels"]) <= set(ia.ALL_BY_ID)
    ]
    if wanted is not None:
        found = {d["doc_id"] for _, d in chosen}
        if wanted - found:
            print(f"  {len(wanted - found)} requested doc_ids not found or not verifiable",
                  flush=True)
    if args.limit:
        chosen = chosen[: args.limit]
    print(f"{corpus}: {len(documents)} documents, re-verifying {len(chosen)} "
          f"(seed {seed}, original verifier {manifest.get('verifier_model')})", flush=True)

    config = dataclasses.replace(LocalLLMConfig.from_env(), model=args.verifier_model)
    failures: Counter = Counter()
    out_rows: list[dict[str, Any]] = []
    started = time.time()
    done = 0

    async with LocalLLM(config, concurrency=args.concurrency, timeout_s=600.0) as llm:
        async def worker(index: int, document: dict[str, Any]) -> None:
            nonlocal done
            verdicts = await verify_one(
                llm, document, random.Random(seed * 104_729 + index),
                paraphrase=False, failures=failures,
            )
            out_rows.append({
                "index": index, "doc_id": document["doc_id"], "split": document["split"],
                "state": document["state"], "intent_labels": document["intent_labels"],
                "verdicts_ref": document.get("verdicts"), "verdicts": verdicts,
            })
            done += 1
            if done % 100 == 0:
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{len(chosen)}  ({rate:.2f} docs/s, "
                      f"~{(len(chosen) - done) / max(rate, 1e-9) / 60:.1f} min left)", flush=True)

        await asyncio.gather(*(worker(i, d) for i, d in chosen))

    out_rows.sort(key=lambda r: r["index"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows),
                   encoding="utf-8")
    meta = {
        "corpus": str(corpus), "corpus_generator": manifest.get("generator_model"),
        "reference_verifier": manifest.get("verifier_model") or manifest.get("generator_model"),
        "verifier_model": args.verifier_model, "seed": seed,
        "documents": len(out_rows), "only_retired": args.only_retired,
        "failures": dict(failures), "elapsed_s": round(time.time() - started, 1),
        "analysis_only": True,
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"-> {out}  ({len(out_rows)} documents, failures {dict(failures)})", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus")
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--all", dest="only_retired", action="store_false",
                        help="re-verify every document, not only those with a retired attribute")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc-ids", default=None,
                        help="file of doc_ids (one per line, as in the corpus) to re-verify; "
                             "overrides the retired-attribute filter")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
