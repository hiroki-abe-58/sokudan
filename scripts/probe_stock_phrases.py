"""Does a business-register formula, on its own, make a verifier read an intent?

    uv run python scripts/probe_stock_phrases.py \
        --verifier-model qwen3:30b-a3b-instruct-2507-q4_K_M \
        --out data/reverify/phrases_qwen3.jsonl

The observational comparison in `compare_verifiers.py` is starved: in mistral's
gold=false documents the formulas are rare (「ご検討…」 appears in 3 of 119 negatives for
`implies_testing_the_waters`). So this manipulates instead of observing.

For each (formula, attribute) pair, take documents written *without* the attribute and
ask the verifier about it twice: once as written, once with the formula appended as a
final sentence. Same document, same question, same criteria; the only difference is the
formula. The rise in 「はい」 is the effect of the formula alone on that verifier.

Run once per verifier with the same seed and the same documents, and the two effects
can be compared directly. Analysis only -- nothing here touches training data.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
from collections import Counter
from pathlib import Path

from scripts.build_intent_corpus import (
    VERIFY_SYSTEM,
    build_verify_prompt,
    parse_verdicts,
    verify_one,
)
from sokudan.config import LocalLLMConfig
from sokudan.data import intent_attributes as ia
from sokudan.data.local_llm import LocalLLM, LocalLLMError

# (formula family, formula appended, attribute it might be read as). The formulas are the
# ones found in documents read by hand: night 3 §6.7 and day 3 §8.1.
PROBES = [
    ("検討のお願い", "再度ご検討いただけますと幸いです。", "implies_testing_the_waters"),
    ("理解のお願い", "何卒ご理解のほどよろしくお願い申し上げます。", "implies_seeking_validation"),
    ("締めの挨拶", "今後ともどうぞよろしくお願いいたします。", "implies_closing_the_topic"),
]
CORPORA = ["data/docs_v5.jsonl", "data/smoke2_v4.jsonl"]


def pick(attribute: str, formula: str, n: int, seed: int) -> list[dict]:
    """Gold=false documents for `attribute` that do not already contain the formula."""
    pool = []
    for path in CORPORA:
        seeds[path] = int(json.loads(
            Path(path).with_suffix(".manifest.json").read_text(encoding="utf-8"))["seed"])
        for index, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            doc = json.loads(line)
            if doc["intent_labels"].get(attribute) == 0 and formula[:4] not in doc["state"]:
                pool.append({"doc_id": f"{Path(path).stem}:{doc['doc_id']}",
                             "state": doc["state"],
                             # For --in-context: enough to rebuild the production prompt.
                             "document": doc, "corpus": path, "index": index})
    random.Random(seed).shuffle(pool)
    return pool[:n]


seeds: dict[str, int] = {}


async def ask_in_context(llm: LocalLLM, doc: dict, state: str, attribute: str,
                         failures: Counter) -> str | None:
    """The production verification prompt -- every question on the document, in the
    order the original run asked them -- with `state` as the body."""
    document = {**doc["document"], "state": state}
    rng = random.Random(seeds[doc["corpus"]] * 104_729 + doc["index"])
    verdicts = await verify_one(llm, document, rng, paraphrase=False, failures=failures)
    return (verdicts or {}).get(attribute)


async def ask(llm: LocalLLM, state: str, attribute: str, failures: Counter) -> str | None:
    a = ia.ALL_BY_ID[attribute]
    prompt = build_verify_prompt(state, [(a.forms[0], a.true_behaviour, a.false_behaviour)])
    try:
        result = await llm.chat([{"role": "system", "content": VERIFY_SYSTEM},
                                 {"role": "user", "content": prompt}],
                                temperature=0.0, max_tokens=100)
    except LocalLLMError as exc:
        failures[type(exc).__name__] += 1
        return None
    verdicts = parse_verdicts(result.text, 1)
    if not verdicts:
        failures["unparseable"] += 1
        return None
    return verdicts.get(1)


async def run(args: argparse.Namespace) -> int:
    config = dataclasses.replace(LocalLLMConfig.from_env(), model=args.verifier_model)
    failures: Counter = Counter()
    rows = []
    async with LocalLLM(config, concurrency=args.concurrency, timeout_s=600.0) as llm:
        for family, formula, attribute in PROBES:
            docs = pick(attribute, formula, args.n, args.seed)

            async def one(doc: dict, family=family, formula=formula, attribute=attribute):
                base = doc["state"].rstrip()
                if args.in_context:
                    plain = await ask_in_context(llm, doc, base, attribute, failures)
                    with_formula = await ask_in_context(
                        llm, doc, base + "\n" + formula, attribute, failures)
                else:
                    plain = await ask(llm, base, attribute, failures)
                    with_formula = await ask(llm, base + "\n" + formula, attribute, failures)
                rows.append({"family": family, "attribute": attribute, "formula": formula,
                             "doc_id": doc["doc_id"], "plain": plain,
                             "with_formula": with_formula})

            await asyncio.gather(*(one(d) for d in docs))
            print(f"  {family}: {len(docs)} documents", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["family"], r["doc_id"]))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")
    print(f"-> {out} ({len(rows)} pairs, failures {dict(failures)})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verifier-model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=120, help="documents per formula")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument(
        "--in-context", action="store_true",
        help="ask with the production verification prompt (every question on the "
             "document, original order) instead of the single question alone")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
