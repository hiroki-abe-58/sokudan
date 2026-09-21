"""Generate the Day 2 document corpus and verify its labels.

    uv run python scripts/build_intent_corpus.py --docs 5000

Two stages, both against the same local model:

1. **Generate.** Each document is conditioned on its domain's own attributes *and*
   on 6-8 cross-domain intent attributes drawn 50/50 true/false. The conditions are
   the gold labels -- there is no annotation step.
2. **Verify.** The same model, at temperature 0, is asked what the finished document
   actually says, **without being shown the gold**. Pairs where it disagrees are
   dropped.

The blindness matters. Shown the gold, the verifier agrees with it, and the
agreement rate measures nothing. Shown only the text, a disagreement means the
generator did not carry out its instruction -- which is the label error this stage
exists to remove.

What this does **not** measure is whether the label is true. The verifier is the
generator; their errors are correlated. It measures instruction-following, and
`docs/benchmarks.md` has to say so.

The verification questions use `forms[0]`, the same wording the generator saw, so a
disagreement is a generation failure rather than a paraphrase disagreement. A
separate subsample re-verifies with a random paraphrase and reports the extra
disagreement, which is the paraphrase-stability number -- conflating the two would
make the 30% per-attribute discard ceiling fire for the wrong reason.

Output is the document corpus, not training rows. `scripts/build_intent_train.py`
expands it. Keeping them apart means a mistake in expansion does not cost another
hour of generation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sokudan.data import intent_attributes as ia
from sokudan.data.builders.synthetic import (
    CATALOG,
    LONG_LENGTHS,
    SYSTEM_PROMPT,
    Domain,
    banned_words_for,
    build_intent_prompt,
    catalog_summary,
    intent_banned_phrases,
)
from sokudan.data.local_llm import LocalLLM, LocalLLMError

MIN_CHARS = 60
MAX_CHARS = 1600
MAX_CHARS_LONG = 3000
META_MARKERS = ["以下", "承知", "```", "本文:", "###", "了解"]

VERIFY_SYSTEM = (
    "あなたは日本語の文面を読み、設問に答える判定器です。"
    "JSON だけを出力し、解説は一切書きません。"
)

YES, NO, UNKNOWN = "はい", "いいえ", "判断できない"

HELD_OUT_IMPLICIT = sorted(
    name for name in ia.HELD_OUT if ia.BY_ID[name].tier == "I"
)


def validate(text: str, banned: list[str], ceiling: int = MAX_CHARS) -> str | None:
    body = text.strip()
    if len(body) < MIN_CHARS:
        return "too_short"
    if len(body) > ceiling:
        return "too_long"
    for word in banned:
        if word in body:
            return f"leaked:{word}"
    head = body[:40]
    for marker in META_MARKERS:
        if marker in head:
            return f"meta:{marker}"
    return None


def plan_document(
    index: int, domain: Domain, split: str, rng: random.Random
) -> dict[str, Any]:
    """Pick the gold labels for one document before anything is generated."""
    domain_labels = {a.name: a.sample(rng) for a in domain.attributes}

    if split == "val":
        # §5.1 wanted every held-out attribute on every validation document. Three of
        # the five are tier I, and `MAX_IMPLICIT_PER_DOC` now forbids that, so the
        # validation split is cut into three rotating groups instead: each document
        # carries one held-out tier-I attribute plus both held-out non-tier-I ones.
        #
        # The gate is unchanged. It reads the tier-I *pool*, which still gets every
        # validation document (~750 pairs, standard error ~0.018 on AUROC); what
        # shrinks is the per-attribute precision, from ~660 pairs to ~250.
        rotation = HELD_OUT_IMPLICIT[index % len(HELD_OUT_IMPLICIT)]
        forced = [ia.BY_ID[rotation]] + [
            ia.BY_ID[name] for name in sorted(ia.HELD_OUT)
            if ia.BY_ID[name].tier != "I"
        ]
        picks = ia.select_attributes(rng, pool=ia.TRAINABLE, count=8, forced=forced)
    else:
        picks = ia.select_attributes(
            rng, pool=ia.TRAINABLE, count=rng.randint(6, 8)
        )

    intents = [(a, 1 if rng.random() < 0.5 else 0) for a in picks]
    return {
        "doc_id": f"i2-{index:06d}",
        "domain": domain.name,
        "split": split,
        "domain_labels": domain_labels,
        "intents": intents,
    }


async def generate_one(
    llm: LocalLLM, domain: Domain, plan: dict[str, Any], rng: random.Random,
    *, temperature: float, max_attempts: int, rejections: Counter,
    lengths: list[tuple[str, str]] | None = None,
) -> dict[str, Any] | None:
    banned = banned_words_for(domain) + intent_banned_phrases(plan["intents"])
    ceiling = MAX_CHARS_LONG if lengths else MAX_CHARS

    for attempt in range(max_attempts):
        prompt, meta = build_intent_prompt(
            domain, plan["domain_labels"], plan["intents"], rng, lengths
        )
        try:
            result = await llm.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                temperature=temperature, max_tokens=1000,
            )
        except LocalLLMError as exc:
            rejections[f"llm_error:{type(exc).__name__}"] += 1
            continue

        body = result.text.strip()
        reason = validate(body, banned, ceiling)
        if reason is None:
            return {
                "doc_id": plan["doc_id"],
                "domain": plan["domain"],
                "split": plan["split"],
                "state": body,
                "domain_labels": plan["domain_labels"],
                "intent_labels": {a.id: label for a, label in plan["intents"]},
                "context": meta["context"],
                "style": meta["style"],
                "length_bucket": meta["length"],
                "attempts": attempt + 1,
            }
        rejections[reason] += 1
    return None


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def parse_verdicts(text: str, count: int) -> dict[int, str] | None:
    match = _JSON_BLOCK.search(text)
    if match is None:
        return None
    try:
        raw = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None

    out: dict[int, str] = {}
    for key, value in raw.items():
        try:
            number = int(str(key).strip().rstrip("."))
        except ValueError:
            continue
        if not 1 <= number <= count:
            continue
        answer = str(value).strip()
        if answer.startswith(YES):
            out[number] = YES
        elif answer.startswith(NO):
            out[number] = NO
        else:
            out[number] = UNKNOWN
    return out or None


def build_verify_prompt(state: str, questions: list[tuple[str, str, str]]) -> str:
    """Ask about a finished document, with the same criteria the writer was given.

    The first smoke run asked the bare question and the answers saturated: the
    verifier called 15 of 15 negative documents positive on
    `implies_reproach_for_delay`, and missed 12 of 14 positives on
    `implies_out_of_depth`. It was answering from a prior rather than from the text.

    That is a definition mismatch, not a label error. The generator worked from a
    precise behavioural spec; the verifier saw a one-line question, and for a tier-I
    question that line is genuinely ambiguous -- almost any complaint can be read as
    reproaching someone for being slow. Discarding on that signal would have deleted
    every negative example of an attribute and taught the model to answer yes.

    Both criteria are shown, always in the same order, never marked as to which one
    holds. The verifier stays blind to the gold; it is only told what the question
    means.
    """
    blocks = []
    for index, (form, when_true, when_false) in enumerate(questions, start=1):
        blocks.append(
            f"{index}. {form}\n"
            f"   「{YES}」と判断する条件: {when_true}\n"
            f"   「{NO}」と判断する条件: {when_false}"
        )
    numbered = "\n".join(blocks)
    shape = ", ".join(f'"{i}": "はい"' for i in range(1, min(len(questions), 3) + 1))
    return f"""次の日本語の文面を読み、各設問に答えてください。

## 文面
{state}

## 設問
{numbered}

## 答え方
- 各設問に「{YES}」「{NO}」「{UNKNOWN}」のいずれかで答えてください
- 示された判断基準だけに従ってください。一般的な印象で答えないでください
- 文面から判断がつかないときは、推測せず「{UNKNOWN}」と答えてください
- JSON だけを出力してください。形式: {{{shape}, ...}}"""


async def verify_one(
    llm: LocalLLM, document: dict[str, Any], rng: random.Random,
    *, paraphrase: bool, failures: Counter,
) -> dict[str, str] | None:
    ids = list(document["intent_labels"])
    rng.shuffle(ids)
    questions = []
    for attribute_id in ids:
        attribute = ia.BY_ID[attribute_id]
        form = rng.choice(attribute.forms[1:]) if paraphrase else attribute.forms[0]
        questions.append((form, attribute.true_behaviour, attribute.false_behaviour))

    try:
        result = await llm.chat(
            [{"role": "system", "content": VERIFY_SYSTEM},
             {"role": "user", "content": build_verify_prompt(document["state"], questions)}],
            temperature=0.0, max_tokens=500,
        )
    except LocalLLMError as exc:
        failures[f"llm_error:{type(exc).__name__}"] += 1
        return None

    verdicts = parse_verdicts(result.text, len(questions))
    if verdicts is None:
        failures["unparseable"] += 1
        return None
    return {ids[number - 1]: answer for number, answer in verdicts.items()}


def summarise(documents: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-attribute and per-tier discard rates, and the length correlation."""
    per_attribute: dict[str, Counter] = defaultdict(Counter)
    for document in documents:
        verdicts = document.get("verdicts") or {}
        for attribute_id, gold in document["intent_labels"].items():
            answer = verdicts.get(attribute_id)
            if answer is None:
                per_attribute[attribute_id]["missing"] += 1
            elif answer == UNKNOWN:
                per_attribute[attribute_id]["unknown"] += 1
            elif (answer == YES) == bool(gold):
                per_attribute[attribute_id]["agree"] += 1
            else:
                per_attribute[attribute_id]["mismatch"] += 1
            per_attribute[attribute_id]["total"] += 1
            per_attribute[attribute_id]["positives"] += int(gold)

    rows = []
    for attribute_id in sorted(per_attribute, key=lambda k: (ia.BY_ID[k].tier, k)):
        counts = per_attribute[attribute_id]
        total = counts["total"]
        dropped = counts["mismatch"] + counts["unknown"] + counts["missing"]
        rows.append({
            "attribute": attribute_id,
            "tier": ia.BY_ID[attribute_id].tier,
            "held_out": attribute_id in ia.HELD_OUT,
            "total": total,
            "positive_rate": round(counts["positives"] / max(total, 1), 4),
            "agree": counts["agree"],
            "mismatch": counts["mismatch"],
            "unknown": counts["unknown"],
            "missing": counts["missing"],
            "mismatch_rate": round(counts["mismatch"] / max(total, 1), 4),
            "discard_rate": round(dropped / max(total, 1), 4),
        })

    by_tier: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for key in ("total", "agree", "mismatch", "unknown", "missing"):
            by_tier[row["tier"]][key] += row[key]

    tiers = {
        tier: {
            "total": counts["total"],
            "mismatch_rate": round(counts["mismatch"] / max(counts["total"], 1), 4),
            "unknown_rate": round(counts["unknown"] / max(counts["total"], 1), 4),
            "discard_rate": round(
                (counts["mismatch"] + counts["unknown"] + counts["missing"])
                / max(counts["total"], 1), 4),
        }
        for tier, counts in sorted(by_tier.items())
    }

    length_correlation = {}
    for split in ("train", "val"):
        pairs = [
            (len(d["state"]), sum(d["intent_labels"].values()))
            for d in documents if d["split"] == split
        ]
        length_correlation[split] = pearson(pairs)

    return {"per_attribute": rows, "by_tier": tiers,
            "length_vs_true_count": length_correlation}


def pearson(pairs: list[tuple[float, float]]) -> dict[str, float]:
    n = len(pairs)
    if n < 2:
        return {"n": n, "r": 0.0}
    mean_x = sum(p[0] for p in pairs) / n
    mean_y = sum(p[1] for p in pairs) / n
    sxy = sum((p[0] - mean_x) * (p[1] - mean_y) for p in pairs)
    sxx = sum((p[0] - mean_x) ** 2 for p in pairs)
    syy = sum((p[1] - mean_y) ** 2 for p in pairs)
    if sxx <= 0 or syy <= 0:
        return {"n": n, "r": 0.0}
    return {"n": n, "r": round(sxy / (sxx * syy) ** 0.5, 4),
            "mean_chars": round(mean_x, 1), "mean_true": round(mean_y, 2)}


async def main_async(args: argparse.Namespace) -> int:
    from scripts.check_catalog_leak import check, check_intent_catalog

    leak, consistency = check(), check_intent_catalog()
    if not (leak["clean"] and consistency["clean"]):
        print("FAIL: catalogue leak check did not pass. Run scripts/check_catalog_leak.py")
        return 1
    print(f"catalogue leak check: CLEAN ({leak['strings_scanned']} strings)")

    rng = random.Random(args.seed)
    by_name = {d.name: d for d in CATALOG}
    domains = [CATALOG[i % len(CATALOG)] for i in range(args.docs)]
    rng.shuffle(domains)

    n_val = int(args.docs * args.val_fraction)
    plans = []
    for index, domain in enumerate(domains):
        split = "val" if index < n_val else "train"
        plans.append(plan_document(
            index, domain, split, random.Random(args.seed * 1_000_003 + index)
        ))

    rejections: Counter = Counter()
    started = time.time()
    done = 0

    async with LocalLLM(concurrency=args.concurrency) as llm:
        async def gen_worker(plan: dict[str, Any], index: int) -> dict[str, Any] | None:
            nonlocal done
            document = await generate_one(
                llm, by_name[plan["domain"]], plan,
                random.Random(args.seed * 7_919 + index),
                temperature=args.temperature, max_attempts=args.max_attempts,
                rejections=rejections,
                lengths=LONG_LENGTHS if args.long else None,
            )
            done += 1
            if done % 250 == 0:
                rate = done / max(time.time() - started, 1e-9)
                print(f"  gen {done}/{args.docs}  ({rate:.2f} docs/s, "
                      f"~{(args.docs - done) / max(rate, 1e-9) / 60:.1f} min left)", flush=True)
            return document

        results = await asyncio.gather(
            *(gen_worker(p, i) for i, p in enumerate(plans))
        )
        documents = [d for d in results if d is not None]
        gen_elapsed = time.time() - started
        print(f"\ngenerated {len(documents)}/{args.docs} in {gen_elapsed/60:.1f} min "
              f"({len(documents)/max(gen_elapsed,1e-9):.2f} docs/s)")

        verify_started = time.time()
        verify_failures: Counter = Counter()
        verified = 0

        async def verify_worker(document: dict[str, Any], index: int) -> None:
            nonlocal verified
            document["verdicts"] = await verify_one(
                llm, document, random.Random(args.seed * 104_729 + index),
                paraphrase=False, failures=verify_failures,
            )
            verified += 1
            if verified % 250 == 0:
                rate = verified / max(time.time() - verify_started, 1e-9)
                print(f"  verify {verified}/{len(documents)}  ({rate:.2f} docs/s, "
                      f"~{(len(documents) - verified) / max(rate, 1e-9) / 60:.1f} min left)",
                      flush=True)

        await asyncio.gather(
            *(verify_worker(d, i) for i, d in enumerate(documents))
        )
        verify_elapsed = time.time() - verify_started
        print(f"verified {len(documents)} in {verify_elapsed/60:.1f} min")

        # Paraphrase stability on a subsample: how much of the disagreement is the
        # generator failing, and how much is the four question forms not meaning
        # quite the same thing. Reported separately so the 30% discard ceiling in
        # §7.3 fires on generation failure only.
        sample = documents[:args.paraphrase_sample]
        para_failures: Counter = Counter()
        para: list[dict[str, Any]] = []
        for index, document in enumerate(sample):
            verdicts = await verify_one(
                llm, document, random.Random(args.seed * 15_485_863 + index),
                paraphrase=True, failures=para_failures,
            )
            para.append({**document, "verdicts": verdicts})
        print(f"paraphrase re-verification on {len(para)} documents")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for document in documents:
            fh.write(json.dumps(document, ensure_ascii=False) + "\n")

    report = summarise(documents)
    para_report = summarise(para) if para else {}

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator_model": llm.config.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "documents_requested": args.docs,
        "documents_kept": len(documents),
        "documents_by_split": dict(Counter(d["split"] for d in documents)),
        "documents_by_domain": dict(sorted(Counter(d["domain"] for d in documents).items())),
        "generation_elapsed_s": round(gen_elapsed, 1),
        "verification_elapsed_s": round(verify_elapsed, 1),
        "rejections": dict(rejections.most_common()),
        "verification_failures": dict(verify_failures.most_common()),
        "domain_catalog": catalog_summary(),
        "intent_catalog": ia.catalog_summary(),
        "verification": report,
        "paraphrase_stability": {
            "n_documents": len(para),
            "failures": dict(para_failures.most_common()),
            "by_tier": para_report.get("by_tier", {}),
        },
        "catalog_leak": leak,
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n-> {out}")
    print(f"-> {manifest_path}")
    print(f"\nby tier: {json.dumps(report['by_tier'], ensure_ascii=False)}")
    print(f"length vs true-count: {json.dumps(report['length_vs_true_count'], ensure_ascii=False)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=5000)
    parser.add_argument("--out", type=str, default="data/docs_v2.jsonl")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--long", action="store_true",
                        help="generate 600-2200 character states for the length stress test")
    parser.add_argument("--paraphrase-sample", type=int, default=300)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
