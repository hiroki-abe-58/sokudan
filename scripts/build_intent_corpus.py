"""Generate the Day 2 document corpus and verify its labels.

    uv run python scripts/build_intent_corpus.py --docs 5000

Two stages, against the same local model by default:

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

What this does **not** measure is whether the label is true. With one model in both
roles, the verifier is the generator and their errors are correlated; it measures
instruction-following, and `docs/benchmarks.md` has to say so.

`--generator-model` / `--verifier-model` put two different models in the two roles
(cross-model verification). A disagreement then cannot be a model agreeing with its own
habits, which is the Limits entry the same-model setup carries.

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
import dataclasses
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sokudan.config import LocalLLMConfig
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
from sokudan.data.latin import latin_intrusion_rate
from sokudan.data.local_llm import LocalLLM, LocalLLMError

MIN_CHARS = 60
MAX_CHARS = 1600
MAX_CHARS_LONG = 3000
META_MARKERS = ["以下", "承知", "```", "本文:", "###", "了解"]
# Instruction wording copied into the body, anywhere in it. A real document never
# says a document name is 架空 or announces 「時候の挨拶」 instead of writing one; both
# are the generator quoting its own prompt. Measured before adding: qwen3 did this in
# 7 of 20,411 documents, mistral-small3.2 in 2 of 20 -- rare enough to have been
# ignorable with one generator, common enough to matter with the other.
INSTRUCTION_ECHOES = ["架空", "時候の挨拶"]
# Share of a document's characters that are English (or other Latin-script) words
# standing in for Japanese -- acronyms, links, file names, units and product names do
# not count (`sokudan.data.latin`). Above this, the document is regenerated.
#
# 0.5%, proposed 2026-09-24 from `scripts/latin_filter_report.py`: documents average
# ~330 characters, so one short stray word already clears 1%, and this threshold drops
# almost exactly the documents that contain any intrusion at all -- 0.5-0.7% of the
# qwen3 corpora, 2.4% of mistral's. A stray English word marks a document as synthetic,
# and regenerating costs compute, not labels. Not applied to existing corpora.
MAX_LATIN_INTRUSION_RATE = 0.005

VERIFY_SYSTEM = (
    "あなたは日本語の文面を読み、設問に答える判定器です。"
    "JSON だけを出力し、解説は一切書きません。"
)

YES, NO, UNKNOWN = "はい", "いいえ", "判断できない"

HELD_OUT_IMPLICIT = sorted(
    name for name in ia.HELD_OUT if ia.BY_ID[name].tier == "I"
)


def validate(text: str, banned: list[str], ceiling: int = MAX_CHARS,
             max_latin_rate: float = MAX_LATIN_INTRUSION_RATE) -> str | None:
    body = text.strip()
    if len(body) < MIN_CHARS:
        return "too_short"
    if len(body) > ceiling:
        return "too_long"
    for word in banned:
        if word in body:
            return f"leaked:{word}"
    for echo in INSTRUCTION_ECHOES:
        if echo in body:
            return f"echo:{echo}"
    if latin_intrusion_rate(body) > max_latin_rate:
        return "latin_intrusion"
    head = body[:40]
    for marker in META_MARKERS:
        if marker in head:
            return f"meta:{marker}"
    return None


def plan_document(
    index: int, domain: Domain, split: str, rng: random.Random,
    *, doc_prefix: str = "i2", retired: ia.IntentAttribute | None = None,
) -> dict[str, Any]:
    """Pick the gold labels for one document before anything is generated.

    `retired` forces one retired tier-I attribute onto a training document. It takes
    the document's single tier-I slot (`MAX_IMPLICIT_PER_DOC`), and its exclusion
    pairs block their partners exactly as an active attribute's would; the remaining
    slots are drawn from the active catalogue as usual.
    """
    if retired is not None and split == "val":
        raise ValueError("retired attributes are forced onto training documents only; "
                         "validation documents spend their tier-I slot on held-out")
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
            rng, pool=ia.TRAINABLE, count=rng.randint(6, 8),
            forced=[retired] if retired is not None else None,
        )

    intents = [(a, 1 if rng.random() < 0.5 else 0) for a in picks]
    return {
        "doc_id": f"{doc_prefix}-{index:06d}",
        "domain": domain.name,
        "split": split,
        "domain_labels": domain_labels,
        "intents": intents,
    }


def build_plans(
    n_docs: int, seed: int, val_fraction: float, *,
    doc_prefix: str = "i2", retired_min: int = 0,
) -> list[dict[str, Any]]:
    """Every document's gold labels, fixed before any generation.

    With `retired_min=0` this reproduces the plans the Day 2/3 corpora were built from,
    draw for draw. With `retired_min=N`, each retired tier-I attribute is forced onto N
    training documents, chosen by a separate RNG so that the domain order and the
    per-document draws of every other document are unchanged by the assignment.
    """
    rng = random.Random(seed)
    domains = [CATALOG[i % len(CATALOG)] for i in range(n_docs)]
    rng.shuffle(domains)
    n_val = int(n_docs * val_fraction)

    assignment: dict[int, ia.IntentAttribute] = {}
    if retired_min:
        train_indices = list(range(n_val, n_docs))
        need = retired_min * len(ia.RETIRED)
        if need > len(train_indices):
            raise ValueError(f"{need} retired placements need {need} training documents; "
                             f"only {len(train_indices)} exist")
        random.Random(seed * 31 + 7).shuffle(train_indices)
        for k, index in enumerate(train_indices[:need]):
            assignment[index] = ia.RETIRED[k % len(ia.RETIRED)]

    return [
        plan_document(
            index, domain, "val" if index < n_val else "train",
            random.Random(seed * 1_000_003 + index),
            doc_prefix=doc_prefix, retired=assignment.get(index),
        )
        for index, domain in enumerate(domains)
    ]


def resolve_models(
    base: LocalLLMConfig, generator: str | None, verifier: str | None,
) -> tuple[LocalLLMConfig, LocalLLMConfig, bool]:
    """The generator and verifier endpoints, and whether they are different models.

    Both share the base endpoint and key; only the model name changes. An unset
    verifier means the generator verifies its own output -- the Day 2/3 behaviour --
    so a run that forgets the flag is labelled same-model in its manifest rather than
    silently claiming a cross-model check.
    """
    gen = dataclasses.replace(base, model=generator) if generator else base
    ver = dataclasses.replace(base, model=verifier) if verifier else gen
    return gen, ver, ver.model != gen.model


def rejected_pairs(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every (document, attribute) the verifier did not confirm, with the reason.

    Same record shape as `build_intent_train.partition()` writes, so the two files can
    be read by the same code.
    """
    out: list[dict[str, Any]] = []
    for document in documents:
        verdicts = document.get("verdicts") or {}
        for attribute_id, gold in document["intent_labels"].items():
            answer = verdicts.get(attribute_id)
            if answer is None:
                reason = "missing"
            elif answer == UNKNOWN:
                reason = "unknown"
            elif (answer == YES) == bool(gold):
                continue
            else:
                reason = "mismatch"
            out.append({
                "doc_id": document["doc_id"], "domain": document["domain"],
                "split": document["split"], "state": document["state"],
                "attribute": attribute_id, "tier": ia.ALL_BY_ID[attribute_id].tier,
                "retired": attribute_id not in ia.BY_ID,
                "label": gold, "verdict": answer, "reason": reason,
            })
    return out


async def generate_one(
    llm: LocalLLM, domain: Domain, plan: dict[str, Any], rng: random.Random,
    *, temperature: float, max_attempts: int, rejections: Counter,
    lengths: list[tuple[str, str]] | None = None,
    max_latin_rate: float = MAX_LATIN_INTRUSION_RATE,
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
        reason = validate(body, banned, ceiling, max_latin_rate)
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
        attribute = ia.ALL_BY_ID[attribute_id]
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
                per_attribute[attribute_id]["kept_positives"] += int(gold)
            else:
                per_attribute[attribute_id]["mismatch"] += 1
                # Split the mismatch by direction. A gold=false the verifier reads as
                # true is the implication failure -- the generator was told to leave the
                # attribute out and the text carries it anyway, usually because another
                # attribute on the same document entails it. The opposite direction is a
                # plain instruction miss. The 30% discard ceiling cannot tell them apart,
                # so the exclusion pairs are tuned against this number instead.
                per_attribute[attribute_id]["false_to_true" if not gold
                                            else "true_to_false"] += 1
            per_attribute[attribute_id]["total"] += 1
            per_attribute[attribute_id]["positives"] += int(gold)

    rows = []
    for attribute_id in sorted(per_attribute, key=lambda k: (ia.ALL_BY_ID[k].tier, k)):
        counts = per_attribute[attribute_id]
        total = counts["total"]
        dropped = counts["mismatch"] + counts["unknown"] + counts["missing"]
        rows.append({
            "attribute": attribute_id,
            "tier": ia.ALL_BY_ID[attribute_id].tier,
            "held_out": attribute_id in ia.HELD_OUT,
            "total": total,
            "positive_rate": round(counts["positives"] / max(total, 1), 4),
            "agree": counts["agree"],
            "mismatch": counts["mismatch"],
            "unknown": counts["unknown"],
            "missing": counts["missing"],
            "mismatch_rate": round(counts["mismatch"] / max(total, 1), 4),
            "discard_rate": round(dropped / max(total, 1), 4),
            "false_to_true": counts["false_to_true"],
            "true_to_false": counts["true_to_false"],
            # Denominators are the gold sides, not the total: "of the times we told the
            # generator to leave this out, how often did it appear anyway".
            "false_to_true_rate": round(
                counts["false_to_true"] / max(total - counts["positives"], 1), 4),
            "true_to_false_rate": round(
                counts["true_to_false"] / max(counts["positives"], 1), 4),
            # The share of surviving rows that are positive, which is what the
            # rebalance in build_intent_train.py has to pull back to 40-60%.
            "kept_total": counts["agree"],
            "kept_positive_rate": round(
                counts["kept_positives"] / max(counts["agree"], 1), 4),
        })

    by_tier: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for key in ("total", "agree", "mismatch", "unknown", "missing",
                    "false_to_true", "true_to_false"):
            by_tier[row["tier"]][key] += row[key]
        by_tier[row["tier"]]["negatives"] += row["total"] - round(
            row["positive_rate"] * row["total"])

    tiers = {
        tier: {
            "total": counts["total"],
            "mismatch_rate": round(counts["mismatch"] / max(counts["total"], 1), 4),
            "unknown_rate": round(counts["unknown"] / max(counts["total"], 1), 4),
            "discard_rate": round(
                (counts["mismatch"] + counts["unknown"] + counts["missing"])
                / max(counts["total"], 1), 4),
            "false_to_true_rate": round(
                counts["false_to_true"] / max(counts["negatives"], 1), 4),
        }
        for tier, counts in sorted(by_tier.items())
    }

    # Reported per split because a correlation present in only one of them would mean
    # the partition is doing something, and on "all" because that is the number the
    # 0.3 ceiling is about -- deciding it on half the documents doubles the standard
    # error of r for no reason, which is how a 200-document smoke test ends up with
    # train at 0.24 and val at 0.37 from the same generator.
    length_correlation = {}
    for split in ("train", "val", "all"):
        pairs = [
            (len(d["state"]), sum(d["intent_labels"].values()))
            for d in documents if split == "all" or d["split"] == split
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

    by_name = {d.name: d for d in CATALOG}
    plans = build_plans(args.docs, args.seed, args.val_fraction,
                        doc_prefix=args.doc_prefix, retired_min=args.retired_min)
    if args.retired_min:
        placed = Counter(a.id for p in plans for a, _ in p["intents"] if a.id not in ia.BY_ID)
        print(f"retired attributes placed: {dict(sorted(placed.items()))}")

    gen_config, ver_config, cross_model = resolve_models(
        LocalLLMConfig.from_env(), args.generator_model, args.verifier_model,
    )
    print(f"generator: {gen_config.model}  verifier: {ver_config.model}"
          f"{'  (cross-model)' if cross_model else ''}")

    rejections: Counter = Counter()
    started = time.time()
    done = 0

    # Two clients even when they share a model. Generation finishes before verification
    # starts, so a cross-model run swaps the loaded model once, not per request.
    async with LocalLLM(gen_config, concurrency=args.concurrency) as gen, \
            LocalLLM(ver_config, concurrency=args.concurrency) as llm:
        async def gen_worker(plan: dict[str, Any], index: int) -> dict[str, Any] | None:
            nonlocal done
            document = await generate_one(
                gen, by_name[plan["domain"]], plan,
                random.Random(args.seed * 7_919 + index),
                temperature=args.temperature, max_attempts=args.max_attempts,
                rejections=rejections,
                lengths=LONG_LENGTHS if args.long else None,
                max_latin_rate=args.max_latin_rate,
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

    rejected = rejected_pairs(documents)
    if args.rejected_out:
        rejected_path = Path(args.rejected_out)
        with rejected_path.open("w", encoding="utf-8") as fh:
            for record in rejected:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"-> {rejected_path} ({len(rejected)} pairs)")

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator_model": gen_config.model,
        "verifier_model": ver_config.model,
        "cross_model_verification": cross_model,
        "doc_prefix": args.doc_prefix,
        "retired_min": args.retired_min,
        "retired_attributes": [a.id for a in ia.RETIRED] if args.retired_min else [],
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
    parser.add_argument("--generator-model", default=None,
                        help="model for writing documents; defaults to SOKUDAN_LOCAL_LLM_MODEL")
    parser.add_argument("--verifier-model", default=None,
                        help="model for blind verification; defaults to the generator")
    parser.add_argument("--doc-prefix", default="i2",
                        help="doc_id namespace, so a new corpus cannot collide with old ones")
    parser.add_argument("--retired-min", type=int, default=0,
                        help="force each retired tier-I attribute onto this many training "
                             "documents (0 = retired attributes are not used)")
    parser.add_argument("--max-latin-rate", type=float, default=MAX_LATIN_INTRUSION_RATE,
                        help="regenerate documents whose English-word share exceeds this "
                             "(1.0 disables the check)")
    parser.add_argument("--rejected-out", default=None,
                        help="write every unconfirmed (document, attribute) pair here")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
