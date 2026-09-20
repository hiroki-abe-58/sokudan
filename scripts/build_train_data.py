"""Build the synthetic training set (SOKUDAN_SPEC.md §7.4b, §7.2).

    uv run python scripts/build_train_data.py --docs 3000 --variants 4

Each generated document is labelled on every attribute of its domain, and each
labelled attribute is expanded into several randomised schemas (§7.2). That is what
makes the day's budget work: the local endpoint saturates near 3 documents/second, so
generating 24k examples one at a time would take over two hours, while 3k documents
expanded 8 ways takes about 17 minutes and produces a *better* dataset -- §7.2 asks
for several schemas over the same state regardless.

The train/validation split is **by document**, never by example. Splitting by example
would put randomised variants of the same document on both sides and turn validation
into a memorisation check.

`bench_ja` is not touched here. It is the held-out test set (§4.2).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sokudan.data.builders.synthetic import (
    CATALOG,
    SYSTEM_PROMPT,
    Domain,
    banned_words_for,
    build_prompt,
    catalog_summary,
)
from sokudan.data.local_llm import LocalLLM, LocalLLMError
from sokudan.data.schema_aug import AugmentConfig, LabelledQuestion, augment
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS

MIN_CHARS = 60
MAX_CHARS = 1400
META_MARKERS = ["以下", "承知", "```", "本文:", "###", "了解"]


@dataclass
class Document:
    doc_id: str
    domain: str
    state: str
    labels: dict[str, int]
    context: str
    style: str
    length: str


def validate(text: str, banned: list[str]) -> str | None:
    body = text.strip()
    if len(body) < MIN_CHARS:
        return "too_short"
    if len(body) > MAX_CHARS:
        return "too_long"
    for word in banned:
        if word in body:
            return f"leaked:{word}"
    head = body[:40]
    for marker in META_MARKERS:
        if marker in head:
            return f"meta:{marker}"
    return None


async def generate_document(
    llm: LocalLLM,
    domain: Domain,
    doc_id: str,
    rng: random.Random,
    *,
    temperature: float,
    max_attempts: int,
    rejections: Counter,
) -> Document | None:
    chosen = {attribute.name: attribute.sample(rng) for attribute in domain.attributes}
    banned = banned_words_for(domain)

    for attempt in range(max_attempts):
        prompt, meta = build_prompt(domain, chosen, rng)
        try:
            result = await llm.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=900,
            )
        except LocalLLMError as exc:
            rejections[f"llm_error:{type(exc).__name__}"] += 1
            continue

        body = result.text.strip()
        reason = validate(body, banned)
        if reason is None:
            return Document(
                doc_id=doc_id, domain=domain.name, state=body, labels=chosen,
                context=meta["context"], style=meta["style"], length=meta["length"],
            )
        rejections[reason] += 1
        if attempt == max_attempts - 1:
            return None
    return None


def expand(
    documents: list[Document],
    n_variants: int,
    rng: random.Random,
    config: AugmentConfig,
) -> list[dict[str, Any]]:
    """One document -> one example per attribute per schema variant (§7.2)."""
    by_name = {domain.name: domain for domain in CATALOG}
    examples: list[dict[str, Any]] = []

    for document in documents:
        domain = by_name[document.domain]
        for attribute in domain.attributes:
            base = LabelledQuestion(
                attribute.to_question(), document.labels[attribute.name]
            )
            for variant in range(n_variants):
                item = base if variant == 0 else augment(
                    base, rng, config, surface_forms=attribute.surface_forms
                )
                examples.append({
                    "doc_id": document.doc_id,
                    "domain": document.domain,
                    "attribute": attribute.name,
                    "kind": attribute.kind,
                    "variant": variant,
                    "state": document.state,
                    "question": item.question.model_dump(mode="json"),
                    "label": item.label,
                    "label_text": item.label_text,
                    "n_options": len(item.question.labels),
                })
    return examples


def leakage_check(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Assert, by counting, that no bench_ja schema reached the training data.

    The one mistake that would invalidate every generalisation number reported later,
    and the kind that is invisible unless it is checked explicitly.
    """
    bench_labels = set(DEPARTMENTS) | set(URGENCY_LEVELS)
    bench_instructions = {
        "この問い合わせはどの部署が担当すべきか",
        "この依頼の緊急度は",
        "送信者は解約・契約終了を示唆しているか",
    }
    label_hits: Counter = Counter()
    instruction_hits: Counter = Counter()
    for example in examples:
        question = example["question"]
        if question.get("instructions") in bench_instructions:
            instruction_hits[question["instructions"]] += 1
        criteria = question.get("criteria")
        labels = list(criteria) if isinstance(criteria, dict) else (criteria or [])
        for label in labels:
            if label in bench_labels:
                label_hits[label] += 1
    return {
        "bench_label_collisions": dict(label_hits),
        "bench_instruction_collisions": dict(instruction_hits),
        "clean": not label_hits and not instruction_hits,
    }


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    rejections: Counter = Counter()
    started = time.time()
    done = 0

    domains = [CATALOG[i % len(CATALOG)] for i in range(args.docs)]
    rng.shuffle(domains)

    async with LocalLLM(concurrency=args.concurrency) as llm:
        async def worker(index: int, domain: Domain) -> Document | None:
            nonlocal done
            # A per-document RNG keeps label sampling reproducible under concurrency,
            # where completion order is not deterministic.
            document = await generate_document(
                llm, domain, f"syn-{index:06d}", random.Random(args.seed * 1_000_003 + index),
                temperature=args.temperature, max_attempts=args.max_attempts,
                rejections=rejections,
            )
            done += 1
            if done % 200 == 0:
                rate = done / max(time.time() - started, 1e-9)
                remaining = (args.docs - done) / max(rate, 1e-9)
                print(f"  {done}/{args.docs}  ({rate:.2f} docs/s, ~{remaining/60:.1f} min left)",
                      flush=True)
            return document

        results = await asyncio.gather(
            *(worker(i, d) for i, d in enumerate(domains))
        )

    documents = [d for d in results if d is not None]
    elapsed = time.time() - started
    print(f"\ngenerated {len(documents)}/{args.docs} documents in {elapsed:.0f}s")

    # Split by document before expanding, so variants of one document never straddle
    # the split.
    split_rng = random.Random(args.seed + 1)
    shuffled = list(documents)
    split_rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_fraction))
    val_docs, train_docs = shuffled[:n_val], shuffled[n_val:]

    config = AugmentConfig()
    train = expand(train_docs, args.variants, random.Random(args.seed + 2), config)
    val = expand(val_docs, args.variants, random.Random(args.seed + 3), config)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("val", val)):
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} examples -> {path}")

    train_ids = {row["doc_id"] for row in train}
    val_ids = {row["doc_id"] for row in val}
    leak = leakage_check(train + val)

    manifest: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator_model": llm.config.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "documents_requested": args.docs,
        "documents_kept": len(documents),
        "variants_per_attribute": args.variants,
        "elapsed_s": round(elapsed, 1),
        "train_examples": len(train),
        "val_examples": len(val),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "document_overlap_between_splits": sorted(train_ids & val_ids),
        "catalog": catalog_summary(),
        "rejections": dict(rejections.most_common()),
        "examples_by_kind": dict(Counter(r["kind"] for r in train)),
        "examples_by_domain": dict(Counter(r["domain"] for r in train)),
        "option_counts": dict(sorted(Counter(r["n_options"] for r in train).items())),
        "bench_ja_leakage": leak,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nby kind:   {manifest['examples_by_kind']}")
    print(f"option K:  {manifest['option_counts']}")
    print(f"split overlap: {manifest['document_overlap_between_splits'] or 'none'}")
    print(f"bench_ja leakage: {'CLEAN' if leak['clean'] else leak}")
    print(f"manifest -> {out_dir / 'manifest.json'}")

    if not leak["clean"]:
        print("\nFAIL: bench_ja schemas reached the training data.")
        return 1
    if train_ids & val_ids:
        print("\nFAIL: a document appears in both splits.")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=int, default=3000)
    parser.add_argument("--variants", type=int, default=4,
                        help="schema variants per labelled attribute (§7.2)")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=4)
    parser.add_argument("--val-fraction", type=float, default=0.06)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
