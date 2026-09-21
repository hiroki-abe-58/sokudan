"""Build `bench_en` by label-conditioned generation, mirroring `bench_ja`.

    uv run python scripts/build_bench_en.py --n 300 --out data/bench_en.jsonl

Same procedure as `scripts/build_bench_ja.py`: draw the gold labels first, have the
local model write text that matches them, throw away anything that violates the
generation contract, and record the rejection counts so the set can be audited.

One addition over `bench_ja`: every kept item is re-read by the same model at
temperature 0, **without being shown the gold**, and the three verdicts are stored on
the item. Nothing is dropped on a disagreement -- the rate is reported instead, since
for a benchmark the useful number is how often the generator and the reader disagree,
not a filtered set that hides it.

`bench_en` exists to answer one question: is Laya's first-option position bias a
property of Japanese or a property of the model. It is evaluation-only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

from sokudan.data.local_llm import LocalLLM, LocalLLMError
from sokudan.eval.bench_en import (
    DEPARTMENTS,
    FORBIDDEN,
    MAX_CHARS,
    META_MARKERS,
    MIN_CHARS,
    SYSTEM_PROMPT,
    URGENCY_LEVELS,
    VERIFY_SYSTEM,
    BenchItem,
    LabelDraw,
    build_generation_prompt,
    build_verify_prompt,
    draw_labels,
)

_JSON = re.compile(r"\{.*\}", re.S)


def validate(text: str) -> str | None:
    """A rejection reason, or None. Case-insensitive: English labels are capitalised."""
    body = text.strip()
    if len(body) < MIN_CHARS:
        return "too_short"
    if len(body) > MAX_CHARS:
        return "too_long"
    lowered = body.lower()
    for word in FORBIDDEN:
        # Word-boundary match so "Other" does not fire on "another" and "cancel"
        # does not fire inside an unrelated compound.
        if re.search(rf"\b{re.escape(word.lower())}\b", lowered):
            return f"leaked:{word}"
    head = lowered[:60]
    for marker in META_MARKERS:
        if marker in head:
            return f"meta:{marker}"
    return None


async def generate_one(
    llm: LocalLLM, draw: LabelDraw, *, item_id: str, temperature: float,
    max_attempts: int, rejections: Counter,
) -> BenchItem | None:
    for attempt in range(max_attempts):
        try:
            result = await llm.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": build_generation_prompt(draw)}],
                temperature=temperature, max_tokens=900,
            )
        except LocalLLMError as exc:
            rejections[f"llm_error:{type(exc).__name__}"] += 1
            continue
        body = result.text.strip()
        reason = validate(body)
        if reason is None:
            return BenchItem(
                item_id=item_id, state=body, department=draw.department,
                urgency=draw.urgency, churn=draw.churn, industry=draw.industry,
                role=draw.role, style=draw.style, length=draw.length_name,
                channel=draw.channel, generator_model=llm.config.model,
                generator_temperature=temperature,
            )
        rejections[reason] += 1
        if attempt == max_attempts - 1:
            return None
    return None


async def verify_one(llm: LocalLLM, item: BenchItem, failures: Counter) -> None:
    """Blind read-back. Stores the verdicts; drops nothing."""
    try:
        result = await llm.chat(
            [{"role": "system", "content": VERIFY_SYSTEM},
             {"role": "user", "content": build_verify_prompt(item.state)}],
            temperature=0.0, max_tokens=200,
        )
    except LocalLLMError as exc:
        failures[f"llm_error:{type(exc).__name__}"] += 1
        return
    match = _JSON.search(result.text)
    if match is None:
        failures["unparseable"] += 1
        return
    try:
        raw = json.loads(match.group(0))
    except ValueError:
        failures["unparseable"] += 1
        return

    department = str(raw.get("department", "")).strip()
    churn = str(raw.get("churn", "")).strip().lower()
    try:
        urgency = int(raw.get("urgency"))
    except (TypeError, ValueError):
        urgency = -1

    item.verdicts = {
        "department": department, "urgency": str(urgency), "churn": churn,
    }
    item.verified = (
        department == item.department
        and urgency == item.urgency
        and ((churn == "yes") == item.churn or churn == "unclear")
    )


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    draws = [draw_labels(rng) for _ in range(args.n)]
    rejections: Counter = Counter()
    started = time.time()
    done = 0

    async with LocalLLM(concurrency=args.concurrency) as llm:
        async def worker(index: int, draw: LabelDraw) -> BenchItem | None:
            nonlocal done
            item = await generate_one(
                llm, draw, item_id=f"bench_en-{index:04d}",
                temperature=args.temperature, max_attempts=args.max_attempts,
                rejections=rejections,
            )
            done += 1
            if done % 50 == 0:
                rate = done / max(time.time() - started, 1e-9)
                print(f"  gen {done}/{args.n} ({rate:.2f}/s)", flush=True)
            return item

        results = await asyncio.gather(*(worker(i, d) for i, d in enumerate(draws)))
        items = [item for item in results if item is not None]
        gen_elapsed = time.time() - started
        print(f"generated {len(items)}/{args.n} in {gen_elapsed/60:.1f} min")

        verify_failures: Counter = Counter()
        await asyncio.gather(*(verify_one(llm, item, verify_failures) for item in items))
        print(f"verified {len(items)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.to_json(), ensure_ascii=False) + "\n")

    agree = sum(1 for i in items if i.verified)
    per_field = {
        "department": sum(1 for i in items
                          if i.verdicts.get("department") == i.department),
        "urgency": sum(1 for i in items if i.verdicts.get("urgency") == str(i.urgency)),
        "churn": sum(1 for i in items
                     if (i.verdicts.get("churn") == "yes") == i.churn),
        "churn_unclear": sum(1 for i in items if i.verdicts.get("churn") == "unclear"),
    }
    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator_model": llm.config.model,
        "temperature": args.temperature,
        "seed": args.seed,
        "requested": args.n,
        "kept": len(items),
        "elapsed_s": round(gen_elapsed, 1),
        "rejections": dict(rejections.most_common()),
        "verification_failures": dict(verify_failures.most_common()),
        "verification_agreement_all_three": round(agree / max(len(items), 1), 4),
        "verification_agreement_per_field": {
            k: round(v / max(len(items), 1), 4) for k, v in per_field.items()
        },
        "gold_counts": {
            "department": dict(Counter(i.department for i in items)),
            "urgency": dict(Counter(URGENCY_LEVELS[i.urgency] for i in items)),
            "churn_true": sum(1 for i in items if i.churn),
            "churn_false": sum(1 for i in items if not i.churn),
        },
        "departments": list(DEPARTMENTS),
        "urgency_levels": list(URGENCY_LEVELS),
    }
    manifest_path = out.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n-> {out}\n-> {manifest_path}")
    print(f"gold: {manifest['gold_counts']}")
    print(f"verification (all three agree): "
          f"{manifest['verification_agreement_all_three']:.1%}")
    print(f"per field: {manifest['verification_agreement_per_field']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="data/bench_en.jsonl")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=4)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
