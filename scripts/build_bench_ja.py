"""Build `bench_ja` by label-conditioned generation (SOKUDAN_SPEC.md §4.2).

    uv run python scripts/build_bench_ja.py --n 300 --out data/bench_ja.jsonl

The gold labels are drawn first; the local LLM writes text to match them. Items that
violate the generation contract (leak a label word, wrong length, meta-commentary)
are rejected and redrawn, and the rejection counts are written to the manifest so the
set can be audited rather than trusted.

`bench_ja` is a held-out test set. It must never be used for training.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from sokudan.data.local_llm import LocalLLM, LocalLLMError
from sokudan.eval.bench_ja import (
    CHURN_TRUE_RATE,
    DEPARTMENT_WEIGHTS,
    DEPARTMENTS,
    SYSTEM_PROMPT,
    URGENCY_LEVELS,
    URGENCY_WEIGHTS,
    BenchItem,
    LabelDraw,
    build_generation_prompt,
    draw_labels,
)

# Words the generator was told not to use. A leak turns the benchmark into keyword
# matching, so a leaked item is thrown away rather than kept.
FORBIDDEN = [*DEPARTMENTS.keys(), *URGENCY_LEVELS, "解約", "緊急"]

# Meta-commentary the model sometimes wraps the answer in.
META_MARKERS = ["以下", "承知", "```", "件名:", "本文:", "###"]

MIN_CHARS = 60
MAX_CHARS = 1200


def validate(text: str) -> str | None:
    """Return a rejection reason, or None if the item is usable."""
    body = text.strip()
    if len(body) < MIN_CHARS:
        return "too_short"
    if len(body) > MAX_CHARS:
        return "too_long"
    for word in FORBIDDEN:
        if word in body:
            return f"leaked:{word}"
    head = body[:40]
    for marker in META_MARKERS:
        if marker in head:
            return f"meta:{marker}"
    return None


async def generate_one(
    llm: LocalLLM,
    draw: LabelDraw,
    *,
    item_id: str,
    temperature: float,
    max_attempts: int,
    rejections: dict[str, int],
) -> BenchItem | None:
    for attempt in range(max_attempts):
        try:
            result = await llm.chat(
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_generation_prompt(draw)},
                ],
                temperature=temperature,
                max_tokens=900,
            )
        except LocalLLMError as exc:
            rejections[f"llm_error:{type(exc).__name__}"] = (
                rejections.get(f"llm_error:{type(exc).__name__}", 0) + 1
            )
            continue

        body = result.text.strip()
        reason = validate(body)
        if reason is None:
            return BenchItem(
                item_id=item_id,
                state=body,
                department=draw.department,
                urgency=draw.urgency,
                churn=draw.churn,
                industry=draw.industry,
                role=draw.role,
                style=draw.style,
                length=draw.length_name,
                channel=draw.channel,
                generator_model=llm.config.model,
                generator_temperature=temperature,
            )
        rejections[reason] = rejections.get(reason, 0) + 1
        if attempt == max_attempts - 1:
            return None
    return None


async def main_async(args: argparse.Namespace) -> int:
    rng = random.Random(args.seed)
    draws = [draw_labels(rng) for _ in range(args.n)]
    rejections: dict[str, int] = {}

    started = time.time()
    done = 0

    async with LocalLLM(concurrency=args.concurrency) as llm:
        async def worker(index: int, draw: LabelDraw) -> BenchItem | None:
            nonlocal done
            item = await generate_one(
                llm, draw,
                item_id=f"bench_ja-{index:04d}",
                temperature=args.temperature,
                max_attempts=args.max_attempts,
                rejections=rejections,
            )
            done += 1
            if done % 20 == 0:
                rate = done / max(time.time() - started, 1e-9)
                print(f"  {done}/{args.n}  ({rate:.2f} items/s)", flush=True)
            return item

        items = await asyncio.gather(*(worker(i, d) for i, d in enumerate(draws)))

    kept = [i for i in items if i is not None]
    elapsed = time.time() - started

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for item in kept:
            fh.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    dept_counts: dict[str, int] = {}
    urg_counts: dict[str, int] = {}
    churn_true = 0
    for item in kept:
        dept_counts[item.department] = dept_counts.get(item.department, 0) + 1
        name = URGENCY_LEVELS[item.urgency]
        urg_counts[name] = urg_counts.get(name, 0) + 1
        churn_true += int(item.churn)

    manifest = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "generator_model": llm.config.model,
        "generator_base_url": llm.config.base_url,
        "temperature": args.temperature,
        "seed": args.seed,
        "requested": args.n,
        "kept": len(kept),
        "dropped": args.n - len(kept),
        "elapsed_s": round(elapsed, 1),
        "rejections": dict(sorted(rejections.items(), key=lambda kv: -kv[1])),
        "target_weights": {
            "department": DEPARTMENT_WEIGHTS,
            "urgency": URGENCY_WEIGHTS,
            "churn_true_rate": CHURN_TRUE_RATE,
        },
        "realised_counts": {
            "department": dept_counts,
            "urgency": urg_counts,
            "churn_true": churn_true,
            "churn_false": len(kept) - churn_true,
        },
        "mean_state_chars": round(sum(len(i.state) for i in kept) / max(len(kept), 1), 1),
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nkept {len(kept)}/{args.n} in {elapsed:.0f}s -> {out_path}")
    print(f"manifest -> {manifest_path}")
    print("realised label counts:", json.dumps(manifest["realised_counts"], ensure_ascii=False))
    print("rejections:", json.dumps(manifest["rejections"], ensure_ascii=False))
    return 0 if len(kept) == args.n else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", type=str, default="data/bench_ja.jsonl")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=4)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
