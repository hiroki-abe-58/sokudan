"""Rebuild the training set with derived boolean schemas. No new generation.

    uv run python scripts/rebuild_train_data.py --variants 4

`docs/benchmarks.md` measured `bool` AUROC at 0.888 on validation and 0.405 on
`bench_ja` -- the ranking inverts on a schema never seen in training. Diagnostics
excluded the alternatives: the state is used (swapping documents moves accuracy by
0.26), the pipeline memorises 100 examples to 0.970, and the P(true) polarity is
consistent through encoding, the baseline and `predict`.

What remains is that the catalogue held only **10** boolean schemas. The model could
recognise those ten rather than learn to read a yes/no question. This rebuild derives
**92** from labels already in hand -- membership questions from `choice`, threshold
questions from `score`, and a template-built negation of every boolean -- without
calling the generator once.

Each boolean schema is then subsampled to a 40-60% positive rate. Membership
questions are 1/K positive by construction, and leaving them that way would teach the
model to answer "no", which is exactly the failure being fixed.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sokudan.data.bool_derive import derive_all, schema_catalog
from sokudan.data.builders.synthetic import CATALOG
from sokudan.data.schema_aug import AugmentConfig, LabelledQuestion, augment
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS

# The bench_ja boolean and anything that reads as the same question. Nothing matching
# these may enter training, however it was derived.
BENCH_BOOL_TERMS = ["解約", "契約終了", "契約の終了", "乗り換え", "他社へ"]
BENCH_INSTRUCTIONS = {
    "この問い合わせはどの部署が担当すべきか",
    "この依頼の緊急度は",
    "送信者は解約・契約終了を示唆しているか",
}
BENCH_LABELS = set(DEPARTMENTS) | set(URGENCY_LEVELS)

TARGET_POSITIVE_LOW = 0.40
TARGET_POSITIVE_HIGH = 0.60


def recover_documents(sources: list[tuple[str, Path]]) -> dict[str, dict[str, Any]]:
    """Rebuild `(doc_id -> state, domain, gold labels)` from the expanded rows.

    Variant 0 is the unaugmented question, so its label is the attribute's own gold
    index. Later variants carry a label into a shuffled option order, which is not
    what a fresh derivation needs.

    The split is passed in rather than read off the filename: `Path("train_v1.jsonl
    .bak").stem` is `"train_v1.jsonl"`, which silently assigned every document to a
    split that did not exist and produced two empty outputs.
    """
    documents: dict[str, dict[str, Any]] = {}
    for split, path in sources:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["variant"] != 0:
                continue
            entry = documents.setdefault(row["doc_id"], {
                "doc_id": row["doc_id"], "domain": row["domain"],
                "state": row["state"], "labels": {},
                "split": split,
            })
            entry["labels"][row["attribute"]] = row["label"]
    return documents


def leak_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    instruction_hits: Counter = Counter()
    term_hits: Counter = Counter()
    label_hits: Counter = Counter()
    for row in rows:
        question = row["question"]
        instructions = question.get("instructions", "")
        if instructions in BENCH_INSTRUCTIONS:
            instruction_hits[instructions] += 1
        for term in BENCH_BOOL_TERMS:
            if term in instructions:
                term_hits[term] += 1
        criteria = question.get("criteria")
        labels = list(criteria) if isinstance(criteria, dict) else (criteria or [])
        for label in labels:
            if label in BENCH_LABELS:
                label_hits[label] += 1
    return {
        "instruction_collisions": dict(instruction_hits),
        "bench_bool_term_hits": dict(term_hits),
        "bench_label_collisions": dict(label_hits),
        "clean": not instruction_hits and not term_hits and not label_hits,
    }


def balance(rows: list[dict[str, Any]], rng: random.Random) -> tuple[list, dict]:
    """Subsample each boolean schema toward a 40-60% positive rate."""
    by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    others: list[dict[str, Any]] = []
    for row in rows:
        if row["kind"] == "bool" and row.get("schema_id"):
            by_schema[row["schema_id"]].append(row)
        else:
            others.append(row)

    kept: list[dict[str, Any]] = []
    report: dict[str, Any] = {"schemas": 0, "dropped": 0, "before": [], "after": []}
    for _schema_id, group in sorted(by_schema.items()):
        positives = [r for r in group if r["label"] == 1]
        negatives = [r for r in group if r["label"] == 0]
        if not positives or not negatives:
            # A schema that is constant on this corpus teaches nothing about reading
            # the question, only about its prior. Drop it entirely.
            report["dropped"] += len(group)
            continue

        rate_before = len(positives) / len(group)
        # Trim the larger side so the smaller is at least 40% of what remains.
        if len(positives) > len(negatives):
            keep_positive = min(len(positives), int(len(negatives) * 1.5))
            positives = rng.sample(positives, keep_positive)
        elif len(negatives) > len(positives):
            keep_negative = min(len(negatives), int(len(positives) * 1.5))
            negatives = rng.sample(negatives, keep_negative)

        merged = positives + negatives
        rng.shuffle(merged)
        kept.extend(merged)
        report["schemas"] += 1
        report["before"].append(round(rate_before, 3))
        report["after"].append(round(len(positives) / len(merged), 3))
        report["dropped"] += len(group) - len(merged)

    return others + kept, report


def build_rows(
    documents: list[dict[str, Any]],
    n_variants: int,
    rng: random.Random,
    config: AugmentConfig,
) -> list[dict[str, Any]]:
    by_name = {domain.name: domain for domain in CATALOG}
    rows: list[dict[str, Any]] = []

    for document in documents:
        domain = by_name[document["domain"]]

        # choice / score keep the original expansion.
        for attribute in domain.attributes:
            if attribute.kind == "bool":
                continue
            base = LabelledQuestion(attribute.to_question(), document["labels"][attribute.name])
            for variant in range(n_variants):
                item = base if variant == 0 else augment(
                    base, rng, config, surface_forms=attribute.surface_forms)
                rows.append({
                    "doc_id": document["doc_id"], "domain": document["domain"],
                    "attribute": attribute.name, "kind": attribute.kind,
                    "schema_id": None, "variant": variant, "state": document["state"],
                    "question": item.question.model_dump(mode="json"),
                    "label": item.label, "label_text": item.label_text,
                    "n_options": len(item.question.labels),
                })

        # bool comes from the derivation instead, with fewer variants each because
        # there are now nine times as many distinct schemas.
        for derived in derive_all(domain, document["labels"], rng):
            base = LabelledQuestion(derived.question, derived.label)
            for variant in range(max(1, n_variants // 2)):
                item = base if variant == 0 else augment(base, rng, config)
                rows.append({
                    "doc_id": document["doc_id"], "domain": document["domain"],
                    "attribute": derived.source_attribute, "kind": "bool",
                    "schema_id": derived.schema_id, "derived_kind": derived.kind,
                    "variant": variant, "state": document["state"],
                    "question": item.question.model_dump(mode="json"),
                    "label": item.label, "label_text": item.label_text,
                    "n_options": 2,
                })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260920)
    args = parser.parse_args()

    started = time.time()
    documents = recover_documents([("train", Path(args.train)), ("val", Path(args.val))])
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for document in documents.values():
        by_split[document["split"]].append(document)
    print(f"recovered {len(documents)} documents "
          f"(train {len(by_split['train'])}, val {len(by_split['val'])})")

    catalog = schema_catalog()
    print(f"derivable bool schemas: {len(catalog)} "
          f"({dict(Counter(v['kind'] for v in catalog.values()))})")

    config = AugmentConfig()
    outputs: dict[str, list[dict[str, Any]]] = {}
    balance_reports: dict[str, Any] = {}
    for split in ("train", "val"):
        rng = random.Random(args.seed + (0 if split == "train" else 1))
        rows = build_rows(by_split[split], args.variants, rng, config)
        rows, report = balance(rows, rng)
        outputs[split] = rows
        balance_reports[split] = report

    everything = outputs["train"] + outputs["val"]
    leak = leak_check(everything)

    bool_rows = [r for r in outputs["train"] if r["kind"] == "bool"]
    distinct_schemas = len({r["schema_id"] for r in bool_rows})
    positive_rate = sum(r["label"] for r in bool_rows) / max(len(bool_rows), 1)

    print()
    print(f"bool schemas in train : {distinct_schemas}")
    print(f"bool examples in train: {len(bool_rows)}")
    print(f"bool positive rate    : {positive_rate:.3f}")
    print(f"derived kinds         : "
          f"{dict(Counter(r.get('derived_kind') for r in bool_rows))}")
    rates = balance_reports["train"]["after"]
    if rates:
        print(f"per-schema positive rate after balancing: "
              f"min {min(rates):.3f} / max {max(rates):.3f} / "
              f"outside 0.40-0.60: {sum(1 for r in rates if not 0.40 <= r <= 0.60)}")
    print(f"by kind (train)       : {dict(Counter(r['kind'] for r in outputs['train']))}")
    print(f"\nbench_ja leakage      : {'CLEAN' if leak['clean'] else leak}")

    if not leak["clean"]:
        print("\nFAIL: bench_ja schemas reached the training data.")
        return 1

    if not outputs["train"] or not outputs["val"]:
        print("FAIL: refusing to write an empty split "
              f"(train {len(outputs['train'])}, val {len(outputs['val'])}).")
        return 1

    out_dir = Path(args.out_dir)
    for split, rows in outputs.items():
        path = out_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} examples -> {path}")

    manifest = {
        "rebuilt_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": "derived from existing gold labels; no new generation",
        "documents": len(documents),
        "bool_schemas_derivable": len(catalog),
        "bool_schemas_in_train": distinct_schemas,
        "bool_examples_train": len(bool_rows),
        "bool_positive_rate_train": positive_rate,
        "balance": balance_reports,
        "by_kind_train": dict(Counter(r["kind"] for r in outputs["train"])),
        "train_examples": len(outputs["train"]),
        "val_examples": len(outputs["val"]),
        "bench_ja_leakage": leak,
        "elapsed_s": round(time.time() - started, 1),
    }
    (out_dir / "manifest_rebuild.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"manifest -> {out_dir / 'manifest_rebuild.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
