"""Expand the verified document corpus into training rows.

    uv run python scripts/build_intent_train.py

Reads `data/docs_v2.jsonl` (documents plus gold labels plus the verifier's verdicts)
and writes three files:

    data/train_v2.jsonl         training rows, trainable attributes only
    data/val_v2.jsonl           validation rows, trainable attributes only
    data/heldout_val_v2.jsonl   validation rows, held-out attributes only
    data/rejected_v2.jsonl      every pair the verifier disagreed with

Rejected pairs are **kept on disk**. Dropping them silently would make it impossible
to answer, later, whether discarding helped or merely deleted the hard cases -- and
`docs/benchmarks.md` §5 says that is the risk the whole discard step
carries.

Three budgets are enforced rather than hoped for:

* **Per-attribute discard ceiling, 30%.** Above it, the attribute's *definition* is
  wrong and filtering would keep only its easy instances, so the attribute leaves
  training altogether and stays in the report as a diagnostic.
* **40-60% positive per attribute, restored after discarding.** Discarding removes
  negatives far more often than positives, and an unbalanced survivor set teaches a
  prior rather than a reading.
* **Negation views at most 15% of boolean views.** A negated question is a genuine
  schema variant, but a corpus made mostly of them teaches the negation template.
* **Derived booleans at most half the new ones.** The mechanical `choice`/`score`
  rewrites are what did not work on Day 1; they stay as a control, not as bulk.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.check_catalog_leak import BENCH_INSTRUCTIONS, BENCH_LABELS, BENCH_TERMS
from sokudan.data import intent_attributes as ia
from sokudan.data.bool_derive import derive_all
from sokudan.data.builders.synthetic import CATALOG
from sokudan.data.schema_aug import AugmentConfig, LabelledQuestion, augment

YES, UNKNOWN = "はい", "判断できない"
DISCARD_CEILING = 0.30
NEGATION_SHARE = 0.15
POSITIVE_LOW, POSITIVE_HIGH = 0.40, 0.60
SPLIT_GAP_FLAG = 0.10


def load(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def partition(documents: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict]:
    """Split every (document, attribute) pair into kept and rejected."""
    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: dict[str, Counter] = defaultdict(Counter)
    by_split: dict[tuple[str, str], Counter] = defaultdict(Counter)

    for document in documents:
        verdicts = document.get("verdicts") or {}
        for attribute_id, gold in document["intent_labels"].items():
            answer = verdicts.get(attribute_id)
            counts[attribute_id]["total"] += 1
            split_counter = by_split[(attribute_id, document["split"])]
            split_counter["total"] += 1
            if answer is not None and answer != UNKNOWN and (answer == YES) == bool(gold):
                split_counter["agree"] += 1
            record = {
                "doc_id": document["doc_id"], "domain": document["domain"],
                "split": document["split"], "state": document["state"],
                "attribute": attribute_id, "tier": ia.BY_ID[attribute_id].tier,
                "label": gold, "verdict": answer,
            }
            if answer is None:
                counts[attribute_id]["missing"] += 1
                rejected.append({**record, "reason": "missing"})
            elif answer == UNKNOWN:
                counts[attribute_id]["unknown"] += 1
                rejected.append({**record, "reason": "unknown"})
            elif (answer == YES) == bool(gold):
                counts[attribute_id]["agree"] += 1
                kept.append(record)
            else:
                counts[attribute_id]["mismatch"] += 1
                rejected.append({**record, "reason": "mismatch"})

    report = {}
    for attribute_id, counter in counts.items():
        total = counter["total"]
        dropped = total - counter["agree"]
        rates = {}
        for split in ("train", "val"):
            split_counter = by_split.get((attribute_id, split))
            if not split_counter or not split_counter["total"]:
                rates[split] = None
                continue
            rates[split] = round(
                1 - split_counter["agree"] / split_counter["total"], 4
            )
        gap = (
            round(abs(rates["train"] - rates["val"]), 4)
            if rates["train"] is not None and rates["val"] is not None else None
        )
        report[attribute_id] = {
            "tier": ia.BY_ID[attribute_id].tier,
            "held_out": attribute_id in ia.HELD_OUT,
            "total": total,
            "agree": counter["agree"],
            "mismatch": counter["mismatch"],
            "unknown": counter["unknown"],
            "missing": counter["missing"],
            "discard_rate": round(dropped / max(total, 1), 4),
            "discard_rate_train": rates["train"],
            "discard_rate_val": rates["val"],
            "split_gap": gap,
            "n_train": by_split.get((attribute_id, "train"), Counter())["total"],
            "n_val": by_split.get((attribute_id, "val"), Counter())["total"],
        }
    return kept, rejected, report


def rebalance(
    kept: list[dict[str, Any]], rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Restore a 40-60% positive rate per attribute and split, after discarding.

    Discarding is not neutral. The verifier disagrees far more often on negatives --
    a document written to imply nothing still reads as implying something -- so
    dropping mismatches pulls the surviving positive rate above 0.5 and teaches the
    head a prior toward "yes". That is the same mistake Day 1 made from the other
    direction, where `bool` learned "always false" and scored 0.683 on it.

    Subsampling the majority side costs volume and nothing else, which is the right
    currency to pay in.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for pair in kept:
        groups[(pair["attribute"], pair["split"])].append(pair)

    out: list[dict[str, Any]] = []
    report: dict[str, Any] = {}
    for (attribute_id, split), group in sorted(groups.items()):
        positives = [p for p in group if p["label"] == 1]
        negatives = [p for p in group if p["label"] == 0]
        before = len(positives) / max(len(group), 1)
        if not positives or not negatives:
            report[f"{attribute_id}/{split}"] = {
                "before": round(before, 4), "after": round(before, 4),
                "kept": 0, "dropped": len(group), "constant": True,
            }
            continue

        # Largest prefix of the majority side that still lands inside [0.40, 0.60].
        minority, majority = sorted((positives, negatives), key=len)
        limit = int(len(minority) * POSITIVE_HIGH / POSITIVE_LOW)
        keep_majority = min(len(majority), max(len(minority), limit))
        rng.shuffle(majority)
        chosen = minority + majority[:keep_majority]
        after = sum(p["label"] for p in chosen) / len(chosen)
        out.extend(chosen)
        report[f"{attribute_id}/{split}"] = {
            "before": round(before, 4), "after": round(after, 4),
            "kept": len(chosen), "dropped": len(group) - len(chosen), "constant": False,
        }
    return out, report


def intent_views(
    pair: dict[str, Any], n_views: int, rng: random.Random, config: AugmentConfig,
    negation_budget: Counter,
) -> list[dict[str, Any]]:
    """Schema variants of one kept (document, attribute) pair."""
    attribute = ia.BY_ID[pair["attribute"]]
    rows: list[dict[str, Any]] = []

    for view in range(n_views):
        use_negation = (
            view > 0
            and negation_budget["used"] < negation_budget["cap"]
            and rng.random() < 0.35
        )
        if use_negation:
            negation_budget["used"] += 1
            base = LabelledQuestion(attribute.negated_question(), 1 - pair["label"])
        else:
            form = rng.randrange(len(attribute.forms)) if view else 0
            base = LabelledQuestion(attribute.question(form), pair["label"])

        item = base if view == 0 else augment(base, rng, config)
        rows.append({
            "doc_id": pair["doc_id"], "domain": pair["domain"],
            "attribute": pair["attribute"], "kind": "bool",
            "source": "intent", "tier": pair["tier"],
            "negation": use_negation, "variant": view,
            "state": pair["state"],
            "question": item.question.model_dump(mode="json"),
            "label": item.label, "label_text": item.label_text,
            "n_options": 2,
        })
    return rows


def domain_views(
    document: dict[str, Any], n_variants: int, rng: random.Random, config: AugmentConfig
) -> list[dict[str, Any]]:
    by_name = {d.name: d for d in CATALOG}
    domain = by_name[document["domain"]]
    rows: list[dict[str, Any]] = []
    for attribute in domain.attributes:
        base = LabelledQuestion(
            attribute.to_question(), document["domain_labels"][attribute.name]
        )
        for variant in range(n_variants):
            item = base if variant == 0 else augment(
                base, rng, config, surface_forms=attribute.surface_forms
            )
            rows.append({
                "doc_id": document["doc_id"], "domain": document["domain"],
                "attribute": attribute.name, "kind": attribute.kind,
                "source": "domain", "tier": None, "negation": False,
                "variant": variant, "state": document["state"],
                "question": item.question.model_dump(mode="json"),
                "label": item.label, "label_text": item.label_text,
                "n_options": len(item.question.labels),
            })
    return rows


def derived_views(
    documents: list[dict[str, Any]], budget: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Day 1's mechanical booleans, kept as a control and capped."""
    by_name = {d.name: d for d in CATALOG}
    pool: list[dict[str, Any]] = []
    for document in documents:
        domain = by_name[document["domain"]]
        for derived in derive_all(domain, document["domain_labels"], rng):
            pool.append({
                "doc_id": document["doc_id"], "domain": document["domain"],
                "attribute": derived.source_attribute, "kind": "bool",
                "source": f"derived:{derived.kind}", "tier": None,
                "negation": derived.kind == "negation", "variant": 0,
                "state": document["state"],
                "question": derived.question.model_dump(mode="json"),
                "label": derived.label,
                "label_text": derived.question.labels[derived.label],
                "n_options": 2, "schema_id": derived.schema_id,
            })

    # Balance each derived schema toward 40-60% positive before the cap, so the cap
    # does not silently take a skewed slice.
    by_schema: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pool:
        by_schema[row["schema_id"]].append(row)
    balanced: list[dict[str, Any]] = []
    for _schema, group in sorted(by_schema.items()):
        positives = [r for r in group if r["label"] == 1]
        negatives = [r for r in group if r["label"] == 0]
        if not positives or not negatives:
            continue
        keep = min(len(positives), len(negatives))
        rng.shuffle(positives)
        rng.shuffle(negatives)
        balanced.extend(positives[:keep] + negatives[:keep])

    rng.shuffle(balanced)
    return balanced[:budget]


def enforce_bool_share(
    rows: list[dict[str, Any]], target: float, rng: random.Random
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Downsample boolean views until they are `target` of the training set.

    s2a put 75% of its views on `bool` against the 63% that §8.1 planned, and every
    validation metric came out below s1's -- `choice` 0.568 against 0.792, `score`
    RPS 0.201 against 0.149 -- although both had *more* views than s1 did. The
    per-epoch trace says why: nothing degrades mid-run, everything simply crawls.
    Dilution, not interference.

    What gets dropped is ranked by what Day 1 proved worthless. The derived
    membership/threshold booleans go first: they are rewrites of a `choice` or
    `score` label and they are the thing that did not transfer. Intent views are
    thinned only after the derived pool is exhausted, and the domain's own handwritten
    booleans are never touched -- validation scores them, so training has to show them.

    Thinning an intent attribute subsamples its positives and negatives at the same
    rate, so the 40-60% balance restored by `rebalance` survives.
    """
    boolean = [r for r in rows if r["kind"] == "bool"]
    other = [r for r in rows if r["kind"] != "bool"]
    if not boolean or not other:
        return rows, {"applied": False}

    budget = int(len(other) * target / max(1.0 - target, 1e-9))
    before = len(boolean) / len(rows)
    if len(boolean) <= budget:
        return rows, {"applied": False, "share_before": round(before, 4)}

    groups = {
        "derived": [r for r in boolean if r["source"].startswith("derived")],
        "intent": [r for r in boolean if r["source"] == "intent"],
        "domain": [r for r in boolean if r["source"] == "domain"],
    }
    kept_domain = groups["domain"]
    remaining = budget - len(kept_domain)

    # The negation cap is a fraction of the *final* boolean count, so it has to be
    # applied here rather than when the views were made: thinning shrinks the
    # denominator, and a 12% share before downsampling came out at 22% after.
    negation_cap = int(budget * NEGATION_SHARE)
    plain = [r for r in groups["intent"] if not r["negation"]]
    negated = [r for r in groups["intent"] if r["negation"]]
    rng.shuffle(negated)
    negated = negated[:negation_cap]
    intent_pool = plain + negated

    kept_intent = intent_pool
    if remaining < len(intent_pool):
        # Subsample per (attribute, label) so the positive rate does not move.
        buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
        for row in intent_pool:
            buckets[(row["attribute"], row["label"])].append(row)
        rate = max(remaining, 0) / len(intent_pool)
        kept_intent = []
        for _key, bucket in sorted(buckets.items()):
            rng.shuffle(bucket)
            kept_intent.extend(bucket[: max(1, round(len(bucket) * rate))])
        rng.shuffle(kept_intent)
        kept_intent = kept_intent[: max(remaining, 0)]
    remaining -= len(kept_intent)

    kept_derived = groups["derived"]
    if remaining < len(kept_derived):
        rng.shuffle(kept_derived)
        kept_derived = kept_derived[: max(remaining, 0)]

    kept = kept_domain + kept_intent + kept_derived
    out = other + kept
    rng.shuffle(out)
    report = {
        "applied": True,
        "target": target,
        "share_before": round(before, 4),
        "share_after": round(len(kept) / len(out), 4),
        "bool_before": len(boolean),
        "bool_after": len(kept),
        "intent": {"before": len(groups["intent"]), "after": len(kept_intent)},
        "negation_cap": negation_cap,
        "negation_after": sum(1 for r in kept_intent if r["negation"]),
        "derived": {"before": len(groups["derived"]), "after": len(kept_derived)},
        "domain_bool": {"before": len(groups["domain"]), "after": len(kept_domain)},
    }
    return out, report


def row_leak_check(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Row-level check, now reading criteria *values* as well as their keys.

    The old version read only `list(criteria)`, which is the keys. A description
    written into a criteria value was invisible to it, and that is how 718 training
    rows ended up carrying 解約 and 868 carrying 解除 on Day 1.
    """
    instruction_hits: Counter = Counter()
    label_hits: Counter = Counter()
    term_hits: Counter = Counter()

    for row in rows:
        question = row["question"]
        instructions = question.get("instructions", "")
        if instructions in BENCH_INSTRUCTIONS:
            instruction_hits[instructions] += 1
        criteria = question.get("criteria")
        if isinstance(criteria, dict):
            options, descriptions = list(criteria), list(criteria.values())
        else:
            options, descriptions = list(criteria or []), []
        for option in options:
            if option in BENCH_LABELS:
                label_hits[option] += 1
        for text in [instructions, *options, *descriptions,
                     question.get("true_label", ""), question.get("false_label", "")]:
            for term in BENCH_TERMS:
                if term in text:
                    term_hits[term] += 1

    return {
        "instruction_collisions": dict(instruction_hits),
        "label_collisions": dict(label_hits),
        "term_hits": dict(term_hits),
        "clean": not (instruction_hits or label_hits or term_hits),
    }


def write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs", type=str, default="data/docs_v2.jsonl")
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--intent-views", type=int, default=2)
    parser.add_argument("--domain-variants", type=int, default=4)
    parser.add_argument("--derived-ratio", type=float, default=0.5,
                        help="derived boolean views as a fraction of new ones (§8.1)")
    parser.add_argument("--bool-share", type=float, default=None,
                        help="downsample boolean views to this fraction of training (e.g. 0.63)")
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = AugmentConfig()
    documents = load(Path(args.docs))
    print(f"{len(documents)} documents "
          f"({Counter(d['split'] for d in documents)})")

    kept, rejected, report = partition(documents)
    over = {
        name: row for name, row in report.items()
        if row["discard_rate"] > DISCARD_CEILING
    }

    def fmt(value: float | None) -> str:
        return "    -" if value is None else f"{value:5.3f}"

    print(f"\n{'attribute':34s} {'tier':4s} {'held':5s} {'n':>5s} "
          f"{'破棄率':>6s} {'train':>6s} {'val':>6s} {'差':>6s}")
    flagged_gap: list[str] = []
    for name in sorted(report, key=lambda k: (report[k]["tier"], -report[k]["discard_rate"])):
        row = report[name]
        marks = ""
        if row["discard_rate"] > DISCARD_CEILING:
            marks += "  上限超"
        if row["split_gap"] is not None and row["split_gap"] > SPLIT_GAP_FLAG:
            marks += "  分割差"
            flagged_gap.append(name)
        print(f"{name:34s} {row['tier']:4s} {str(row['held_out']):5s} {row['total']:5d} "
              f"{row['discard_rate']:6.3f} {fmt(row['discard_rate_train'])} "
              f"{fmt(row['discard_rate_val'])} {fmt(row['split_gap'])}{marks}")

    if over:
        # An attribute above the ceiling has a definition the generator cannot carry
        # out reliably. Filtering it would keep only its easy instances, so it leaves
        # training entirely and stays in the report as a diagnostic.
        print(f"\n{DISCARD_CEILING:.0%} 超のため学習から除外: {sorted(over)}")
    if flagged_gap:
        print(f"train と val の破棄率が {SPLIT_GAP_FLAG:.0%} 超離れた属性: {sorted(flagged_gap)}")
        print("  val 文書は 8 属性条件・held-out 強制（§5.1）。差はその副作用の大きさ")

    out_dir = Path(args.out_dir)
    write(out_dir / "rejected_v2.jsonl", rejected)

    kept = [pair for pair in kept if pair["attribute"] not in over]
    kept, balance_report = rebalance(kept, rng)
    print(f"\n再バランス後のペア数: {len(kept)}")

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []

    n_bool_target = len(kept) * args.intent_views
    negation_budget = Counter({"cap": int(n_bool_target * NEGATION_SHARE), "used": 0})

    for pair in kept:
        rows = intent_views(pair, args.intent_views, rng, config, negation_budget)
        if pair["attribute"] in ia.HELD_OUT:
            heldout_rows.extend(rows)
        elif pair["split"] == "train":
            train_rows.extend(rows)
        else:
            val_rows.extend(rows)

    for document in documents:
        rows = domain_views(document, args.domain_variants, rng, config)
        (train_rows if document["split"] == "train" else val_rows).extend(rows)

    new_bool_train = sum(1 for r in train_rows if r["source"] == "intent")
    derived_budget = int(new_bool_train * args.derived_ratio)
    derived = derived_views(
        [d for d in documents if d["split"] == "train"], derived_budget, rng
    )
    train_rows.extend(derived)

    train_rows, share_report = (
        enforce_bool_share(train_rows, args.bool_share, rng)
        if args.bool_share else (train_rows, {"applied": False})
    )
    if share_report.get("applied"):
        print(f"\nbool 比率 {share_report['share_before']:.3f} -> "
              f"{share_report['share_after']:.3f}  "
              f"(intent {share_report['intent']['before']}->{share_report['intent']['after']}, "
              f"derived {share_report['derived']['before']}->{share_report['derived']['after']}, "
              f"domain bool {share_report['domain_bool']['after']} 維持)")

    rng.shuffle(train_rows)
    write(out_dir / "train_v2.jsonl", train_rows)
    write(out_dir / "val_v2.jsonl", val_rows)
    write(out_dir / "heldout_val_v2.jsonl", heldout_rows)

    leak = row_leak_check(train_rows + val_rows + heldout_rows)
    train_ids = {r["doc_id"] for r in train_rows}
    val_ids = {r["doc_id"] for r in val_rows} | {r["doc_id"] for r in heldout_rows}

    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": args.docs,
        "documents": len(documents),
        "pairs_kept": len(kept),
        "pairs_rejected": len(rejected),
        "discard_by_attribute": report,
        "over_ceiling": sorted(over),
        "excluded_from_training": sorted(over),
        "split_gap_flagged": sorted(flagged_gap),
        "rebalance": balance_report,
        "pairs_after_rebalance": len(kept),
        "train_views": len(train_rows),
        "val_views": len(val_rows),
        "heldout_val_views": len(heldout_rows),
        "train_by_kind": dict(Counter(r["kind"] for r in train_rows)),
        "train_by_source": dict(Counter(r["source"] for r in train_rows)),
        "train_by_tier": dict(Counter(
            r["tier"] for r in train_rows if r["tier"] is not None)),
        "heldout_by_tier": dict(Counter(r["tier"] for r in heldout_rows)),
        # Counted off the rows that were actually written. Deriving it from the
        # generation-time budget reported 22% after downsampling had shrunk the
        # denominator, which is the number nobody would have checked.
        "negation_views": sum(1 for r in train_rows if r.get("negation")),
        "negation_share_of_bool": round(
            sum(1 for r in train_rows if r.get("negation"))
            / max(sum(1 for r in train_rows if r["kind"] == "bool"), 1), 4),
        "derived_views": len(derived),
        "option_counts": dict(sorted(Counter(r["n_options"] for r in train_rows).items())),
        "document_overlap_between_splits": sorted(train_ids & val_ids),
        "bool_share": share_report,
        "bench_ja_leakage": leak,
    }
    (out_dir / "manifest_v2.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\ntrain {len(train_rows)}  val {len(val_rows)}  held-out val {len(heldout_rows)}")
    print(f"by kind:   {manifest['train_by_kind']}")
    print(f"by source: {manifest['train_by_source']}")
    print(f"held-out by tier: {manifest['heldout_by_tier']}")
    print(f"negation share of bool: {manifest['negation_share_of_bool']}")
    print(f"bench_ja leakage: {'CLEAN' if leak['clean'] else leak}")

    if not leak["clean"]:
        return 1
    if train_ids & val_ids:
        print("FAIL: a document appears in both splits.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
