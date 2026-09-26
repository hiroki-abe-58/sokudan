"""Corpus planning and the cross-model verification switch (night 3 §2).

The generation run itself needs an LLM and is not tested here. What is tested is
everything decided *before* the first request: which document carries which gold
labels, where the retired attributes land, which model writes and which one checks.
A mistake in any of those is invisible in the generated text and only shows up as a
wrong conclusion three hours later.
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from scripts.build_intent_corpus import (
    build_plans,
    plan_document,
    rejected_pairs,
    resolve_models,
)
from sokudan.config import LocalLLMConfig
from sokudan.data import intent_attributes as ia
from sokudan.data.builders.synthetic import CATALOG

RETIRED_IDS = {a.id for a in ia.RETIRED}


def _legacy_plans(n_docs: int, seed: int, val_fraction: float) -> list[dict]:
    """The planning loop exactly as `main_async` inlined it before `build_plans`."""
    rng = random.Random(seed)
    domains = [CATALOG[i % len(CATALOG)] for i in range(n_docs)]
    rng.shuffle(domains)
    n_val = int(n_docs * val_fraction)
    return [
        plan_document(index, domain, "val" if index < n_val else "train",
                      random.Random(seed * 1_000_003 + index))
        for index, domain in enumerate(domains)
    ]


def _signature(plans: list[dict]) -> list[tuple]:
    return [
        (p["doc_id"], p["domain"], p["split"], tuple(sorted(p["domain_labels"].items())),
         tuple((a.id, label) for a, label in p["intents"]))
        for p in plans
    ]


# --- catalogue --------------------------------------------------------------------


def test_retired_attributes_are_outside_the_training_catalogue():
    # `build_intent_train.py` drops labels it cannot find in BY_ID. If a retired
    # attribute were still there, it would be trained on without anyone deciding to.
    assert len(ia.RETIRED) + len(ia.READMITTED) == 10
    assert ia.READMITTED <= {a.id for a in ia._RETIRED_DEFINITIONS}
    assert ia.READMITTED <= set(ia.BY_ID)
    assert not RETIRED_IDS & set(ia.BY_ID)
    assert RETIRED_IDS <= set(ia.ALL_BY_ID)
    assert all(a.tier == "I" for a in ia.RETIRED)


def test_retired_attributes_are_not_held_out():
    # They are forced onto *training* documents; a held-out one would never be placed.
    assert not RETIRED_IDS & ia.HELD_OUT


def test_retired_exclusion_pairs_resolve_and_block():
    for left, right in ia.RETIRED_EXCLUSIVE_PAIRS:
        assert left in ia.ALL_BY_ID and right in ia.ALL_BY_ID
        assert right in ia.conflicts_with(left)
        assert left in ia.conflicts_with(right)


def test_catalogue_leak_check_covers_retired_attributes():
    from scripts.check_catalog_leak import check, check_intent_catalog

    assert check()["clean"]
    assert check_intent_catalog()["clean"]


# --- planning ---------------------------------------------------------------------


def test_build_plans_reproduces_the_legacy_planning_draw_for_draw():
    # v2/v3/v4 were planned by the inlined loop; the refactor must not change them.
    for seed in (20260921, 20260925):
        assert _signature(build_plans(300, seed, 0.15)) == \
            _signature(_legacy_plans(300, seed, 0.15))


def test_no_retired_attribute_appears_unless_asked_for():
    plans = build_plans(600, 7, 0.15)
    used = {a.id for p in plans for a, _ in p["intents"]}
    assert not used & RETIRED_IDS


def test_each_retired_attribute_is_placed_exactly_n_times_on_training_documents():
    plans = build_plans(1200, 11, 0.15, retired_min=40)
    counts: Counter = Counter()
    for plan in plans:
        for attribute, _ in plan["intents"]:
            if attribute.id in RETIRED_IDS:
                assert plan["split"] == "train"
                counts[attribute.id] += 1
    assert counts == Counter({name: 40 for name in RETIRED_IDS})


def test_a_forced_retired_attribute_is_the_documents_only_tier_i():
    plans = build_plans(1200, 13, 0.15, retired_min=40)
    for plan in plans:
        ids = [a.id for a, _ in plan["intents"]]
        retired_here = [i for i in ids if i in RETIRED_IDS]
        if not retired_here:
            continue
        assert len(retired_here) == 1
        tier_i = [a for a, _ in plan["intents"] if a.tier == "I"]
        assert [a.id for a in tier_i] == retired_here
        # Its exclusion partners are blocked, as an active attribute's would be.
        assert not set(ids) & ia.conflicts_with(retired_here[0])


def test_retired_labels_are_roughly_balanced():
    plans = build_plans(3000, 20260926, 0.15, retired_min=230)
    labels = [label for p in plans for a, label in p["intents"] if a.id in RETIRED_IDS]
    # Counted from the catalogue: RETIRED shrinks as attributes are re-admitted.
    assert len(labels) == 230 * len(ia.RETIRED)
    assert 0.45 < sum(labels) / len(labels) < 0.55


def test_other_documents_are_unchanged_by_the_assignment():
    # The assignment uses its own RNG. Documents that do not receive a retired
    # attribute must be planned exactly as they would be without the flag.
    plain = build_plans(800, 17, 0.15)
    forced = build_plans(800, 17, 0.15, retired_min=20)
    changed = [
        i for i, (a, b) in enumerate(zip(_signature(plain), _signature(forced), strict=True))
        if a != b
    ]
    touched = [i for i, p in enumerate(forced)
               if any(x.id in RETIRED_IDS for x, _ in p["intents"])]
    assert set(changed) <= set(touched)
    assert len(touched) == 20 * len(ia.RETIRED)


def test_too_few_training_documents_is_an_error_not_a_shortfall():
    with pytest.raises(ValueError, match="retired placements"):
        build_plans(100, 1, 0.15, retired_min=20)


def test_retired_attribute_on_a_validation_document_is_refused():
    with pytest.raises(ValueError, match="training documents only"):
        plan_document(0, CATALOG[0], "val", random.Random(0), retired=ia.RETIRED[0])


def test_doc_prefix_namespaces_the_ids():
    plans = build_plans(50, 3, 0.15, doc_prefix="i5")
    assert all(p["doc_id"].startswith("i5-") for p in plans)
    assert len({p["doc_id"] for p in plans}) == 50


# --- models ------------------------------------------------------------------------


BASE = LocalLLMConfig(base_url="http://localhost:11434/v1", model="qwen3:30b", api_key="x")


def test_default_is_same_model_and_says_so():
    gen, ver, cross = resolve_models(BASE, None, None)
    assert gen.model == ver.model == "qwen3:30b"
    assert cross is False


def test_generator_override_alone_still_verifies_with_the_generator():
    gen, ver, cross = resolve_models(BASE, "mistral-small3.2:24b", None)
    assert gen.model == ver.model == "mistral-small3.2:24b"
    assert cross is False


def test_cross_model_keeps_the_endpoint_and_changes_only_the_model():
    gen, ver, cross = resolve_models(BASE, "mistral-small3.2:24b", "qwen3:30b")
    assert (gen.model, ver.model, cross) == ("mistral-small3.2:24b", "qwen3:30b", True)
    assert gen.base_url == ver.base_url == BASE.base_url
    assert gen.api_key == ver.api_key == BASE.api_key


# --- rejected pairs ----------------------------------------------------------------


def test_rejected_pairs_records_every_unconfirmed_pair_with_its_reason():
    retired = ia.RETIRED[0].id
    documents = [{
        "doc_id": "i5-000000", "domain": "x", "split": "train", "state": "本文",
        "intent_labels": {"mentions_deadline": 1, "cites_numbers": 0,
                          "states_gratitude": 1, retired: 0, "mentions_attachment": 1},
        "verdicts": {"mentions_deadline": "はい", "cites_numbers": "はい",
                     "states_gratitude": "判断できない", retired: "はい"},
    }]
    out = {r["attribute"]: r for r in rejected_pairs(documents)}
    assert set(out) == {"cites_numbers", "states_gratitude", retired, "mentions_attachment"}
    assert out["cites_numbers"]["reason"] == "mismatch"
    assert out["states_gratitude"]["reason"] == "unknown"
    assert out["mentions_attachment"]["reason"] == "missing"
    assert out[retired]["retired"] is True and out[retired]["tier"] == "I"
    assert out["cites_numbers"]["retired"] is False


# --- validation --------------------------------------------------------------------


def test_instruction_echoes_are_rejected_anywhere_in_the_body():
    from scripts.build_intent_corpus import validate

    body = "お世話になっております。" * 6
    assert validate(body + "架空管理用帳票を添付します。", []) == "echo:架空"
    assert validate("拝啓 時候の挨拶申し上げます。" + body, []) == "echo:時候の挨拶"
    assert validate(body + "書類を添付します。", []) is None


# --- re-admission rule ------------------------------------------------------------


def test_retired_definitions_keep_all_ten_after_readmission():
    # Reports that compare the original ten across re-tests read RETIRED_DEFINITIONS;
    # RETIRED shrinks every time one is put back, and a report built on it drops rows.
    ids = {a.id for a in ia.RETIRED_DEFINITIONS}
    assert len(ids) == 10
    assert ids == RETIRED_IDS | set(ia.READMITTED)


def test_readmission_rule_is_the_false_to_true_bar_the_attributes_failed():
    # They were retired for false->true >= 0.35; a re-test must clear that same bar.
    assert ia.READMIT_MAX_FALSE_TO_TRUE == 0.35
