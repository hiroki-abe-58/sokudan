"""The S-tier gate: per attribute, positional attributes exempt (docs/benchmarks.md §9)."""

from __future__ import annotations

from scripts.eval_heldout import s_tier_gate
from sokudan.data import intent_attributes as ia


def _attrs(**aurocs):
    return {name: {"tier": ia.BY_ID[name].tier, "auroc": value}
            for name, value in aurocs.items()}


def test_positional_attributes_are_defined_and_surface_tier():
    assert {"ends_with_question", "includes_greeting"} <= ia.POSITIONAL_ATTRIBUTES
    assert all(ia.BY_ID[n].tier == "S" for n in ia.POSITIONAL_ATTRIBUTES)
    # Detectable wherever it occurs, so not positional.
    assert "uses_bullet_points" not in ia.POSITIONAL_ATTRIBUTES


def test_a_positional_failure_alone_passes():
    ok, failing, exempted, judged = s_tier_gate(
        _attrs(ends_with_question=0.70, uses_bullet_points=0.90))
    assert ok and failing == [] and exempted == ["ends_with_question"]
    assert judged == ["uses_bullet_points"]


def test_each_non_positional_attribute_must_clear_the_bar_on_its_own():
    # A pooled S AUROC could clear 0.85 with one attribute at 0.83; per attribute it fails.
    ok, failing, _, _ = s_tier_gate(
        _attrs(ends_with_question=0.95, uses_bullet_points=0.83))
    assert not ok and failing == ["uses_bullet_points"]


def test_no_judged_attribute_passes_vacuously_and_says_so():
    ok, failing, exempted, judged = s_tier_gate(_attrs(ends_with_question=0.60))
    assert ok and failing == [] and judged == [] and exempted == ["ends_with_question"]


def test_other_tiers_are_ignored():
    ok, _, _, judged = s_tier_gate(_attrs(implies_declining=0.10, uses_bullet_points=0.90))
    assert ok and judged == ["uses_bullet_points"]


def test_missing_auroc_fails():
    ok, failing, _, _ = s_tier_gate(
        {"uses_bullet_points": {"tier": "S", "auroc": None}})
    assert not ok and failing == ["uses_bullet_points"]
