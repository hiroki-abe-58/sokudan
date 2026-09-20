"""Schema randomisation tests (SOKUDAN_SPEC.md §7.2).

§7.2 calls schema randomisation the most important part of the data work and asks for
one property to be tested above all: **the transform is reversible, in the sense that
the gold label follows the options**. If a shuffle moves an option without moving the
label, every affected training example teaches the wrong answer, and nothing later in
the pipeline would notice.

So the invariant under test is not "the label index is preserved" -- it must *change*
when options move -- but "the label still names the same semantic option".
"""

from __future__ import annotations

import random

import pytest

from sokudan.data.schema_aug import (
    AugmentConfig,
    LabelledQuestion,
    augment,
    augment_many,
)
from sokudan.schema.question import BoolQuestion, ChoiceQuestion, ScoreQuestion

CHOICE = ChoiceQuestion(
    instructions="この文書の区分は",
    criteria={"契約": "契約に関するもの", "見積": "見積に関するもの",
              "納品": "納品に関するもの", "保守": "保守に関するもの"},
)
SCORE = ScoreQuestion(
    instructions="影響範囲は",
    criteria=["影響なし", "一部に影響", "広範囲に影響", "全面停止"],
)
BOOL = BoolQuestion(instructions="決定事項が含まれているか")

ALWAYS = AugmentConfig(
    shuffle_options=1.0, reverse_score=1.0, vary_surface=1.0, add_distractors=1.0,
    drop_options=1.0, paraphrase_instructions=1.0, vary_bool_labels=1.0,
)
NEVER = AugmentConfig(
    shuffle_options=0.0, reverse_score=0.0, vary_surface=0.0, add_distractors=0.0,
    drop_options=0.0, paraphrase_instructions=0.0, vary_bool_labels=0.0,
)


# --------------------------------------------------------------------------
# the central invariant: the label follows the option
# --------------------------------------------------------------------------

@pytest.mark.parametrize("gold", range(4))
@pytest.mark.parametrize("seed", range(25))
def test_choice_label_still_names_the_same_option(gold, seed) -> None:
    item = LabelledQuestion(CHOICE, gold)
    out = augment(item, random.Random(seed), ALWAYS)
    # Surface variation is off for this schema (no synonyms supplied), so the label
    # text itself must be identical after every reordering.
    assert out.label_text == item.label_text


@pytest.mark.parametrize("gold", range(4))
@pytest.mark.parametrize("seed", range(25))
def test_score_label_still_names_the_same_level(gold, seed) -> None:
    item = LabelledQuestion(SCORE, gold)
    out = augment(item, random.Random(seed), ALWAYS)
    assert out.label_text == item.label_text


@pytest.mark.parametrize("seed", range(20))
def test_bool_label_meaning_is_preserved(seed) -> None:
    for gold in (0, 1):
        out = augment(LabelledQuestion(BOOL, gold), random.Random(seed), ALWAYS)
        assert out.label == gold, "bool options never swap position, so the index holds"


def test_the_label_index_really_does_move() -> None:
    """A test that only checked `label_text` could pass on a no-op implementation."""
    moved = False
    for seed in range(50):
        out = augment(LabelledQuestion(CHOICE, 0), random.Random(seed), ALWAYS)
        if out.label != 0:
            moved = True
            break
    assert moved, "shuffling never moved the gold option; the transform is a no-op"


# --------------------------------------------------------------------------
# option order -- the property docs/baseline_ja.md §6.2 measured Laya failing
# --------------------------------------------------------------------------

def test_shuffling_spreads_the_gold_across_every_position() -> None:
    """If the gold always landed in one slot, training would teach position."""
    positions: dict[int, int] = {}
    rng = random.Random(0)
    for _ in range(400):
        out = augment(LabelledQuestion(CHOICE, 1), rng, AugmentConfig(
            shuffle_options=1.0, add_distractors=0.0, drop_options=0.0,
            vary_surface=0.0, paraphrase_instructions=0.0,
        ))
        positions[out.label] = positions.get(out.label, 0) + 1
    assert len(positions) == 4, f"gold only ever appeared at {sorted(positions)}"
    assert min(positions.values()) > 400 / 4 * 0.5


def test_reversing_a_score_flips_the_label_to_the_mirrored_index() -> None:
    item = LabelledQuestion(SCORE, 0)  # 影響なし, the lowest level
    out = augment(item, random.Random(0), AugmentConfig(
        reverse_score=1.0, vary_surface=0.0, paraphrase_instructions=0.0,
    ))
    assert out.question.criteria == list(reversed(SCORE.criteria))
    assert out.label == 3
    assert out.label_text == "影響なし"


def test_score_options_are_never_shuffled_only_reversed() -> None:
    """A shuffled ordinal scale is not an ordinal scale."""
    forward = list(SCORE.criteria)
    backward = list(reversed(forward))
    rng = random.Random(0)
    for _ in range(100):
        out = augment(LabelledQuestion(SCORE, 2), rng, ALWAYS)
        assert list(out.question.criteria) in (forward, backward)


# --------------------------------------------------------------------------
# distractors and dropping
# --------------------------------------------------------------------------

def test_distractors_are_added_without_disturbing_the_label() -> None:
    item = LabelledQuestion(CHOICE, 2)
    out = augment(item, random.Random(3), AugmentConfig(
        add_distractors=1.0, shuffle_options=1.0, drop_options=0.0,
        vary_surface=0.0, paraphrase_instructions=0.0,
    ))
    assert out.question.n_options > CHOICE.n_options
    assert out.label_text == item.label_text


def test_dropping_never_removes_the_gold_option() -> None:
    for gold in range(4):
        for seed in range(30):
            out = augment(LabelledQuestion(CHOICE, gold), random.Random(seed),
                          AugmentConfig(drop_options=1.0, add_distractors=0.0,
                                        vary_surface=0.0, paraphrase_instructions=0.0))
            assert out.label_text == CHOICE.labels[gold]


def test_dropping_leaves_at_least_two_options() -> None:
    rng = random.Random(0)
    for _ in range(100):
        out = augment(LabelledQuestion(CHOICE, 0), rng,
                      AugmentConfig(drop_options=1.0, add_distractors=0.0))
        assert out.question.n_options >= 2


# --------------------------------------------------------------------------
# surface variation
# --------------------------------------------------------------------------

def test_surface_forms_rename_the_option_without_moving_the_label() -> None:
    forms = {"契約": ["ご契約", "契約関連", "コントラクト"]}
    item = LabelledQuestion(CHOICE, 0)
    seen = set()
    rng = random.Random(0)
    for _ in range(60):
        out = augment(item, rng, AugmentConfig(
            vary_surface=1.0, shuffle_options=1.0, add_distractors=0.0,
            drop_options=0.0, paraphrase_instructions=0.0,
        ), surface_forms=forms)
        seen.add(out.label_text)
        assert out.label_text in {"契約", *forms["契約"]}
    assert len(seen) > 1, "surface variation never fired"


def test_renaming_never_produces_duplicate_option_labels() -> None:
    """Two options collapsing into one label would make the schema ambiguous."""
    forms = {"契約": ["同じ"], "見積": ["同じ"], "納品": ["同じ"], "保守": ["同じ"]}
    rng = random.Random(0)
    for _ in range(50):
        out = augment(LabelledQuestion(CHOICE, 1), rng, ALWAYS, surface_forms=forms)
        assert len(set(out.question.labels)) == out.question.n_options


def test_score_renaming_keeps_levels_distinct() -> None:
    forms = {level: ["同一"] for level in SCORE.criteria}
    rng = random.Random(0)
    for _ in range(50):
        out = augment(LabelledQuestion(SCORE, 2), rng, ALWAYS, surface_forms=forms)
        assert len(set(out.question.criteria)) == len(out.question.criteria)


def test_bool_label_pairs_stay_false_then_true() -> None:
    """Column 1 is P(true) everywhere; a swapped pair would silently invert it."""
    rng = random.Random(0)
    for _ in range(50):
        out = augment(LabelledQuestion(BOOL, 1), rng, ALWAYS)
        assert out.question.labels[1] == out.question.true_label
        assert out.label_text == out.question.true_label


# --------------------------------------------------------------------------
# instructions, determinism, batching
# --------------------------------------------------------------------------

def test_instructions_are_paraphrased_but_keep_their_stem() -> None:
    rng = random.Random(0)
    variants = {
        augment(LabelledQuestion(CHOICE, 0), rng,
                AugmentConfig(paraphrase_instructions=1.0, shuffle_options=0.0,
                              add_distractors=0.0, drop_options=0.0, vary_surface=0.0)
                ).question.instructions
        for _ in range(60)
    }
    assert len(variants) > 1
    assert all("この文書の区分は" in v for v in variants)


def test_nothing_changes_when_every_probability_is_zero() -> None:
    out = augment(LabelledQuestion(CHOICE, 2), random.Random(0), NEVER)
    assert out.question == CHOICE
    assert out.label == 2


def test_augmentation_is_reproducible_from_the_seed() -> None:
    a = augment(LabelledQuestion(CHOICE, 1), random.Random(42), ALWAYS)
    b = augment(LabelledQuestion(CHOICE, 1), random.Random(42), ALWAYS)
    assert a.question == b.question and a.label == b.label


def test_augment_many_produces_distinct_variants_that_all_keep_the_label() -> None:
    item = LabelledQuestion(CHOICE, 3)
    variants = augment_many(item, 12, random.Random(1), ALWAYS)
    assert len(variants) == 12
    assert all(v.label_text == item.label_text for v in variants)
    assert len({tuple(v.question.labels) for v in variants}) > 1


def test_augmented_questions_still_validate_and_encode(tokenizer) -> None:
    """An augmented schema has to survive the real encoder, not just the dataclass."""
    from sokudan.encoding.question import encode_question

    rng = random.Random(0)
    for base, gold in ((CHOICE, 1), (SCORE, 2), (BOOL, 1)):
        for _ in range(10):
            out = augment(LabelledQuestion(base, gold), rng, ALWAYS)
            encoded = encode_question(out.question, tokenizer)
            assert encoded.n_markers == len(out.question.labels)
            assert 0 <= out.label < encoded.n_markers
            assert encoded.input_ids[encoded.marker_positions[out.label]] == \
                tokenizer.mask_token_id


def test_label_out_of_range_is_refused() -> None:
    with pytest.raises(ValueError, match="outside"):
        LabelledQuestion(CHOICE, 4)
