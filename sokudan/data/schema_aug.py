"""Schema randomisation (SOKUDAN_SPEC.md §7.2 -- "最重要").

The hypothesis: Laya's zero-shot generalisation fell below a majority-class baseline
because the model learned label *identities* rather than learning to read a schema.
The countermeasure is to make the schema unstable during training, so that the only
thing worth learning is how to read it.

`docs/baseline_ja.md` §6.2 turned that from a hypothesis into a measurement. Across
five schema variants on the same 300 items, `laya-multilingual` put 0, 0, 1, 1 and 0
predictions on the first presented option out of 300 -- and "急がない" drew 0
predictions in first position against 250 in last position. A model that had read the
options could not behave that way. So **option-order shuffling is not optional here**,
and it is the one randomisation that §14.4 must never cut.

Every transform is *label-following*: the gold answer is rewritten into the new option
order, never left pointing at the old one. `test_schema_aug.py` checks that the label
still names the same semantic option after every transform, which is the property the
whole approach rests on.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from sokudan.schema.question import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion


@dataclass(frozen=True)
class LabelledQuestion:
    """A question together with the gold answer *in that question's own ordering*."""

    question: Question
    label: int
    """Index into the question's options. Moves whenever the options move."""

    def __post_init__(self) -> None:
        from sokudan.schema.question import n_markers

        if not 0 <= self.label < n_markers(self.question):
            raise ValueError(
                f"label {self.label} is outside the {n_markers(self.question)} options"
            )

    @property
    def label_text(self) -> str:
        """The option the label currently points at. The invariant under augmentation."""
        return self.question.labels[self.label]


# --------------------------------------------------------------------------
# Surface variation
# --------------------------------------------------------------------------

BOOL_LABEL_PAIRS: list[tuple[str, str]] = [
    ("いいえ", "はい"),
    ("該当しない", "該当する"),
    ("いいえ、当てはまりません", "はい、当てはまります"),
    ("否", "是"),
    ("no", "yes"),
    ("false", "true"),
    ("あてはまらない", "あてはまる"),
]

INSTRUCTION_SUFFIXES: list[str] = [
    "",
    "。",
    "を判定してください。",
    "を1つ選んでください。",
    "は次のうちどれですか。",
    "について答えよ。",
]

# Note the absence of 「その他」. It is the obvious catch-all distractor, and it is
# also one of `bench_ja`'s four department options -- so including it would put a
# bench option string into the training data. `scripts/build_train_data.py` checks
# for exactly this and caught it; the word is not worth the contamination.
DISTRACTOR_LABELS: list[tuple[str, str]] = [
    ("該当なし", "どの区分にも当てはまらない"),
    ("判断できない", "本文からは判断がつかない"),
    ("保留", "追加情報が必要"),
    ("対象外", "この分類の対象ではない"),
    ("未分類", "まだ分類されていない"),
    ("上記以外", "いずれにも当てはまらない"),
    ("不明", "本文に手がかりがない"),
]


@dataclass
class AugmentConfig:
    """Probabilities for each transform. All independent."""

    shuffle_options: float = 0.9
    """§7.2 plus the §6.2 measurement. Kept high on purpose."""
    reverse_score: float = 0.35
    """Reverse an ordinal scale and flip the label. Directly attacks position bias."""
    vary_surface: float = 0.6
    add_distractors: float = 0.35
    drop_options: float = 0.2
    paraphrase_instructions: float = 0.5
    vary_bool_labels: float = 0.7
    max_distractors: int = 3


def augment(
    item: LabelledQuestion,
    rng: random.Random,
    config: AugmentConfig | None = None,
    *,
    surface_forms: dict[str, list[str]] | None = None,
) -> LabelledQuestion:
    """Randomise a schema while carrying the gold label along with it.

    Args:
        item: the question and its gold answer.
        rng: seeded, so a dataset build is reproducible.
        config: transform probabilities.
        surface_forms: optional `{canonical label: [synonyms]}`. A synonym replaces
            the label text; the gold index is unaffected because the option keeps
            its slot -- only its spelling changes.
    """
    config = config or AugmentConfig()
    question = item.question

    if isinstance(question, ChoiceQuestion):
        return _augment_choice(item, rng, config, surface_forms or {})
    if isinstance(question, ScoreQuestion):
        return _augment_score(item, rng, config, surface_forms or {})
    if isinstance(question, BoolQuestion):
        return _augment_bool(item, rng, config)
    return item


def _paraphrase(instructions: str, rng: random.Random, config: AugmentConfig) -> str:
    if rng.random() >= config.paraphrase_instructions:
        return instructions
    base = instructions.rstrip("。.？?")
    return (base + rng.choice(INSTRUCTION_SUFFIXES)) or instructions


def _apply_surface(
    label: str, rng: random.Random, config: AugmentConfig, forms: dict[str, list[str]]
) -> str:
    if rng.random() >= config.vary_surface:
        return label
    options = forms.get(label)
    return rng.choice(options) if options else label


def _augment_choice(
    item: LabelledQuestion,
    rng: random.Random,
    config: AugmentConfig,
    forms: dict[str, list[str]],
) -> LabelledQuestion:
    question = item.question
    assert isinstance(question, ChoiceQuestion)

    pairs = list(question.criteria.items())
    gold_key = pairs[item.label][0]

    # Drop some non-gold options. The gold is never a candidate, so the label
    # always survives; a question needs at least two options left to be meaningful.
    if rng.random() < config.drop_options and len(pairs) > 2:
        droppable = [i for i, (key, _) in enumerate(pairs) if key != gold_key]
        n_drop = rng.randint(1, max(1, min(len(droppable), len(pairs) - 2)))
        for index in sorted(rng.sample(droppable, n_drop), reverse=True):
            pairs.pop(index)

    if rng.random() < config.add_distractors:
        existing = {key for key, _ in pairs}
        pool = [d for d in DISTRACTOR_LABELS if d[0] not in existing]
        rng.shuffle(pool)
        pairs.extend(pool[: rng.randint(1, config.max_distractors)])

    if rng.random() < config.shuffle_options:
        rng.shuffle(pairs)

    # The gold index is *recomputed* from where the gold option ended up, rather
    # than tracked through the permutation. Recomputing cannot drift.
    label = next(i for i, (key, _) in enumerate(pairs) if key == gold_key)

    renamed: dict[str, str] = {}
    gold_renamed = gold_key
    for index, (key, description) in enumerate(pairs):
        new_key = _apply_surface(key, rng, config, forms)
        while new_key in renamed:  # a synonym collided with another option
            new_key = new_key + "・"
        if index == label:
            gold_renamed = new_key
        renamed[new_key] = description

    question = ChoiceQuestion(
        instructions=_paraphrase(question.instructions, rng, config),
        criteria=renamed,
    )
    return LabelledQuestion(question, list(renamed).index(gold_renamed))


def _augment_score(
    item: LabelledQuestion,
    rng: random.Random,
    config: AugmentConfig,
    forms: dict[str, list[str]],
) -> LabelledQuestion:
    """Ordinal scales keep their order -- it carries meaning -- but may be reversed.

    Reversal is the one order change that is still a valid ordinal question, and it is
    the transform that a position-biased model cannot survive: the same semantic level
    appears at the top of the list half the time and at the bottom the other half.
    Option *shuffling* is deliberately absent here; a shuffled ordinal scale is not an
    ordinal scale.
    """
    question = item.question
    assert isinstance(question, ScoreQuestion)

    levels = list(question.criteria)
    label = item.label

    if rng.random() < config.reverse_score:
        levels.reverse()
        label = len(levels) - 1 - label

    renamed = [_apply_surface(level, rng, config, forms) for level in levels]
    for index in range(len(renamed)):  # keep levels distinct after renaming
        while renamed.index(renamed[index]) != index:
            renamed[index] = renamed[index] + "・"

    return LabelledQuestion(
        ScoreQuestion(
            instructions=_paraphrase(question.instructions, rng, config),
            criteria=renamed,
        ),
        label,
    )


def _augment_bool(
    item: LabelledQuestion, rng: random.Random, config: AugmentConfig
) -> LabelledQuestion:
    question = item.question
    assert isinstance(question, BoolQuestion)

    false_label, true_label = question.false_label, question.true_label
    if rng.random() < config.vary_bool_labels:
        false_label, true_label = rng.choice(BOOL_LABEL_PAIRS)

    # No position swap: `BoolQuestion` always renders false first so that column 1
    # is P(true) everywhere (`binary_to_probs`). Only the wording varies.
    return LabelledQuestion(
        BoolQuestion(
            instructions=_paraphrase(question.instructions, rng, config),
            false_label=false_label,
            true_label=true_label,
        ),
        item.label,
    )


def augment_many(
    item: LabelledQuestion,
    n_variants: int,
    rng: random.Random,
    config: AugmentConfig | None = None,
    *,
    surface_forms: dict[str, list[str]] | None = None,
) -> list[LabelledQuestion]:
    """`n_variants` independent augmentations of one labelled question.

    This is the lever that makes the day's data budget work: one generated document
    yields one example per attribute per variant, so a few thousand documents become
    tens of thousands of training examples without another second of generation.
    """
    return [
        augment(item, rng, config, surface_forms=surface_forms) for _ in range(n_variants)
    ]
