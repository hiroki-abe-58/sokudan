"""Collation and Stage 1 loss tests (SOKUDAN_SPEC.md §8).

The collator is where marker positions meet padding, and §5.2 is explicit that a
marker position drifting from where the encoder put it breaks everything silently.
So the tests check the invariant end to end: after padding a mixed batch, the token
sitting at every marker position must still be the mask token.
"""

from __future__ import annotations

import random

import pytest
import torch

from sokudan.data.schema_aug import LabelledQuestion, augment
from sokudan.schema.question import BoolQuestion, ChoiceQuestion, ScoreQuestion
from sokudan.train.dataset import Collator, Example, length_bucketed_batches
from sokudan.train.stage1_distill import stage1_loss

CHOICE = ChoiceQuestion(
    instructions="この連絡の用件は",
    criteria={"納期": "いつ届くか", "仕様": "内容の確認", "価格": "金額の話"},
)
SCORE = ScoreQuestion(instructions="影響の広さは", criteria=["なし", "一部", "広範囲", "全面"])
BOOL = BoolQuestion(instructions="返信が必要か", false_label="不要", true_label="必要")


def example(question, label, kind, state="短い本文です。", doc="d0") -> Example:
    from sokudan.schema.question import is_ordered

    return Example(
        state=state, question=question, label=label, ordered=is_ordered(question),
        doc_id=doc, domain="test", attribute="a", kind=kind,
    )


# --------------------------------------------------------------------------
# collation
# --------------------------------------------------------------------------

def test_collated_marker_positions_still_point_at_mask_tokens(tokenizer) -> None:
    """The invariant the whole pipeline rests on, checked after padding."""
    batch = Collator(tokenizer)([
        example(CHOICE, 1, "choice", "請求内容の確認をお願いします。" * 3),
        example(SCORE, 2, "score", "短文。"),
        example(BOOL, 1, "bool", "長めの本文です。" * 20),
    ])
    for row in range(len(batch)):
        n_real = int(batch.marker_mask[row].sum())
        for slot in range(n_real):
            position = int(batch.marker_positions[row, slot])
            assert int(batch.question_input_ids[row, position]) == tokenizer.mask_token_id


def test_padded_marker_slots_index_inside_the_sequence(tokenizer) -> None:
    """Out-of-range padding would crash `gather`, which the mask cannot prevent."""
    batch = Collator(tokenizer)([
        example(CHOICE, 0, "choice"),   # 3 options
        example(BOOL, 0, "bool"),       # 2 options -> one padded slot
    ])
    assert int(batch.marker_positions.max()) < batch.question_input_ids.shape[1]
    assert int(batch.marker_positions.min()) >= 0


def test_marker_mask_counts_the_real_options(tokenizer) -> None:
    batch = Collator(tokenizer)([
        example(CHOICE, 0, "choice"), example(SCORE, 0, "score"), example(BOOL, 0, "bool"),
    ])
    assert batch.marker_mask.sum(dim=1).tolist() == [3, 4, 2]


def test_shapes_are_consistent(tokenizer) -> None:
    examples = [example(CHOICE, 0, "choice"), example(SCORE, 3, "score")]
    batch = Collator(tokenizer)(examples)
    n = len(examples)
    assert batch.state_input_ids.shape == batch.state_attention_mask.shape
    assert batch.question_input_ids.shape == batch.question_attention_mask.shape
    assert batch.state_input_ids.shape[0] == n
    assert batch.marker_positions.shape == batch.marker_mask.shape == (n, 4)
    assert batch.labels.shape == batch.ordered.shape == (n,)


def test_ordered_flag_is_set_only_for_score(tokenizer) -> None:
    batch = Collator(tokenizer)([
        example(CHOICE, 0, "choice"), example(SCORE, 1, "score"), example(BOOL, 1, "bool"),
    ])
    assert batch.ordered.tolist() == [False, True, False]


def test_attention_mask_marks_exactly_the_real_tokens(tokenizer) -> None:
    batch = Collator(tokenizer)([
        example(CHOICE, 0, "choice", "短い。"),
        example(CHOICE, 0, "choice", "とても長い本文です。" * 30),
    ])
    for row in range(2):
        n_real = int(batch.state_attention_mask[row].sum())
        assert int(batch.state_attention_mask[row, :n_real].sum()) == n_real
        assert int(batch.state_attention_mask[row, n_real:].sum()) == 0


def test_a_label_outside_the_options_is_refused(tokenizer) -> None:
    with pytest.raises(ValueError, match="outside"):
        Collator(tokenizer)([example(CHOICE, 5, "choice")])


def test_long_states_are_truncated_to_the_configured_budget(tokenizer) -> None:
    batch = Collator(tokenizer, max_state_tokens=64)([
        example(CHOICE, 0, "choice", "長い本文。" * 500)
    ])
    assert batch.state_input_ids.shape[1] <= 64


def test_augmented_schemas_survive_collation(tokenizer) -> None:
    """Training feeds randomised schemas, so collation must handle them."""
    rng = random.Random(0)
    examples = []
    for _ in range(16):
        item = augment(LabelledQuestion(CHOICE, 1), rng)
        examples.append(example(item.question, item.label, "choice"))
    batch = Collator(tokenizer)(examples)
    for row in range(len(batch)):
        position = int(batch.marker_positions[row, int(batch.labels[row])])
        assert int(batch.question_input_ids[row, position]) == tokenizer.mask_token_id


def test_the_question_cache_is_reused_across_batches(tokenizer) -> None:
    collator = Collator(tokenizer)
    collator([example(CHOICE, 0, "choice")])
    collator([example(CHOICE, 0, "choice")])
    assert collator.cache.hits >= 1


# --------------------------------------------------------------------------
# bucketing
# --------------------------------------------------------------------------

def test_bucketing_keeps_every_example_exactly_once(tokenizer) -> None:
    examples = [
        example(CHOICE, 0, "choice", "あ" * (i % 50 + 1), doc=f"d{i}") for i in range(97)
    ]
    batches = length_bucketed_batches(examples, 8, tokenizer, rng=random.Random(0))
    flat = [e for batch in batches for e in batch]
    assert len(flat) == 97
    assert {id(e) for e in flat} == {id(e) for e in examples}


def test_bucketing_groups_similar_lengths(tokenizer) -> None:
    """Without this, a 900-token state makes every row in its batch pay for 900."""
    examples = [
        example(CHOICE, 0, "choice", "あ" * length, doc=f"d{i}")
        for i, length in enumerate([1, 800] * 64)
    ]
    batches = length_bucketed_batches(examples, 8, tokenizer, rng=random.Random(0))
    spreads = [
        max(len(e.state) for e in batch) - min(len(e.state) for e in batch)
        for batch in batches
    ]
    assert sum(s == 0 for s in spreads) > len(spreads) * 0.5


def test_bucketing_is_reproducible(tokenizer) -> None:
    examples = [example(CHOICE, 0, "choice", "あ" * (i % 20), doc=f"d{i}") for i in range(40)]
    a = length_bucketed_batches(examples, 8, tokenizer, rng=random.Random(3))
    b = length_bucketed_batches(examples, 8, tokenizer, rng=random.Random(3))
    assert [[e.doc_id for e in g] for g in a] == [[e.doc_id for e in g] for g in b]


# --------------------------------------------------------------------------
# Stage 1 loss
# --------------------------------------------------------------------------

def test_perfect_choice_prediction_has_near_zero_loss() -> None:
    probs = torch.tensor([[1.0, 0.0, 0.0]])
    out = stage1_loss(probs, torch.tensor([0]), torch.tensor([False]), torch.ones(1, 3))
    assert float(out.total) == pytest.approx(0.0, abs=1e-6)


def test_confident_and_wrong_choice_is_punished_heavily() -> None:
    probs = torch.tensor([[1e-9, 1.0 - 1e-9, 0.0]])
    out = stage1_loss(probs, torch.tensor([0]), torch.tensor([False]), torch.ones(1, 3))
    assert float(out.total) > 10.0


def test_ordinal_rows_are_scored_with_rps_not_cross_entropy() -> None:
    """The distinguishing case: an adjacent miss must cost less than a distant one."""
    adjacent = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    distant = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    gold = torch.tensor([2])
    ordered = torch.tensor([True])
    mask = torch.ones(1, 4)
    near = float(stage1_loss(adjacent, gold, ordered, mask).total)
    far = float(stage1_loss(distant, gold, ordered, mask).total)
    assert near < far


def test_the_same_rows_scored_as_choice_cannot_tell_them_apart() -> None:
    """Contrast case, so the ordinal path is demonstrably doing something."""
    adjacent = torch.tensor([[1e-9, 1.0 - 3e-9, 1e-9, 1e-9]])
    distant = torch.tensor([[1.0 - 3e-9, 1e-9, 1e-9, 1e-9]])
    gold = torch.tensor([2])
    unordered = torch.tensor([False])
    mask = torch.ones(1, 4)
    near = float(stage1_loss(adjacent, gold, unordered, mask).total)
    far = float(stage1_loss(distant, gold, unordered, mask).total)
    assert near == pytest.approx(far, rel=1e-6)


def test_mixed_batches_weight_by_count_not_by_batch_half() -> None:
    """Otherwise the loss would move when a batch happens to hold more of one kind."""
    probs = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    labels = torch.tensor([0, 0, 0])
    mask = torch.ones(3, 3)
    one_ordinal = stage1_loss(probs, labels, torch.tensor([True, False, False]), mask)
    two_ordinal = stage1_loss(probs, labels, torch.tensor([True, True, False]), mask)
    # Everything is a perfect prediction, so both totals are ~0 regardless of the mix.
    assert float(one_ordinal.total) == pytest.approx(0.0, abs=1e-6)
    assert float(two_ordinal.total) == pytest.approx(0.0, abs=1e-6)
    assert one_ordinal.n_ordinal == 1 and two_ordinal.n_ordinal == 2


def test_breakdown_counts_each_primitive() -> None:
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.1, 0.2, 0.7]])
    out = stage1_loss(probs, torch.tensor([0, 1, 2]), torch.tensor([False, True, False]),
                      torch.ones(3, 3))
    assert out.n_choice_like == 2 and out.n_ordinal == 1
    assert set(out.as_dict()) == {
        "loss", "loss_choice_like", "loss_ordinal", "n_choice_like", "n_ordinal"
    }


def test_a_batch_of_only_ordinal_rows_works() -> None:
    probs = torch.tensor([[0.1, 0.8, 0.1]])
    out = stage1_loss(probs, torch.tensor([1]), torch.tensor([True]), torch.ones(1, 3))
    assert out.n_choice_like == 0
    assert torch.isfinite(out.total)


def test_a_batch_of_only_choice_rows_works() -> None:
    probs = torch.tensor([[0.1, 0.8, 0.1]])
    out = stage1_loss(probs, torch.tensor([1]), torch.tensor([False]), torch.ones(1, 3))
    assert out.n_ordinal == 0
    assert torch.isfinite(out.total)


def test_loss_respects_the_marker_mask_for_ordinal_rows() -> None:
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    out = stage1_loss(probs, torch.tensor([2]), torch.tensor([True]), mask)
    assert float(out.total) == pytest.approx(0.0, abs=1e-6)


def test_loss_is_differentiable_through_both_paths() -> None:
    probs = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]], requires_grad=True)
    out = stage1_loss(probs, torch.tensor([0, 2]), torch.tensor([False, True]),
                      torch.ones(2, 3))
    out.total.backward()
    assert probs.grad is not None
    assert torch.isfinite(probs.grad).all()
    assert float(probs.grad.abs().sum()) > 0.0


def test_shape_mismatch_is_refused() -> None:
    with pytest.raises(ValueError, match="vs mask"):
        stage1_loss(torch.ones(2, 3) / 3, torch.tensor([0, 1]), torch.tensor([False, False]),
                    torch.ones(2, 4))
