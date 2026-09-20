"""Tests for the single metrics implementation (SOKUDAN_SPEC.md §1-7).

`calibration` may not be merged without tests. The values below are hand-computed,
not captured from a previous run, so a regression cannot quietly rewrite them.
"""

from __future__ import annotations

import numpy as np
import pytest

from sokudan.calibration.metrics import (
    accuracy,
    binary_to_probs,
    brier,
    ece,
    macro_f1,
    nll,
    ordinal_mae,
    reliability_bins,
    rps,
    summarize_ordinal,
)

# --------------------------------------------------------------------------
# accuracy / macro-F1
# --------------------------------------------------------------------------

def test_accuracy_counts_argmax_hits() -> None:
    probs = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    labels = np.array([0, 1, 1])
    assert accuracy(probs, labels) == pytest.approx(2 / 3)


def test_macro_f1_perfect_is_one() -> None:
    probs = np.array([[0.9, 0.1], [0.1, 0.9]])
    assert macro_f1(probs, np.array([0, 1])) == pytest.approx(1.0)


def test_macro_f1_ignores_classes_absent_from_gold() -> None:
    """A 3-way schema where class 2 never occurs must not be scored as a 0."""
    probs = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    assert macro_f1(probs, np.array([0, 1])) == pytest.approx(1.0)


# --------------------------------------------------------------------------
# proper scoring rules
# --------------------------------------------------------------------------

def test_nll_of_certain_correct_is_zero() -> None:
    assert nll(np.array([[1.0, 0.0]]), np.array([0])) == pytest.approx(0.0)


def test_nll_matches_hand_computation() -> None:
    probs = np.array([[0.8, 0.2], [0.25, 0.75]])
    expected = -(np.log(0.8) + np.log(0.75)) / 2
    assert nll(probs, np.array([0, 1])) == pytest.approx(expected)


def test_brier_matches_hand_computation() -> None:
    # one item, p=[0.7, 0.3], gold=0 -> (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18
    assert brier(np.array([[0.7, 0.3]]), np.array([0])) == pytest.approx(0.18)


def test_brier_of_certain_correct_is_zero() -> None:
    assert brier(np.array([[0.0, 1.0]]), np.array([1])) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# ECE
# --------------------------------------------------------------------------

def test_ece_is_zero_when_confidence_matches_accuracy() -> None:
    """80% confident on 10 items, exactly 8 correct -> perfectly calibrated."""
    probs = np.tile(np.array([[0.8, 0.2]]), (10, 1))
    labels = np.array([0] * 8 + [1] * 2)
    assert ece(probs, labels) == pytest.approx(0.0, abs=1e-9)


def test_ece_is_one_when_certain_and_always_wrong() -> None:
    probs = np.tile(np.array([[1.0, 0.0]]), (5, 1))
    labels = np.ones(5, dtype=int)
    assert ece(probs, labels) == pytest.approx(1.0)


def test_ece_detects_overconfidence() -> None:
    """90% confident but only 50% correct -> gap of 0.4."""
    probs = np.tile(np.array([[0.9, 0.1]]), (10, 1))
    labels = np.array([0] * 5 + [1] * 5)
    assert ece(probs, labels) == pytest.approx(0.4)


# --------------------------------------------------------------------------
# RPS -- the reason `score` is not treated as plain multiclass (§6.3)
# --------------------------------------------------------------------------

def test_rps_of_perfect_prediction_is_zero() -> None:
    probs = np.array([[0.0, 0.0, 1.0]])
    assert rps(probs, np.array([2])) == pytest.approx(0.0)


def test_rps_punishes_distant_errors_more_than_adjacent_ones() -> None:
    """The whole point of RPS over cross-entropy for ordinal scales."""
    gold = np.array([2])  # middle level of 5
    adjacent = np.array([[0.0, 1.0, 0.0, 0.0, 0.0]])   # off by one
    distant = np.array([[1.0, 0.0, 0.0, 0.0, 0.0]])    # off by two
    assert rps(adjacent, gold) < rps(distant, gold)


def test_cross_entropy_cannot_tell_those_apart() -> None:
    """Contrast case: NLL gives the same penalty, which is why CE is unsuitable."""
    gold = np.array([2])
    adjacent = np.array([[1e-9, 1 - 4e-9, 1e-9, 1e-9, 1e-9]])
    distant = np.array([[1 - 4e-9, 1e-9, 1e-9, 1e-9, 1e-9]])
    assert nll(adjacent, gold) == pytest.approx(nll(distant, gold))


def test_rps_is_normalised_across_different_k() -> None:
    """Worst possible prediction scores 1.0 whether K is 3 or 7."""
    for k in (3, 5, 7):
        probs = np.zeros((1, k))
        probs[0, 0] = 1.0
        assert rps(probs, np.array([k - 1])) == pytest.approx(1.0)


def test_ordinal_mae_argmax_vs_expectation() -> None:
    probs = np.array([[0.5, 0.5, 0.0]])
    # argmax -> 0 (ties resolve to the first), |0 - 1| = 1
    assert ordinal_mae(probs, np.array([1])) == pytest.approx(1.0)
    # expectation -> 0*0.5 + 1*0.5 = 0.5, |0.5 - 1| = 0.5
    assert ordinal_mae(probs, np.array([1]), use_expectation=True) == pytest.approx(0.5)


# --------------------------------------------------------------------------
# helpers and guards
# --------------------------------------------------------------------------

def test_binary_to_probs_puts_true_in_column_one() -> None:
    out = binary_to_probs(np.array([0.25, 0.9]))
    assert out.shape == (2, 2)
    np.testing.assert_allclose(out[:, 1], [0.25, 0.9])
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_binary_metrics_flow_through_the_same_functions() -> None:
    """bool questions reuse the multiclass path -- no second implementation."""
    probs = binary_to_probs(np.array([0.9, 0.1, 0.8]))
    labels = np.array([1, 0, 1])
    assert accuracy(probs, labels) == pytest.approx(1.0)
    assert 0.0 <= ece(probs, labels) <= 1.0


def test_reliability_bins_counts_every_item_once() -> None:
    rng = np.random.default_rng(0)
    p = rng.random((200, 1))
    probs = np.hstack([p, 1 - p])
    probs /= probs.sum(axis=1, keepdims=True)
    labels = rng.integers(0, 2, 200)
    bins = reliability_bins(probs, labels, n_bins=15)
    assert bins.counts.sum() == 200


def test_rows_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        accuracy(np.array([[0.5, 0.2]]), np.array([0]))


def test_labels_must_be_in_range() -> None:
    with pytest.raises(ValueError, match="out of range"):
        accuracy(np.array([[0.5, 0.5]]), np.array([2]))


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        accuracy(np.zeros((0, 3)), np.zeros(0, dtype=int))


def test_summarize_ordinal_returns_every_documented_key() -> None:
    probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])
    out = summarize_ordinal(probs, np.array([0, 1]))
    for key in ("accuracy", "macro_f1", "ece", "brier", "nll",
                "rps", "mae_argmax", "mae_expectation", "mean_confidence"):
        assert key in out, key


# --------------------------------------------------------------------------
# rank-based diagnostics (AUROC / rate-matched accuracy / Spearman)
# --------------------------------------------------------------------------

def test_auroc_perfect_separation_is_one() -> None:
    from sokudan.calibration.metrics import auroc

    assert auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == pytest.approx(1.0)


def test_auroc_inverted_ordering_is_zero() -> None:
    from sokudan.calibration.metrics import auroc

    assert auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == pytest.approx(0.0)


def test_auroc_of_constant_scores_is_half() -> None:
    """All ties carry no ordering information."""
    from sokudan.calibration.metrics import auroc

    assert auroc(np.full(6, 0.5), np.array([0, 1, 0, 1, 0, 1])) == pytest.approx(0.5)


def test_auroc_is_invariant_to_monotone_rescaling() -> None:
    """The point of AUROC: a threshold offset cannot change it."""
    from sokudan.calibration.metrics import auroc

    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, 300)
    scores = rng.random(300) * 0.3 + labels * 0.2
    shifted = np.clip(scores + 0.4, 0, 1)
    assert auroc(shifted, labels) == pytest.approx(auroc(scores, labels))


def test_auroc_requires_both_classes() -> None:
    from sokudan.calibration.metrics import auroc

    with pytest.raises(ValueError, match="undefined"):
        auroc(np.array([0.1, 0.9]), np.array([1, 1]))


def test_rate_matched_accuracy_removes_a_pure_threshold_offset() -> None:
    """A perfectly ranking model sitting at the wrong threshold recovers fully."""
    from sokudan.calibration.metrics import rate_matched_accuracy

    labels = np.array([0] * 7 + [1] * 3)
    scores = np.array([0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.95, 0.96, 0.97])
    accuracy, _ = rate_matched_accuracy(scores, labels)
    assert accuracy == pytest.approx(1.0)
    # At the natural 0.5 threshold every item is called positive: 3/10 correct.
    assert ((scores >= 0.5).astype(int) == labels).mean() == pytest.approx(0.3)


def test_rate_matched_accuracy_predicts_exactly_the_gold_positive_count() -> None:
    from sokudan.calibration.metrics import rate_matched_accuracy

    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, 100)
    accuracy, _ = rate_matched_accuracy(rng.random(100), labels)
    assert 0.0 <= accuracy <= 1.0


def test_spearman_is_one_for_any_increasing_map() -> None:
    from sokudan.calibration.metrics import spearman

    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(x, np.exp(x)) == pytest.approx(1.0)


def test_spearman_is_minus_one_when_reversed() -> None:
    from sokudan.calibration.metrics import spearman

    assert spearman(np.array([1.0, 2, 3, 4]), np.array([4.0, 3, 2, 1])) == pytest.approx(-1.0)


def test_spearman_rejects_a_constant_input() -> None:
    from sokudan.calibration.metrics import spearman

    with pytest.raises(ValueError, match="constant"):
        spearman(np.ones(5), np.arange(5.0))


def test_expected_level_is_the_probability_weighted_mean() -> None:
    from sokudan.calibration.metrics import expected_level

    probs = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.5, 0.0, 0.5]])
    np.testing.assert_allclose(expected_level(probs), [0.0, 2.0, 1.0])
