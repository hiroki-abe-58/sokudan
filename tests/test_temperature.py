"""Tests for temperature scaling (SOKUDAN_SPEC.md §1-7: calibration needs tests)."""

from __future__ import annotations

import numpy as np
import pytest

from sokudan.calibration.metrics import ece, nll
from sokudan.calibration.temperature import (
    apply_temperature,
    cross_fit_temperature,
    fit_temperature,
)


def test_temperature_one_is_the_identity() -> None:
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.1, 0.8]])
    np.testing.assert_allclose(apply_temperature(probs, 1.0), probs, atol=1e-9)


def test_high_temperature_flattens_toward_uniform() -> None:
    probs = np.array([[0.98, 0.01, 0.01]])
    out = apply_temperature(probs, 20.0)
    assert out.max() < 0.98
    np.testing.assert_allclose(out.sum(axis=1), 1.0)


def test_low_temperature_sharpens() -> None:
    probs = np.array([[0.5, 0.3, 0.2]])
    out = apply_temperature(probs, 0.2)
    assert out.max() > 0.5


def test_temperature_never_reorders_the_options() -> None:
    """Scaling is monotone, so argmax and the ranking must be preserved."""
    rng = np.random.default_rng(0)
    probs = rng.random((50, 5))
    probs /= probs.sum(axis=1, keepdims=True)
    for temperature in (0.1, 0.5, 2.0, 10.0):
        out = apply_temperature(probs, temperature)
        np.testing.assert_array_equal(out.argsort(axis=1), probs.argsort(axis=1))


def test_rows_stay_normalised() -> None:
    rng = np.random.default_rng(1)
    probs = rng.random((20, 4))
    probs /= probs.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(apply_temperature(probs, 3.7).sum(axis=1), 1.0)


def test_non_positive_temperature_raises() -> None:
    with pytest.raises(ValueError, match="positive"):
        apply_temperature(np.array([[0.5, 0.5]]), 0.0)


def test_fit_recovers_one_for_already_calibrated_data() -> None:
    """Data generated at T=1 should not want rescaling."""
    rng = np.random.default_rng(2)
    n = 4000
    probs = rng.dirichlet([2.0, 2.0, 2.0], size=n)
    labels = np.array([rng.choice(3, p=row) for row in probs])
    assert fit_temperature(probs, labels) == pytest.approx(1.0, abs=0.15)


def test_fit_softens_an_overconfident_model() -> None:
    """Confidence 0.95 with only 60% accuracy must be pushed toward T > 1."""
    n = 600
    probs = np.tile(np.array([[0.95, 0.05]]), (n, 1))
    labels = np.array([0] * int(0.6 * n) + [1] * (n - int(0.6 * n)))
    assert fit_temperature(probs, labels) > 1.0


def test_fitting_reduces_nll_and_ece_on_the_same_data() -> None:
    n = 600
    probs = np.tile(np.array([[0.95, 0.05]]), (n, 1))
    labels = np.array([0] * int(0.6 * n) + [1] * (n - int(0.6 * n)))
    temperature = fit_temperature(probs, labels)
    scaled = apply_temperature(probs, temperature)
    assert nll(scaled, labels) < nll(probs, labels)
    assert ece(scaled, labels) < ece(probs, labels)


def test_cross_fit_never_uses_an_item_to_calibrate_itself() -> None:
    """Two folds, disjoint: every item is scaled by a temperature fitted elsewhere."""
    rng = np.random.default_rng(3)
    n = 400
    probs = np.tile(np.array([[0.9, 0.1]]), (n, 1))
    labels = rng.integers(0, 2, n)
    result = cross_fit_temperature(probs, labels, n_folds=2)
    assert result.n_folds == 2
    assert len(result.temperatures) == 2
    assert result.probs.shape == probs.shape
    np.testing.assert_allclose(result.probs.sum(axis=1), 1.0)


def test_cross_fit_improves_a_badly_calibrated_model() -> None:
    n = 800
    probs = np.tile(np.array([[0.97, 0.03]]), (n, 1))
    labels = np.array([0] * (n // 2) + [1] * (n // 2))
    result = cross_fit_temperature(probs, labels, n_folds=2)
    assert ece(result.probs, labels) < ece(probs, labels)


def test_cross_fit_rejects_impossible_fold_counts() -> None:
    probs = np.array([[0.5, 0.5], [0.5, 0.5]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="n_folds"):
        cross_fit_temperature(probs, labels, n_folds=5)
