"""Temperature scaling (SOKUDAN_SPEC.md §8 Stage 2).

One temperature per `(question type, option count)` bucket, fitted by minimising NLL
on held-out data. Scaling is applied in log space: `softmax(log(p) / T)`.

`T > 1` softens an over-confident model, `T < 1` sharpens an under-confident one.
`T = 1` leaves the distribution unchanged.

This module does temperature and nothing else (§2, single responsibility).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from sokudan.calibration.metrics import EPS, nll

# Wide enough to cover a badly miscalibrated model without running off to infinity
# on a bucket that happens to be perfectly separable.
T_BOUNDS = (0.05, 20.0)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale a probability matrix by `temperature` in log space."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    logits = np.log(np.clip(np.asarray(probs, dtype=np.float64), EPS, None)) / temperature
    logits -= logits.max(axis=1, keepdims=True)  # stabilise before exponentiating
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probs: np.ndarray, labels: np.ndarray) -> float:
    """Find the temperature minimising NLL on this data."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) == 0:
        raise ValueError("cannot fit a temperature on an empty set")

    result = minimize_scalar(
        lambda t: nll(apply_temperature(probs, t), labels),
        bounds=T_BOUNDS,
        method="bounded",
        options={"xatol": 1e-4},
    )
    return float(result.x)


@dataclass
class CrossFitResult:
    """Temperatures fitted out-of-fold, plus the probabilities they produced."""

    probs: np.ndarray
    temperatures: list[float]
    n_folds: int


def cross_fit_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    n_folds: int = 2,
    seed: int = 20260920,
) -> CrossFitResult:
    """Calibrate every item using a temperature fitted **without** that item.

    Fitting and evaluating a temperature on the same 300 items would flatter the
    model. Cross-fitting gives a temperature-scaled number that is still honest,
    which is the comparison Laya's own model card asks for ("refit one temperature
    per (question type, option count) on held-out data before trusting these").
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    if n_folds < 2 or n_folds > n:
        raise ValueError(f"n_folds must be in [2, {n}], got {n_folds}")

    order = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(order, n_folds)

    out = np.empty_like(probs)
    temperatures: list[float] = []
    for fold in folds:
        mask = np.ones(n, dtype=bool)
        mask[fold] = False
        temperature = fit_temperature(probs[mask], labels[mask])
        temperatures.append(temperature)
        out[fold] = apply_temperature(probs[fold], temperature)

    return CrossFitResult(probs=out, temperatures=temperatures, n_folds=n_folds)
