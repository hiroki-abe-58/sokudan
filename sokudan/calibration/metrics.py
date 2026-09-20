"""Metrics for typed answers with probabilities. **One implementation only** (§2 DRY).

`eval`, `train` and the baseline runner all import from here. If a second copy of
ECE or RPS appears anywhere in this repo, delete it and import this one instead.

Conventions
-----------
* `probs` is `(N, K)`, rows sum to 1. `labels` is `(N,)` of ints in `[0, K)`.
* Ordinal (`score`) metrics assume the K classes are ordered 0..K-1.
* Nothing here rounds or clips silently; a degenerate input raises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def _check(probs: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError(f"probs must be (N, K), got shape {probs.shape}")
    if labels.ndim != 1 or labels.shape[0] != probs.shape[0]:
        raise ValueError(f"labels shape {labels.shape} does not match probs {probs.shape}")
    if probs.shape[0] == 0:
        raise ValueError("empty input")
    if labels.min() < 0 or labels.max() >= probs.shape[1]:
        raise ValueError(f"labels out of range for K={probs.shape[1]}")
    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        raise ValueError(f"probability rows must sum to 1; worst row sums to {row_sums.min()}")
    return probs, labels


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _check(probs, labels)
    return float((probs.argmax(axis=1) == labels).mean())


def macro_f1(probs: np.ndarray, labels: np.ndarray) -> float:
    probs, labels = _check(probs, labels)
    pred = probs.argmax(axis=1)
    n_classes = probs.shape[1]
    scores = []
    for k in range(n_classes):
        tp = float(((pred == k) & (labels == k)).sum())
        fp = float(((pred == k) & (labels != k)).sum())
        fn = float(((pred != k) & (labels == k)).sum())
        if tp == 0 and (fp == 0 or fn == 0) and (labels == k).sum() == 0:
            continue  # class absent from the gold set; excluded rather than scored 0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        scores.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    return float(np.mean(scores)) if scores else 0.0


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the gold class (natural log)."""
    probs, labels = _check(probs, labels)
    return float(-np.log(np.clip(probs[np.arange(len(labels)), labels], EPS, 1.0)).mean())


def brier(probs: np.ndarray, labels: np.ndarray) -> float:
    """Multiclass Brier score: mean squared error against the one-hot gold vector."""
    probs, labels = _check(probs, labels)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 prediction, equal-width bins.

    Bins are on confidence in [0, 1]; empty bins contribute nothing. This is the
    standard top-label ECE, so it is comparable with the numbers model cards quote.
    """
    probs, labels = _check(probs, labels)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        count = int(in_bin.sum())
        if count == 0:
            continue
        total += (count / len(confidence)) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
    return float(total)


def rps(probs: np.ndarray, labels: np.ndarray) -> float:
    """Ranked probability score for ordinal classes (lower is better).

    Mean over items of `sum_k (CDF_pred(k) - CDF_gold(k))^2`, normalised by K-1 so
    that values are comparable across different numbers of levels.
    """
    probs, labels = _check(probs, labels)
    n_classes = probs.shape[1]
    if n_classes < 2:
        raise ValueError("RPS needs at least 2 ordered classes")
    pred_cdf = np.cumsum(probs, axis=1)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(labels)), labels] = 1.0
    gold_cdf = np.cumsum(onehot, axis=1)
    return float((((pred_cdf - gold_cdf) ** 2).sum(axis=1) / (n_classes - 1)).mean())


def ordinal_mae(probs: np.ndarray, labels: np.ndarray, *, use_expectation: bool = False) -> float:
    """Mean absolute error in level units. argmax by default; expectation optionally."""
    probs, labels = _check(probs, labels)
    if use_expectation:
        levels = np.arange(probs.shape[1], dtype=np.float64)
        pred = probs @ levels
    else:
        pred = probs.argmax(axis=1).astype(np.float64)
    return float(np.abs(pred - labels).mean())


@dataclass
class ReliabilityBins:
    """Per-bin aggregates behind a reliability diagram."""

    edges: np.ndarray
    counts: np.ndarray
    mean_confidence: np.ndarray
    mean_accuracy: np.ndarray


def reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> ReliabilityBins:
    probs, labels = _check(probs, labels)
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    counts = np.zeros(n_bins, dtype=np.int64)
    mean_conf = np.full(n_bins, np.nan)
    mean_acc = np.full(n_bins, np.nan)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        in_bin = (confidence > lo) & (confidence <= hi) if lo > 0 else (confidence <= hi)
        counts[i] = int(in_bin.sum())
        if counts[i]:
            mean_conf[i] = confidence[in_bin].mean()
            mean_acc[i] = correct[in_bin].mean()
    return ReliabilityBins(edges, counts, mean_conf, mean_acc)


def binary_to_probs(p_true: np.ndarray) -> np.ndarray:
    """Turn P(true) into the (N, 2) form the metrics above expect: column 1 is True."""
    p_true = np.asarray(p_true, dtype=np.float64).reshape(-1)
    if np.any(p_true < -1e-9) or np.any(p_true > 1 + 1e-9):
        raise ValueError("P(true) must lie in [0, 1]")
    p_true = np.clip(p_true, 0.0, 1.0)
    return np.stack([1.0 - p_true, p_true], axis=1)


def summarize_choice(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    return {
        "accuracy": accuracy(probs, labels),
        "macro_f1": macro_f1(probs, labels),
        "ece": ece(probs, labels, n_bins),
        "brier": brier(probs, labels),
        "nll": nll(probs, labels),
        "mean_confidence": float(np.asarray(probs, dtype=np.float64).max(axis=1).mean()),
    }


def summarize_ordinal(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> dict[str, float]:
    out = summarize_choice(probs, labels, n_bins)
    out["rps"] = rps(probs, labels)
    out["mae_argmax"] = ordinal_mae(probs, labels)
    out["mae_expectation"] = ordinal_mae(probs, labels, use_expectation=True)
    return out


# --------------------------------------------------------------------------
# Threshold-free and rank-based diagnostics.
#
# Accuracy on a binary question conflates two different failures: the model may
# rank items badly, or it may rank them well and sit at the wrong threshold. AUROC
# separates them, so a claim that a model is "broken" can be checked against the
# possibility that it is merely offset.
# --------------------------------------------------------------------------

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks starting at 1, with ties sharing their average rank."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)

    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            if end - start > 1:
                ranks[order[start:end]] = (start + 1 + end) / 2.0
            start = end
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve for a binary question, via the rank identity.

    0.5 means the score carries no ordering information; below 0.5 means the
    ordering is actively inverted. Ties are handled by average ranks, which is the
    same convention as the Mann-Whitney U statistic.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.shape != labels.shape:
        raise ValueError(f"shape mismatch: {scores.shape} vs {labels.shape}")
    if set(np.unique(labels).tolist()) - {0, 1}:
        raise ValueError("labels must be 0/1")

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC is undefined when one class is absent")

    ranks = _average_ranks(scores)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def rate_matched_accuracy(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Accuracy when the threshold is set so the predicted positive rate equals gold's.

    This removes the model's threshold offset and leaves only its ability to rank.
    Returns `(accuracy, threshold)`.
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    n_pos = int((labels == 1).sum())
    if n_pos == 0 or n_pos == len(labels):
        raise ValueError("rate matching is undefined when one class is absent")

    # The n_pos highest-scoring items are called positive. Ties at the boundary are
    # resolved by mergesort order, which is stable and therefore reproducible.
    order = np.argsort(-scores, kind="mergesort")
    pred = np.zeros(len(labels), dtype=np.int64)
    pred[order[:n_pos]] = 1
    threshold = float(scores[order[n_pos - 1]])
    return float((pred == labels).mean()), threshold


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation. Pearson correlation of the average ranks."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    if len(x) < 2:
        raise ValueError("need at least 2 points")

    rx, ry = _average_ranks(x), _average_ranks(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denominator = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denominator == 0:
        raise ValueError("a constant input has no rank correlation")
    return float((rx * ry).sum() / denominator)


def expected_level(probs: np.ndarray) -> np.ndarray:
    """Expected ordinal level per item: sum_k k * p_k."""
    probs = np.asarray(probs, dtype=np.float64)
    return probs @ np.arange(probs.shape[1], dtype=np.float64)
