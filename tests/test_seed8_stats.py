"""The exact permutation test and bootstrap of docs/seed8.md, on made-up numbers."""

from __future__ import annotations

from math import comb

import numpy as np

from scripts.seed8_stats import bootstrap_ci, exact_permutation_p


def test_enumerates_every_split():
    a = [0.9] * 8
    b = [0.8] * 8
    obs, p, splits = exact_permutation_p(a, b)
    assert splits == comb(16, 8) == 12_870
    assert abs(obs - 0.1) < 1e-12
    # Only the observed split puts all eight high values in `a`.
    assert p == 1 / 12_870


def test_complete_separation_small_case():
    # 3 vs 3: the most extreme split is 1 of C(6, 3) = 20.
    obs, p, splits = exact_permutation_p([3.0, 4.0, 5.0], [0.0, 1.0, 2.0])
    assert splits == 20 and p == 1 / 20


def test_no_effect_gives_large_p():
    rng = np.random.default_rng(0)
    x = list(rng.normal(size=16))
    _, p, _ = exact_permutation_p(x[:8], x[8:])
    assert 0 < p <= 1
    _, p_same, _ = exact_permutation_p([1.0] * 8, [1.0] * 8)
    assert p_same == 1.0  # every split ties the observed statistic


def test_p_is_one_sided():
    _, p_up, _ = exact_permutation_p([3.0, 4.0, 5.0], [0.0, 1.0, 2.0])
    _, p_down, _ = exact_permutation_p([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
    assert p_up == 1 / 20 and p_down == 1.0


def test_bootstrap_ci_is_ordered_and_contains_the_difference():
    rng = np.random.default_rng(1)
    lo, hi = bootstrap_ci([0.9, 0.91, 0.92], [0.8, 0.81, 0.82], rng)
    assert lo <= 0.1 <= hi and lo < hi


def test_holm_step_down():
    from scripts.nf_test import holm

    r = holm({"a": 0.01, "b": 0.04})
    assert r["a"]["reject"] and r["b"]["reject"]          # 0.01 <= 0.025, 0.04 <= 0.05
    assert abs(r["a"]["p_holm"] - 0.02) < 1e-12 and abs(r["b"]["p_holm"] - 0.04) < 1e-12
    r = holm({"a": 0.03, "b": 0.04})
    assert not r["a"]["reject"] and not r["b"]["reject"]  # 0.03 > 0.025 stops everything
    assert abs(r["a"]["p_holm"] - 0.06) < 1e-12 and abs(r["b"]["p_holm"] - 0.06) < 1e-12
