"""The two-sided exact permutation test of docs/regression_check.md."""

from __future__ import annotations

import itertools

import numpy as np

from scripts.regression_eval import exact_permutation_p_two_sided
from scripts.seed8_stats import exact_permutation_p


def test_complete_separation_gives_two_extreme_splits():
    # 8 vs 8 with every a above every b: only the observed split and its mirror are as
    # extreme, so p = 2 / 12,870 either way round.
    a = [1.0 + i for i in range(8)]
    b = [-1.0 - i for i in range(8)]
    for x, y in ((a, b), (b, a)):
        diff, p, splits = exact_permutation_p_two_sided(x, y)
        assert splits == 12_870
        assert p == 2 / 12_870
        assert np.sign(diff) == np.sign(np.mean(x) - np.mean(y))


def test_symmetric_in_the_order_of_the_groups():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=8).tolist(), rng.normal(size=8).tolist()
    d1, p1, _ = exact_permutation_p_two_sided(a, b)
    d2, p2, _ = exact_permutation_p_two_sided(b, a)
    assert p1 == p2 and d1 == -d2


def test_matches_brute_force_and_bounds_the_one_sided_p():
    rng = np.random.default_rng(7)
    a, b = rng.normal(0.3, 1, size=5).tolist(), rng.normal(size=5).tolist()
    _, p, _ = exact_permutation_p_two_sided(a, b)
    pooled = a + b
    observed = abs(np.mean(a) - np.mean(b))
    stats = [abs(np.mean([pooled[i] for i in idx])
                 - np.mean([pooled[i] for i in range(10) if i not in idx]))
             for idx in itertools.combinations(range(10), 5)]
    assert p == sum(s >= observed - 1e-12 for s in stats) / len(stats)
    _, one_sided, _ = exact_permutation_p(a, b)
    assert one_sided <= p <= 2 * one_sided + 1e-12


def test_monte_carlo_p_is_close_to_the_exact_p_and_counts_the_observed_split():
    from scripts.regression16_eval import monte_carlo_p

    rng = np.random.default_rng(11)
    a, b = rng.normal(0.8, 1, size=6).tolist(), rng.normal(size=6).tolist()
    _, exact, _ = exact_permutation_p_two_sided(a, b)
    _, approx, n = monte_carlo_p(a, b, n=20_000, seed=1)
    assert n == 20_000 and abs(approx - exact) < 0.01
    # complete separation of 16 vs 16: no relabelling reaches it, p = 1 / (1 + n)
    _, p, n = monte_carlo_p([1.0 + i for i in range(16)], [-1.0 - i for i in range(16)],
                            n=1_000, seed=2)
    assert p == 1 / 1_001


def test_monte_carlo_p_is_reproducible_under_its_seed():
    from scripts.regression16_eval import monte_carlo_p

    a, b = [0.1, 0.4, 0.35, 0.2], [0.3, 0.25, 0.5, 0.45]
    assert monte_carlo_p(a, b, n=500, seed=7) == monte_carlo_p(a, b, n=500, seed=7)
