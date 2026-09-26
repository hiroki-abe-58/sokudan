"""docs/regression_check.md: score the new-code v0.1 seeds 0-7 and run the pre-registered test.

    uv run python scripts/regression_eval.py

Collects per-row predictions for `runs/v01new_seed<s>/model.pt`
(`runs/ens/probs_v01new_seed<s>.npz`, the scoring of `scripts/soup_eval.py`), scores them
with `soup_eval.Scorer`, checks M1 / M1m against the stored evaluation outputs, and compares
them with the old v0.1 seeds 0-7 (the `scores` of `runs/soup/results.json`, same scorer):

- **Main test**: M1m, two-sided exact permutation test, new vs old, unpaired 8 vs 8
  (C(16, 8) = 12,870 splits), alpha 0.05; statistic mean(new) - mean(old).
- **Description**: every other metric's mean difference and a bootstrap 95% interval
  (10,000 draws, numpy seed 0); and new 0-7 against the new-code seeds 8-15.

Writes `runs/regression/results.json`.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401
from scripts.soup_eval import METRICS, PROBS, Scorer, collect, load_data

ALPHA = 0.05
BOOT_N = 10_000
NEW = [f"v01new_seed{s}" for s in range(8)]
OLD = [f"v01_seed{s}" for s in range(8)]
LATER = [f"v01_seed{s}" for s in range(8, 16)]


def exact_permutation_p_two_sided(a: list[float], b: list[float]) -> tuple[float, float, int]:
    """Two-sided p for |mean(a) - mean(b)|, over every relabelling (observed split included).

    Returns (observed statistic mean(a) - mean(b), p, number of splits).
    """
    pooled = np.array(list(a) + list(b), dtype=float)
    n, k = len(pooled), len(a)
    observed = float(np.mean(a) - np.mean(b))
    total = pooled.sum()
    count = splits = 0
    for idx in itertools.combinations(range(n), k):
        s = pooled[list(idx)].sum()
        stat = s / k - (total - s) / (n - k)
        # A small tolerance so a split equal to the observed one is not lost to rounding.
        count += abs(stat) >= abs(observed) - 1e-12
        splits += 1
    return observed, count / splits, splits


def bootstrap_ci(a: list[float], b: list[float], seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.array(a), np.array(b)
    ia = rng.integers(0, len(a_arr), size=(BOOT_N, len(a_arr)))
    ib = rng.integers(0, len(b_arr), size=(BOOT_N, len(b_arr)))
    diffs = a_arr[ia].mean(axis=1) - b_arr[ib].mean(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def m1m_of(entry: dict) -> float:
    return float(np.mean([v["auroc"] for v in entry["M1_by_attribute"].values()
                          if v["n"] >= 30 and "auroc" in v]))


def verdict(diff: float, p: float) -> str:
    if p >= ALPHA:
        return "退行は検出されず。seed 8〜15 の低さは偶然として扱う"
    return "退行あり" if diff < 0 else "退行は検出されず（new が高い）"


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    data = load_data()
    available = [n for n in NEW if Path(f"runs/{n}/model.pt").exists()]
    for name in available:
        collect(name, f"runs/{name}/model.pt", tokenizer, data)
    score = Scorer(tokenizer, data)
    new = {}
    for name in available:
        d = np.load(PROBS / f"probs_{name}.npz")
        new[name] = score(d["frozen"], d["long"], d["val"], d["val_n"])
    worst = 0.0
    for name in available:
        e = json.loads(Path(f"runs/regression/eval_{name}.json").read_text(encoding="utf-8"))[0]
        worst = max(worst, abs(new[name]["M1"] - e["M1"]["auroc"]),
                    abs(new[name]["M1m"] - m1m_of(e)))
    old_all = json.loads(Path("runs/soup/results.json").read_text(encoding="utf-8"))["scores"]
    old = {n: old_all[n] for n in OLD}
    later = {n: old_all[n] for n in LATER}
    res: dict = {"new": new, "check_max_abs_diff_vs_stored_eval": worst, "n_new": len(new)}
    summary = {}
    for label, group in (("new", new), ("old", old), ("seeds_8_15", later)):
        if group:
            summary[label] = {
                "mean": {m: float(np.mean([g[m] for g in group.values()])) for m in METRICS},
                "sd": {m: float(np.std([g[m] for g in group.values()], ddof=1)) for m in METRICS},
            }
    res["summary"] = summary
    if len(new) == 8:
        a = [new[n]["M1m"] for n in NEW]
        b = [old[n]["M1m"] for n in OLD]
        diff, p, splits = exact_permutation_p_two_sided(a, b)
        res["main_test"] = {"metric": "M1m", "diff_new_minus_old": diff, "p_two_sided": p,
                            "splits": splits, "reject": bool(p < ALPHA),
                            "verdict": verdict(diff, p)}
        res["describe_new_vs_old"] = {
            m: {"diff": float(np.mean([new[n][m] for n in NEW])
                              - np.mean([old[n][m] for n in OLD])),
                "ci95": bootstrap_ci([new[n][m] for n in NEW], [old[n][m] for n in OLD])}
            for m in METRICS}
        res["describe_new_vs_seeds_8_15"] = {
            m: {"diff": float(np.mean([new[n][m] for n in NEW])
                              - np.mean([later[n][m] for n in LATER])),
                "ci95": bootstrap_ci([new[n][m] for n in NEW], [later[n][m] for n in LATER])}
            for m in METRICS}
    else:
        res["main_test"] = f"not run: {len(new)} of 8 new runs (判定不能)"
    out = Path("runs/regression/results.json")
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    brief = {k: v for k, v in res.items() if k != "new"}
    print(json.dumps(brief, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
