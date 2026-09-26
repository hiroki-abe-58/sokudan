"""docs/regression_check.md §9: old16 against new16, and the pre-registered decision.

    uv run python scripts/regression16_eval.py

- old16: old-code v0.1 seeds 0-7 (`runs/soup/results.json` scores) + old-code seeds 8-15
  (the old-code worktree's `runs/v01_seed<s>`, see scripts/oldcode_train.py; collected as
  `v01old_seed<s>`; their `eval_local_attention --save-probs` output is reused when present).
- new16: new-code seeds 0-7 (`runs/regression/results.json`) + seeds 8-15
  (`runs/soup/results.json`).

Every score comes from `soup_eval.Scorer` on per-row predictions, the scoring used for
all 32 models. Main test: M1m, two-sided Monte Carlo permutation, 100,000 relabellings
(`numpy.random.default_rng(20260927)`), p = (1 + #{|stat| >= |observed| - 1e-12}) /
(1 + 100,000), alpha 0.05. Description: every other metric's mean difference with a
bootstrap 95% interval (10,000 draws, seed 0), and old 8-15 against old 0-7.
Writes `runs/regression16/results.json`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401
from scripts.regression_eval import bootstrap_ci, m1m_of
from scripts.soup_eval import METRICS, PROBS, Scorer, collect, load_data

WORKTREE = Path(os.environ.get("SOKUDAN_OLDCODE_WORKTREE",
                               Path(__file__).resolve().parents[2] / "sokudan-oldcode"))
ALPHA = 0.05
N_PERM = 100_000
PERM_SEED = 20260927


def monte_carlo_p(a: list[float], b: list[float], n: int = N_PERM,
                  seed: int = PERM_SEED) -> tuple[float, float, int]:
    """Two-sided p of mean(a) - mean(b) over `n` random relabellings (observed counted once)."""
    pooled = np.array(list(a) + list(b), dtype=float)
    k = len(a)
    observed = float(np.mean(a) - np.mean(b))
    rng = np.random.default_rng(seed)
    total = pooled.sum()
    hits = 0
    for _ in range(n):
        idx = rng.permutation(len(pooled))[:k]
        s = pooled[idx].sum()
        stat = s / k - (total - s) / (len(pooled) - k)
        hits += abs(stat) >= abs(observed) - 1e-12
    return observed, (1 + hits) / (1 + n), n


def verdict(diff: float, p: float) -> tuple[str, str, str]:
    """(record, code, baseline distribution) per §9.3."""
    if p >= ALPHA:
        return "退行は検出されず", "修正後のコードを維持", "new16"
    if diff < 0:
        return "修正は退行を伴う", "set_seed の位置を修正前に戻す（revert）", "old16"
    return "退行は検出されず（new16 が高い）", "修正後のコードを維持", "new16"


def main() -> int:
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    data = load_data()
    old_late = {}
    worst = 0.0
    score = None
    for s in range(8, 16):
        ckpt = WORKTREE / "runs" / f"v01_seed{s}" / "model.pt"
        if not ckpt.exists():
            continue
        name = f"v01old_seed{s}"
        eval_json = Path(f"runs/regression16/eval_{name}.json")
        collect(name, ckpt, tokenizer, data,
                precomputed=Path(f"{eval_json}.probs0.npz"))
        score = score or Scorer(tokenizer, data)
        d = np.load(PROBS / f"probs_{name}.npz")
        old_late[name] = score(d["frozen"], d["long"], d["val"], d["val_n"])
        if eval_json.exists():
            e = json.loads(eval_json.read_text(encoding="utf-8"))[0]
            worst = max(worst, abs(old_late[name]["M1"] - e["M1"]["auroc"]),
                        abs(old_late[name]["M1m"] - m1m_of(e)))
    soup_scores = json.loads(Path("runs/soup/results.json").read_text(encoding="utf-8"))["scores"]
    new_early = json.loads(Path("runs/regression/results.json").read_text(encoding="utf-8"))["new"]
    old16 = [soup_scores[f"v01_seed{s}"] for s in range(8)] + list(old_late.values())
    new16 = [new_early[f"v01new_seed{s}"] for s in range(8)] + \
        [soup_scores[f"v01_seed{s}"] for s in range(8, 16)]
    res: dict = {"old_8_15": old_late, "n_old16": len(old16), "n_new16": len(new16),
                 "check_max_abs_diff_vs_stored_eval": worst}

    def summary(group: list[dict]) -> dict:
        return {"mean": {m: float(np.mean([g[m] for g in group])) for m in METRICS},
                "sd": {m: float(np.std([g[m] for g in group], ddof=1)) for m in METRICS}}

    res["summary"] = {"old16": summary(old16), "new16": summary(new16),
                      "old_0_7": summary(old16[:8])}
    if old_late:
        res["summary"]["old_8_15"] = summary(list(old_late.values()))
    if len(old16) == 16:
        diff, p, n = monte_carlo_p([g["M1m"] for g in new16], [g["M1m"] for g in old16])
        record, code, baseline = verdict(diff, p)
        res["main_test"] = {"metric": "M1m", "diff_new16_minus_old16": diff,
                            "p_two_sided_monte_carlo": p, "permutations": n,
                            "seed": PERM_SEED, "reject": bool(p < ALPHA),
                            "verdict": record, "code": code, "baseline": baseline}
        res["describe_new16_vs_old16"] = {
            m: {"diff": float(np.mean([g[m] for g in new16]) - np.mean([g[m] for g in old16])),
                "ci95": bootstrap_ci([g[m] for g in new16], [g[m] for g in old16])}
            for m in METRICS}
        res["describe_old_8_15_vs_old_0_7"] = {
            m: {"diff": float(np.mean([g[m] for g in old16[8:]])
                              - np.mean([g[m] for g in old16[:8]])),
                "ci95": bootstrap_ci([g[m] for g in old16[8:]], [g[m] for g in old16[:8]])}
            for m in METRICS}
    else:
        res["main_test"] = (f"not run: {len(old16) - 8} of 8 old-code seeds 8-15 "
                            "(判定不能: コードは修正後のまま、基準分布は更新しない)")
    out = Path("runs/regression16/results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "old_8_15"}, ensure_ascii=False,
                     indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
