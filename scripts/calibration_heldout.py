"""docs/calibration.md: temperature per (question type, option count) for v0.2, fitted and
judged by two-fold cross-fitting, then fitted on everything for shipping.

    uv run python -m scripts.calibration_heldout

Rows: v0.2's (S8_old) per-row probabilities as saved in `runs/ens/probs_k8_0_7.npz`
(frozen held-out P(true), `val_v2` distributions; no inference here). bool comes from the
frozen held-out set; choice and score from `val_v2` (the held-out set has no choice or
score rows). Each set is split in two by the SHA-256 of `doc_id` (first byte even: A,
odd: B); a temperature fitted on A (NLL, `sokudan.calibration.temperature`) is applied to
B and vice versa, and every row is scored with the temperature it was not fitted on.
Buckets with fewer than 25 rows in the fitting half keep T = 1. Writes
`runs/calibration/results.json` and, if the shipping condition holds,
`runs/release_candidate/calibration.json` (the `load(temperatures=...)` format).
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from scripts.soup_eval import load_data
from sokudan.calibration.metrics import auroc, ece, rps
from sokudan.calibration.temperature import apply_temperature, fit_temperature

PROBS = Path("runs/ens/probs_k8_0_7.npz")
OUT = Path("runs/calibration/results.json")
SHIP = Path("runs/release_candidate/calibration.json")
MIN_BUCKET = 25


def half(doc_id: str) -> int:
    return hashlib.sha256(doc_id.encode("utf-8")).digest()[0] % 2


def p_true_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Binary ECE of P(true): equal-width bins on [0, 1], |mean p - mean y| weighted."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() / len(p) * abs(p[m].mean() - y[m].mean())
    return float(total)


def fit(rows: list[tuple[np.ndarray, int]]) -> float:
    if len(rows) < MIN_BUCKET:
        return 1.0
    return float(fit_temperature(np.stack([p for p, _ in rows]),
                                 np.array([y for _, y in rows])))


def cross_fit(items: list[tuple[str, tuple, np.ndarray, int]]):
    """items: (doc_id, bucket, probs, label). Returns per-row calibrated probs (cross-fit),
    the per-half temperatures, and the temperatures fitted on everything."""
    by = {h: defaultdict(list) for h in (0, 1)}
    for doc, key, p, y in items:
        by[half(doc)][key].append((p, y))
    temps = {h: {k: fit(v) for k, v in by[h].items()} for h in (0, 1)}
    out = []
    for doc, key, p, _y in items:
        t = temps[1 - half(doc)].get(key, 1.0)
        out.append(apply_temperature(p[None, :], t)[0] if t != 1.0 else p)
    everything = defaultdict(list)
    for _, key, p, y in items:
        everything[key].append((p, y))
    return out, temps, {k: fit(v) for k, v in everything.items()}


def main() -> int:
    data = load_data()
    saved = np.load(PROBS)
    frozen = data["frozen"]
    # ---- bool: frozen held-out
    p = saved["frozen"].astype(np.float64)
    y = np.array([int(r["label"]) for r in frozen])
    items = [(r["doc_id"], ("bool", 2), np.array([1 - pi, pi]), yi)
             for r, pi, yi in zip(frozen, p, y, strict=True)]
    cal, bool_temps, bool_all = cross_fit(items)
    q = np.array([c[1] for c in cal])
    bool_res = {
        "n": len(p), "n_half": {h: int(sum(half(r["doc_id"]) == h for r in frozen))
                                for h in (0, 1)},
        "temperature_by_half": {str(h): v[("bool", 2)] for h, v in bool_temps.items()},
        "before": {"ece_p_true_10": p_true_ece(p, y),
                   "mean_p_minus_rate": float(p.mean() - y.mean()),
                   "acc_05": float(((p > 0.5) == (y == 1)).mean()),
                   "auroc": float(auroc(p, y))},
        "after_cross_fit": {"ece_p_true_10": p_true_ece(q, y),
                            "mean_p_minus_rate": float(q.mean() - y.mean()),
                            "acc_05": float(((q > 0.5) == (y == 1)).mean()),
                            "auroc": float(auroc(q, y))},
    }
    bool_res["argmax_unchanged"] = bool(np.array_equal(p > 0.5, q > 0.5))
    # ---- choice and score: val_v2
    val = data["val"]
    v, n = saved["val"], saved["val_n"]
    typed = {}
    for kind in ("choice", "score"):
        idx = [i for i, e in enumerate(val) if e.kind == kind]
        items = [(val[i].doc_id, (kind, int(n[i])), v[i, :n[i]].astype(np.float64),
                  int(val[i].label)) for i in idx]
        cal, temps, full = cross_fit(items)
        before = [it[2] for it in items]
        labels = [it[3] for it in items]
        if kind == "choice":
            def metric(ps, labels=labels):
                # top-label ECE, 10 bins, pooled over K (padded to the widest)
                w = max(len(x) for x in ps)
                m = np.zeros((len(ps), w))
                for i, x in enumerate(ps):
                    m[i, :len(x)] = x
                return float(ece(m, np.array(labels), n_bins=10))
            name = "ece_top_label_10"
        else:
            def metric(ps, labels=labels):
                return float(np.mean([rps(x[None, :], np.array([yy]))
                                      for x, yy in zip(ps, labels, strict=True)]))
            name = "rps"
        typed[kind] = {
            "n": len(idx), name: {"before": metric(before), "after_cross_fit": metric(cal)},
            "argmax_unchanged": bool(all(int(np.argmax(a)) == int(np.argmax(b))
                                         for a, b in zip(before, cal, strict=True))),
            "temperature_by_half": {str(h): {f"{k[0]}/{k[1]}": t for k, t in d.items()}
                                    for h, d in temps.items()},
            "temperature_all": {f"{k[0]}/{k[1]}": t for k, t in full.items()},
        }
    ship = (bool_res["after_cross_fit"]["ece_p_true_10"] < bool_res["before"]["ece_p_true_10"]
            and bool_res["after_cross_fit"]["acc_05"] - bool_res["before"]["acc_05"] >= -0.005)
    shipped = {}
    if ship:
        shipped["bool/2"] = bool_all[("bool", 2)]
        if typed["choice"]["ece_top_label_10"]["after_cross_fit"] <= \
                typed["choice"]["ece_top_label_10"]["before"]:
            shipped.update(typed["choice"]["temperature_all"])
        if typed["score"]["rps"]["after_cross_fit"] <= typed["score"]["rps"]["before"]:
            shipped.update(typed["score"]["temperature_all"])
    res = {"source": str(PROBS), "bool_heldout": bool_res, **typed,
           "ship": ship, "shipped_temperatures": shipped}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float),
                   encoding="utf-8")
    if ship:
        SHIP.write_text(json.dumps({
            "checkpoint": "runs/release_candidate/model.pt",
            "fitted_on": "bool: frozen held-out (all rows); choice / score: val_v2 (all rows)",
            "procedure": "docs/calibration.md", "min_bucket": MIN_BUCKET,
            "temperatures": dict(sorted(shipped.items()))}, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
