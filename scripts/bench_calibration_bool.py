"""docs/calibration.md §8: the bool temperature alone, on the saved bench outputs.

    uv run python -m scripts.bench_calibration_bool

No inference: `runs/calibration/bench_ja.npz` / `bench_en.npz` (§7's one re-inference) hold
each item's raw marker logits and probabilities. The bool temperature of
`calibration.json` (bool/2) is applied to the bool logits (`softmax(logits / T)`), and the
result is checked against `apply_temperature` on the saved probabilities (the way
`predict` applies it). choice and score are left as they are. Writes
`runs/calibration/bench_bool.json`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.calibration_heldout import p_true_ece
from sokudan.calibration.metrics import auroc
from sokudan.calibration.temperature import apply_temperature
from sokudan.predict import read_temperatures

SOURCE = Path("runs/release_candidate/calibration.json")


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> int:
    t = read_temperatures(SOURCE)[("bool", 2)]
    out = {"temperature_bool_2": t, "source": str(SOURCE), "benches": {}}
    for name in ("bench_ja", "bench_en"):
        d = np.load(f"runs/calibration/{name}.npz")
        y = d["gold_bool"]
        before = d["bool_probs"][:, 1].astype(np.float64)
        after_logits = softmax(d["bool_logits"].astype(np.float64) / t)[:, 1]
        after_probs = apply_temperature(d["bool_probs"], t)[:, 1]
        m = {}
        for label, p in (("before", before), ("after", after_logits)):
            m[label] = {"bool_ece_p_true_10": p_true_ece(p, y),
                        "abs_mean_p_minus_rate": float(abs(p.mean() - y.mean())),
                        "mean_p_minus_rate": float(p.mean() - y.mean()),
                        "bool_acc_05": float(((p > 0.5) == (y == 1)).mean()),
                        "bool_auroc": float(auroc(p, y))}
        m["logits_vs_probs_max_abs_diff"] = float(np.abs(after_logits - after_probs).max())
        m["argmax_unchanged"] = bool(np.array_equal(before > 0.5, after_logits > 0.5))
        m["choice_and_score_untouched"] = True  # not read, not written: only bool is scaled
        out["benches"][name] = m
    b = out["benches"]
    out["default_on"] = bool(all(
        b[n]["after"]["bool_ece_p_true_10"] < b[n]["before"]["bool_ece_p_true_10"]
        and b[n]["after"]["abs_mean_p_minus_rate"] < b[n]["before"]["abs_mean_p_minus_rate"]
        for n in ("bench_ja", "bench_en")))
    Path("runs/calibration/bench_bool.json").write_text(json.dumps(out, indent=1),
                                                       encoding="utf-8")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
