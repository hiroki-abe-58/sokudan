"""docs/calibration.md §6: bench_ja / bench_en re-inferred once for v0.2, raw outputs kept,
and `runs/release_candidate/calibration.json` applied afterwards.

    uv run python -m scripts.bench_calibration

The inference is the recorded path itself -- `SokudanBaseline.run` (bench_en with its three
questions swapped in, as `scripts/run_bench_en_sokudan.py` does) -- with `run_model` wrapped
to keep every call's marker logits and probabilities. Its scores (`score_baseline`) are
checked against the recorded `results.json` (choice acc, score RPS, score acc, bool acc,
bool AUROC; tolerance 1e-4); on a mismatch nothing further is computed. Then each bucket's
temperature from `calibration.json` is applied to the probabilities (`apply_temperature`,
as `predict` does) and the metrics are taken before and after. Writes
`runs/calibration/bench_<name>.npz` (logits and probabilities) and
`runs/calibration/bench.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import sokudan.config  # noqa: F401
from scripts.calibration_heldout import p_true_ece
from sokudan.calibration.metrics import ece, rps
from sokudan.calibration.temperature import apply_temperature
from sokudan.eval import bench_en, bench_ja, sokudan_baseline
from sokudan.eval.report import score_baseline
from sokudan.predict import read_temperatures

CHECKPOINT = "runs/release_candidate/model.pt"
OUT = Path("runs/calibration")
TOL = 1e-4
BENCHES = {
    "bench_ja": ("data/bench_ja.jsonl", "runs/release_candidate/bench_ja/results.json",
                 "sokudan-ja-310m", bench_ja),
    "bench_en": ("data/bench_en.jsonl", "runs/release_candidate/bench_en/results.json",
                 "sokudan (bench_en)", bench_en),
}


def run_one(name: str):
    from scripts.run_baseline_en import load_bench

    path, _, label, module = BENCHES[name]
    items = load_bench(Path(path))
    captured: list[tuple[np.ndarray, np.ndarray]] = []
    original = sokudan_baseline.run_model

    def recording(model, batch):
        out = original(model, batch)
        captured.append((out.marker_logits.float().cpu().numpy(),
                         out.probs.float().cpu().numpy()))
        return out

    sokudan_baseline.run_model = recording
    sokudan_baseline.bench_questions = module.bench_questions
    try:
        output = sokudan_baseline.SokudanBaseline(CHECKPOINT, label).run(items)
    finally:
        sokudan_baseline.run_model = original
        sokudan_baseline.bench_questions = bench_ja.bench_questions
    keys = list(module.DEPARTMENTS)
    gold = {"choice": np.array([keys.index(i.department) for i in items]),
            "score": np.array([i.urgency for i in items]),
            "bool": np.array([int(i.churn) for i in items])}
    # calls cycle department (choice), urgency (score), churn (bool) per chunk
    parts = {k: [] for k in ("choice", "score", "bool")}
    for i, (logits, probs) in enumerate(captured):
        parts[("choice", "score", "bool")[i % 3]].append((logits, probs))
    raw = {}
    for k, v in parts.items():
        width = {"choice": len(keys), "score": len(module.URGENCY_LEVELS), "bool": 2}[k]
        raw[f"{k}_logits"] = np.vstack([lg[:, :width] for lg, _ in v])
        raw[f"{k}_probs"] = np.vstack([p[:, :width] for _, p in v])
    return output, gold, raw


def metrics(choice: np.ndarray, score: np.ndarray, p_true: np.ndarray, gold) -> dict:
    return {"bool_ece_p_true_10": p_true_ece(p_true, gold["bool"]),
            "bool_mean_p_minus_rate": float(p_true.mean() - gold["bool"].mean()),
            "bool_acc_05": float(((p_true > 0.5) == (gold["bool"] == 1)).mean()),
            "choice_ece_top_label_10": float(ece(choice, gold["choice"], n_bins=10)),
            "score_rps": float(rps(score, gold["score"]))}


def main() -> int:
    temps = read_temperatures("runs/release_candidate/calibration.json")
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": CHECKPOINT, "temperatures_used": {}, "benches": {}}
    for name, (_, recorded_path, label, _module) in BENCHES.items():
        output, gold, raw = run_one(name)
        np.savez_compressed(OUT / f"{name}.npz", **raw, **{f"gold_{k}": v
                                                         for k, v in gold.items()})
        mine = score_baseline(output, gold)
        recorded = next(r for r in json.loads(Path(recorded_path).read_text(
            encoding="utf-8"))["results"] if r["name"] == label)
        checks = {"choice_accuracy": (mine["choice"]["accuracy"],
                                      recorded["choice"]["accuracy"]),
                  "score_rps": (mine["score"]["rps"], recorded["score"]["rps"]),
                  "score_accuracy": (mine["score"]["accuracy"], recorded["score"]["accuracy"]),
                  "bool_accuracy": (mine["bool"]["accuracy"], recorded["bool"]["accuracy"]),
                  "bool_auroc": (mine["bool"]["auroc"], recorded["bool"]["auroc"])}
        same = all(abs(a - b) <= TOL for a, b in checks.values())
        entry = {"reproduces_record": same,
                 "checks": {k: {"now": a, "recorded": b, "diff": a - b}
                            for k, (a, b) in checks.items()}}
        # raw-logit consistency: the saved probabilities are the head's outputs
        entry["argmax_from_logits_matches_probs"] = bool(np.array_equal(
            raw["choice_logits"].argmax(1), raw["choice_probs"].argmax(1)))
        if not same:
            report["benches"][name] = entry
            print(json.dumps(report, ensure_ascii=False, indent=1))
            (OUT / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
            print(f"STOP: {name} does not reproduce the record", file=sys.stderr)
            return 2
        c, s, b = raw["choice_probs"], raw["score_probs"], raw["bool_probs"]
        k_c, k_s = c.shape[1], s.shape[1]
        t_c, t_s, t_b = (temps.get(("choice", k_c), 1.0), temps.get(("score", k_s), 1.0),
                         temps.get(("bool", 2), 1.0))
        report["temperatures_used"] = {f"choice/{k_c}": t_c, f"score/{k_s}": t_s,
                                       "bool/2": t_b}
        entry["before"] = metrics(c, s, b[:, 1], gold)
        cc, ss, bb = (apply_temperature(c, t_c), apply_temperature(s, t_s),
                      apply_temperature(b, t_b))
        entry["after"] = metrics(cc, ss, bb[:, 1], gold)
        entry["argmax_unchanged"] = bool(np.array_equal(c.argmax(1), cc.argmax(1))
                                         and np.array_equal(s.argmax(1), ss.argmax(1))
                                         and np.array_equal(b.argmax(1), bb.argmax(1)))
        entry["recorded_ece_definition"] = {"bool_ece_before": recorded["bool"]["ece"]}
        report["benches"][name] = entry
    ja = report["benches"]["bench_ja"]
    report["default_on"] = bool(
        ja["after"]["bool_ece_p_true_10"] < ja["before"]["bool_ece_p_true_10"]
        and ja["after"]["score_rps"] - ja["before"]["score_rps"] < 0.005)
    (OUT / "bench.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
