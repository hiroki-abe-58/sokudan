"""Check our Laya numbers against the official package used the plain way.

    uv run python scripts/verify_laya_harness.py

`docs/baseline_ja.md` reports a position-bias finding strong enough to be worth
sending to Laya's authors: across five schema variants on the same 300 items,
`laya-multilingual` put 0, 0, 1, 1 and 0 predictions on the first presented option,
and 「急がない」 drew 0 predictions in first position against 250 in last.

Before reporting that anywhere, the harness itself has to be ruled out. This script
calls `laya.load(...)` and `agent.predict(...)` directly -- the two lines from the
package's own README, with no wrapper of ours in between -- on `bench_ja` items, and
compares what comes back with the probabilities stored in
`runs/baseline_ja/choice_probs.npz`.

Two comparisons, because the stored numbers are not raw:

* **raw**: what the package returned, renormalised only.
* **floored**: after `floor_and_renormalise` at `PROB_FLOOR = 5e-5`, which is what
  `run_baseline_ja.py` stores.

A mismatch in `floored` means our harness changed the answer. A mismatch in `raw`
alone means the floor moved it, which is expected and bounded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

import sokudan.config  # noqa: F401
from sokudan.eval.baselines import PROB_FLOOR, floor_and_renormalise
from sokudan.eval.bench_ja import DEPARTMENTS, URGENCY_LEVELS, BenchItem, bench_questions

STORED = Path("runs/baseline_ja/choice_probs.npz")
MODEL_ID = "convaiinnovations/laya-multilingual"
STORED_NAME = "laya-multilingual (ja)"


def load_bench(path: Path, limit: int) -> list[BenchItem]:
    items = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                items.append(BenchItem(**json.loads(line)))
            if len(items) >= limit:
                break
    return items


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", default="data/bench_ja.jsonl")
    parser.add_argument("--n", type=int, default=1,
                        help="how many items to re-run through the plain API")
    parser.add_argument("--out", default="runs/diagnostics/laya_harness_check.json")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    items = load_bench(Path(args.bench), args.n)
    questions = bench_questions()

    # --- the two lines from the package README, nothing of ours in between ---
    import laya

    agent = laya.load(MODEL_ID)
    raw_results = [agent.predict({"body": item.state}, questions) for item in items]
    # -----------------------------------------------------------------------

    stored = np.load(STORED, allow_pickle=True)
    names = [str(n) for n in stored["names"]]
    index = names.index(STORED_NAME)
    stored_choice = stored[f"p{index}"]

    dept_keys = list(DEPARTMENTS)
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "laya_version": getattr(laya, "__version__", "unknown"),
        "stored_run": str(STORED),
        "stored_series": STORED_NAME,
        "prob_floor": PROB_FLOOR,
        "n_items": len(items),
        "items": [],
    }
    worst_raw = 0.0
    worst_floored = 0.0

    for position, (item, result) in enumerate(zip(items, raw_results, strict=True)):
        answers = result.get("answers", {})
        dept = answers.get("department", {}).get("probabilities", {})
        row = np.array([float(dept.get(k, 0.0)) for k in dept_keys])
        raw = row / row.sum() if row.sum() > 0 else np.full(len(dept_keys), 1 / len(dept_keys))
        floored = floor_and_renormalise(raw.reshape(1, -1))[0]

        expected = stored_choice[position]
        delta_raw = float(np.abs(raw - expected).max())
        delta_floored = float(np.abs(floored - expected).max())
        worst_raw = max(worst_raw, delta_raw)
        worst_floored = max(worst_floored, delta_floored)

        urgency = answers.get("urgency", {}).get("probabilities", {})
        noul = answers.get("churn", {}).get("noul")

        report["items"].append({
            "item_id": item.item_id,
            "gold_department": item.department,
            "departments": dept_keys,
            "raw": [round(float(v), 6) for v in raw],
            "floored": [round(float(v), 6) for v in floored],
            "stored": [round(float(v), 6) for v in expected],
            "max_abs_delta_raw": delta_raw,
            "max_abs_delta_floored": delta_floored,
            "urgency_probabilities": {
                str(k): float(urgency.get(str(k), 0.0)) for k in range(len(URGENCY_LEVELS))
            },
            "churn_noul": None if noul is None else float(noul),
            "answer_keys": sorted(answers),
        })

        print(f"\n{item.item_id}  gold={item.department}")
        print(f"  departments {dept_keys}")
        print(f"  raw      {np.round(raw, 6)}")
        print(f"  floored  {np.round(floored, 6)}")
        print(f"  stored   {np.round(expected, 6)}")
        print(f"  |Δ| raw {delta_raw:.2e}   floored {delta_floored:.2e}")
        print(f"  urgency {report['items'][-1]['urgency_probabilities']}  "
              f"churn.noul {report['items'][-1]['churn_noul']}")

    report["max_abs_delta_raw"] = worst_raw
    report["max_abs_delta_floored"] = worst_floored
    report["matches"] = worst_floored <= args.tolerance
    report["tolerance"] = args.tolerance

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nmax |Δ| vs stored:  raw {worst_raw:.3e}   floored {worst_floored:.3e}")
    print(f"-> {out}")

    if report["matches"]:
        print("\nMATCH: 素の laya API と昨日のハーネスの確率は一致。"
              "位置バイアスはハーネス由来ではない。")
        return 0
    print("\nMISMATCH: 差分あり。上の raw / floored / stored を比較すること。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
