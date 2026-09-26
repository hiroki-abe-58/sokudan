"""docs/release_candidate.md Phase 5: the one bench run of the chosen soup, and the release rule.

    uv run python scripts/release_bench.py run     # runs the bench once (refuses a second time)
    uv run python scripts/release_bench.py judge   # applies the rule to what `run` wrote

`run`, in this order, each step skipped if its output already exists and never repeated
once it has produced one:

1. the scoring code changed after v0.1's bench record (68d217b), so v0.1 seed 0 is
   re-scored on `bench_ja` with the current code (`runs/release_candidate/bench_v01_seed0/`);
2. `scripts/calibrate.py` on `val_v2` for the chosen soup (v0.1's procedure; the calibrated
   row is description only);
3. `bench_ja` with `scripts/run_baseline_ja.py --skip-laya` (the v0.1 command);
4. `bench_en` with `scripts/run_bench_en_sokudan.py` (description only);
5. the position check of `docs/benchmarks.md` §2 (`scripts/probe_position_bias.py`, the five
   `bench_ja` schema variants; description only).

`judge` compares the soup's uncalibrated `bench_ja` row with the mean of v0.1's three seeds
(`runs/bench_v01_seeds.json`, uncalibrated, as `docs/benchmarks.md` §1 records them):

- main: bool AUROC >= v0.1 + 0.01 and score RPS <= v0.1 - 0.005;
- guardrails: choice acc, bool acc and score acc each >= v0.1 - 0.01.

If the re-scored seed 0 differs from its record by more than 1e-4 on any of the five
metrics, the rule is also applied against the three-seed mean with seed 0 replaced by the
re-score, and must pass against both. Writes `runs/release_candidate/judgement.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

OUT = Path("runs/release_candidate")
MODEL = OUT / "model.pt"
FIVE = (("choice", "accuracy"), ("score", "rps"), ("score", "accuracy"),
        ("bool", "accuracy"), ("bool", "auroc"))
MATCH = 1e-4
EPS = 1e-12


def step(name: str, done: Path, command: list[str]) -> None:
    if done.exists():
        print(f"skip {name}: {done} exists (never repeated)", flush=True)
        return
    print(f"run {name}: {' '.join(command)}", flush=True)
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    with open(OUT / f"{name}.log", "w", encoding="utf-8") as fh:
        code = subprocess.run(command, stdout=fh, stderr=subprocess.STDOUT, env=env).returncode
    if code != 0 or not done.exists():
        raise SystemExit(f"{name} failed (exit {code}); see {OUT / f'{name}.log'}")


def run() -> None:
    selection = json.loads((OUT / "selection.json").read_text(encoding="utf-8"))
    if selection.get("chosen") is None or not MODEL.exists():
        raise SystemExit("no release candidate: Phase 5 is not run")
    py = ["uv", "run", "python"]
    step("bench_v01_seed0", OUT / "bench_v01_seed0" / "results.json",
         py + ["scripts/run_baseline_ja.py", "--skip-laya", "--out", str(OUT / "bench_v01_seed0"),
               "--sokudan-checkpoint", "runs/v01_seed0/model.pt",
               "--sokudan-temperatures", "runs/v01_seed0/temperatures.json"])
    step("calibrate", OUT / "temperatures.json",
         py + ["scripts/calibrate.py", "--checkpoint", str(MODEL), "--val", "data/val_v2.jsonl"])
    step("bench_ja", OUT / "bench_ja" / "results.json",
         py + ["scripts/run_baseline_ja.py", "--skip-laya", "--out", str(OUT / "bench_ja"),
               "--sokudan-checkpoint", str(MODEL),
               "--sokudan-temperatures", str(OUT / "temperatures.json")])
    step("bench_en", OUT / "bench_en" / "results.json",
         py + ["scripts/run_bench_en_sokudan.py", "--checkpoint", str(MODEL),
               "--out", str(OUT / "bench_en")])
    step("position_bias_ja", OUT / "position_bias_ja.json",
         py + ["scripts/probe_position_bias.py", "--model", f"sokudan:{MODEL}",
               "--out", str(OUT / "position_bias_ja.json")])


def row(path: Path, name: str) -> dict:
    results = json.loads(path.read_text(encoding="utf-8"))["results"]
    return next(r for r in results if r["name"] == name)


def five(r: dict) -> dict[str, float]:
    return {f"{k}/{m}": float(r[k][m]) for k, m in FIVE}


def rule(soup: dict[str, float], ref: dict[str, float]) -> dict:
    checks = {
        "bool/auroc >= ref + 0.01": soup["bool/auroc"] - ref["bool/auroc"] >= 0.01 - EPS,
        "score/rps <= ref - 0.005": soup["score/rps"] - ref["score/rps"] <= -0.005 + EPS,
        **{f"{k} >= ref - 0.01": soup[k] - ref[k] >= -0.01 - EPS
           for k in ("choice/accuracy", "bool/accuracy", "score/accuracy")},
    }
    return {"ref": ref, "diff": {k: soup[k] - ref[k] for k in soup}, "checks": checks,
            "pass": all(checks.values())}


def judge() -> dict:
    seeds = json.loads(Path("runs/bench_v01_seeds.json").read_text(encoding="utf-8"))
    recorded = {k: [float(v) for v in seeds["uncalibrated"][k]["values"]] for k in
                (f"{a}/{b}" for a, b in FIVE)}
    ref = {k: float(np.mean(v)) for k, v in recorded.items()}
    rescored = five(row(OUT / "bench_v01_seed0" / "results.json", "sokudan-ja-310m"))
    seed0_diff = {k: rescored[k] - recorded[k][0] for k in rescored}
    matched = all(abs(d) <= MATCH for d in seed0_diff.values())
    soup = five(row(OUT / "bench_ja" / "results.json", "sokudan-ja-310m"))
    res = {"soup": soup, "v01_seed0_rescore": rescored, "v01_seed0_diff": seed0_diff,
           "v01_seed0_matched": matched, "against_recorded": rule(soup, ref)}
    passed = res["against_recorded"]["pass"]
    if not matched:
        replaced = {k: float(np.mean([rescored[k]] + v[1:])) for k, v in recorded.items()}
        res["against_seed0_replaced"] = rule(soup, replaced)
        passed = passed and res["against_seed0_replaced"]["pass"]
    res["release"] = ("v0.2 候補として確定。HF への公開はチャット側の確認後" if passed
                      else "v0.1 を維持")
    (OUT / "judgement.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run", "judge"))
    args = parser.parse_args()
    if args.command == "run":
        run()
    else:
        print(json.dumps(judge(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
