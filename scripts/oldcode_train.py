"""docs/regression_check.md §8: v0.1 seeds 8-15 with the OLD code (condition old16).

    uv run python scripts/oldcode_train.py --deadline-epoch <unix time>

Trains in a worktree of the old code (`git worktree add ../sokudan-oldcode f75e3f2^`, never
edited; another location via `SOKUDAN_OLDCODE_WORKTREE`), with this repository's `.venv`
interpreter and `PYTHONPATH` pointing at the worktree, so `scripts/train.py` and the
`sokudan` package are the pre-fix ones. v0.1's command (question_first, batch 24, no
accumulation, the usual VRAM cap); the data files are read from this repository by absolute
path. Checkpoints go to `<worktree>/runs/v01_seed<s>` (local only); evaluation runs with
THIS repository's `scripts/eval_local_attention.py` into
`runs/regression16/eval_v01old_seed<s>.json`. Stops, logged to
`runs/regression16/train.log`: a non-zero exit (CUDA out of memory included; no fallback to
accumulation), a NaN / inf loss, the first 50 steps over 2x 6.3 s (over 1.5x only recorded),
seed 8's M1m below 0.791169, and a run that would end past the deadline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REFERENCE_50 = 6.3
M1M_FLOOR = 0.791169
LOG = Path("runs/regression16/train.log")
REPO = Path(__file__).resolve().parents[1]
WORKTREE = Path(os.environ.get("SOKUDAN_OLDCODE_WORKTREE", REPO.parent / "sokudan-oldcode"))
PYTHON = str(REPO / ".venv" / "Scripts" / "python.exe")
SEEDS = range(8, 16)
STEP = re.compile(r"epoch (\d+) step (\d+)/\d+ loss (\S+) elapsed ([\d.]+)s")


def log(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {line}\n")
    print(f"{stamp}  {line}", flush=True)


def m1m(entry: dict) -> float:
    return float(np.mean([v["auroc"] for v in entry["M1_by_attribute"].values()
                          if v["n"] >= 30 and "auroc" in v]))


def train(seed: int) -> bool:
    run_id = f"v01_seed{seed}"
    log_path = Path(f"runs/regression16/train_v01old_seed{seed}.log")
    env = {**os.environ, "PYTHONPATH": str(WORKTREE),
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    command = [PYTHON, "scripts/train.py", "--train", str(REPO / "data" / "train_v2b.jsonl"),
               "--val", str(REPO / "data" / "val_v2.jsonl"), "--encoding", "joint",
               "--epochs", "2",
               "--batch-size", "24", "--lr", "2e-5", "--head-lr", "2e-4",
               "--seed", str(seed), "--max-vram-fraction", "0.88",
               "--input-order", "question_first", "--save", "--run-id", run_id]
    log(f"train start {run_id}: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(command, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                cwd=WORKTREE)
    checked = False
    while proc.poll() is None:
        time.sleep(5)
        steps = STEP.findall(log_path.read_text(encoding="utf-8", errors="replace"))
        for _, _, loss, _ in steps:
            value = float(loss)
            if math.isnan(value) or math.isinf(value):
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                log(f"STOP {run_id}: loss {loss}")
                return False
        first = [s for s in steps if s[0] == "0" and s[1] == "50"]
        if first and not checked:
            checked = True
            seconds = float(first[0][3])
            ratio = seconds / REFERENCE_50
            log(f"{run_id} first 50 steps {seconds:.1f}s = {ratio:.2f}x v0.1's "
                f"{REFERENCE_50}s{' (over 1.5x, recorded)' if ratio > 1.5 else ''}")
            if ratio > 2:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                log(f"STOP {run_id}: over 2x (paging suspected)")
                return False
    if proc.returncode != 0:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        oom = "OutOfMemoryError" in text or "CUDA out of memory" in text
        log(f"STOP {run_id}: exit code {proc.returncode}{' (CUDA out of memory)' if oom else ''}")
        return False
    report = (WORKTREE / "runs" / run_id / "report.md").read_text(encoding="utf-8")
    peak = re.findall(r"peak_alloc ([\d.]+)GiB peak_reserved ([\d.]+)GiB",
                      log_path.read_text(encoding="utf-8", errors="replace"))
    seconds = re.search(r"\| 学習 \| (\d+) 秒", report)
    stamp = re.search(r"実行日時: (\S+)", report)
    log(f"train done {run_id}: {seconds.group(1) if seconds else '?'} s training, "
        f"peak alloc/reserved {peak[-1] if peak else '?'} GiB, "
        f"report {stamp.group(1) if stamp else '?'}")
    return True


def evaluate(seed: int) -> dict:
    run_id = f"v01old_seed{seed}"
    out = Path(f"runs/regression16/eval_{run_id}.json")
    log(f"eval start {run_id}")
    with open(f"runs/regression16/eval_{run_id}.log", "w", encoding="utf-8") as fh:
        code = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.eval_local_attention", "--checkpoints",
             str(WORKTREE / "runs" / f"v01_seed{seed}" / "model.pt"), "--out", str(out)],
            stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    if code != 0:
        raise SystemExit(f"evaluation of {run_id} failed: {code}")
    e = json.loads(out.read_text(encoding="utf-8"))[0]
    assert e["input_order"] == "question_first" and e["local_attention"] == 128
    assert not e["inference_only_override"]
    log(f"eval done {run_id}: M1m {m1m(e):.6f} M1 {e['M1']['auroc']:.6f} "
        f"M2 {e['M2']['auroc']:.6f} M5 {e['M1_by_attribute']['implies_declining']['auroc']:.6f}")
    return e


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    longest = 0.0
    for seed in SEEDS:
        if longest and time.time() + longest > args.deadline_epoch:
            log(f"STOP deadline: seed {seed} would end past the wind-down "
                f"(longest run {longest:.0f}s)")
            return 0
        began = time.time()
        if not train(seed):
            return 1
        entry = evaluate(seed)
        longest = max(longest, time.time() - began)
        if seed == SEEDS[0]:
            value = m1m(entry)
            if value < M1M_FLOOR:
                log(f"BROKEN v01old seeds 8-15: seed {seed} M1m {value:.6f} < {M1M_FLOOR}")
                return 0
            log(f"not broken: seed {seed} M1m {value:.6f} >= {M1M_FLOOR}")
    log("v01old seeds 8-15 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
