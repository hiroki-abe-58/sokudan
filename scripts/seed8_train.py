"""Training and evaluation for docs/seed8.md, in the pre-registered order.

    uv run python scripts/seed8_train.py --deadline-epoch <unix time>

Every run: question_first, no `--local-attention`, batch 12 with gradient accumulation
2, v0.1's other settings, the same VRAM cap as every earlier run. base12 trains on
`data/train_v2b.jsonl`, long12 on `data/l2x2/train_long.jsonl`. long12 seeds 0-2 are
the 2x2 runs (`runs/l2x2_qf_long_seed{0,1,2}`), pooled after the check in docs/seed8.md.

Stops, each logged to `runs/seed8/train.log` with the time read at that moment:

- **speed**: the first 50 micro-batches take more than twice the reference (long12:
  4.4 s, qf-long seed 0 with the same command; base12: 3.15 s, v0.1's 6.3 s per 50
  steps of 24 scaled to 50 micro-batches of 12). The run is killed and the phase stops.
- **breakage**: a condition's first new seed has M1 < 0.854142; its later seeds are
  skipped.
- **deadline**: a run is not started if the longest run of its condition so far would
  end past the start of the wind-down (the first run of a condition always starts).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ORDER = [("base12", 0), ("long12", 3), ("base12", 1), ("long12", 4), ("base12", 2),
         ("long12", 5), ("base12", 3), ("long12", 6), ("base12", 4), ("long12", 7),
         ("base12", 5), ("base12", 6), ("base12", 7)]
DATA = {"base12": "data/train_v2b.jsonl", "long12": "data/l2x2/train_long.jsonl"}
REFERENCE_50 = {"base12": 3.15, "long12": 4.4}
FIRST_NEW = {"base12": 0, "long12": 3}
M1_FLOOR = 0.854142
LOG = Path("runs/seed8/train.log")
STEP50 = re.compile(r"epoch 0 step 50/\d+ .* elapsed ([\d.]+)s")


def log(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {line}\n")
    print(f"{stamp}  {line}", flush=True)


def train(cond: str, seed: int) -> bool:
    run_id = f"seed8_{cond}_seed{seed}"
    log_path = Path(f"runs/seed8/train_{cond}_seed{seed}.log")
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    command = ["uv", "run", "python", "scripts/train.py", "--train", DATA[cond],
               "--val", "data/val_v2.jsonl", "--encoding", "joint", "--epochs", "2",
               "--batch-size", "12", "--grad-accum", "2", "--lr", "2e-5", "--head-lr", "2e-4",
               "--seed", str(seed), "--max-vram-fraction", "0.88",
               "--input-order", "question_first", "--save", "--run-id", run_id]
    log(f"train start {run_id}: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(command, stdout=fh, stderr=subprocess.STDOUT, env=env)
    checked = False
    while proc.poll() is None:
        time.sleep(5)
        if checked:
            continue
        match = STEP50.search(log_path.read_text(encoding="utf-8", errors="replace"))
        if match:
            checked = True
            seconds = float(match.group(1))
            limit = 2 * REFERENCE_50[cond]
            log(f"{run_id} first 50 micro-batches {seconds:.1f}s, limit {limit:.2f}s")
            if seconds > limit:
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True)
                log(f"STOP {run_id}: slower than 2x the reference (paging suspected)")
                return False
    if proc.returncode != 0:
        log(f"STOP {run_id}: exit code {proc.returncode}")
        return False
    report = Path(f"runs/{run_id}/report.md").read_text(encoding="utf-8")
    peak = re.findall(r"peak_alloc ([\d.]+)GiB peak_reserved ([\d.]+)GiB",
                      log_path.read_text(encoding="utf-8", errors="replace"))
    seconds = re.search(r"\| 学習 \| (\d+) 秒", report)
    stamp = re.search(r"実行日時: (\S+)", report)
    log(f"train done {run_id}: {seconds.group(1) if seconds else '?'} s training, "
        f"peak alloc/reserved {peak[-1] if peak else '?'} GiB, "
        f"report {stamp.group(1) if stamp else '?'}")
    return True


def evaluate(cond: str, seed: int) -> dict:
    run_id = f"seed8_{cond}_seed{seed}"
    out = Path(f"runs/seed8/eval_{cond}_seed{seed}.json")
    log(f"eval start {run_id}")
    with open(f"runs/seed8/eval_{cond}_seed{seed}.log", "w", encoding="utf-8") as fh:
        code = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.eval_local_attention", "--checkpoints",
             f"runs/{run_id}/model.pt", "--out", str(out)],
            stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    if code != 0:
        raise SystemExit(f"evaluation of {run_id} failed: {code}")
    entry = json.loads(out.read_text(encoding="utf-8"))[0]
    assert entry["input_order"] == "question_first" and entry["local_attention"] == 128
    assert not entry["inference_only_override"]
    log(f"eval done {run_id}: M1 {entry['M1']['auroc']:.6f} M2 {entry['M2']['auroc']:.6f} "
        f"M5 {entry['M1_by_attribute']['implies_declining']['auroc']:.6f}")
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()

    broken: set[str] = set()
    longest: dict[str, float] = {}
    for cond, seed in ORDER:
        if cond in broken:
            log(f"skip {cond} seed {seed}: condition broken")
            continue
        if cond in longest and time.time() + longest[cond] > args.deadline_epoch:
            log(f"STOP deadline: {cond} seed {seed} would end past the wind-down "
                f"(longest {cond} run {longest[cond]:.0f}s)")
            return 0
        began = time.time()
        if not train(cond, seed):
            return 1
        entry = evaluate(cond, seed)
        longest[cond] = max(longest.get(cond, 0.0), time.time() - began)
        if seed == FIRST_NEW[cond]:
            m1 = entry["M1"]["auroc"]
            if m1 < M1_FLOOR:
                broken.add(cond)
                log(f"BROKEN {cond}: seed {seed} M1 {m1:.6f} < {M1_FLOOR}")
            else:
                log(f"not broken {cond}: seed {seed} M1 {m1:.6f} >= {M1_FLOOR}")
    log("all runs complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
