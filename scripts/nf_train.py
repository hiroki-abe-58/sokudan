"""Training for docs/noise_floor.md: more seeds of an existing batch-24 configuration.

    uv run python scripts/nf_train.py --cond v01 --seeds 3 4 5 6 7 --deadline-epoch <t>
    uv run python scripts/nf_train.py --cond io_sf --seeds 3 4 5 6 7 --deadline-epoch <t>

`v01` is v0.1's exact command (question_first); `io_sf` is the same with
`--input-order state_first`. Batch 24, no gradient accumulation, no
`--local-attention`, under the VRAM cap every run now uses (a memory policy only). Run
ids continue the existing ones (`runs/v01_seed<s>`, `runs/io_sf_seed<s>`), and each
checkpoint is evaluated into `runs/nf/eval_<cond>_seed<s>.json`.

Stops (logged to `runs/nf/train.log` with the time read at that moment):
- the logged loss is NaN or inf;
- the first 50 steps take more than twice the earlier measurement of the same
  configuration (v01: 6.3 s, io_sf: 6.1 s, both 2026-09-24);
- a run would end past the wind-down, judged by the longest run so far.
There is deliberately no M1 breakage stop here (the point is to measure the tail).
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

ORDER = {"v01": "question_first", "io_sf": "state_first"}
REFERENCE_50 = {"v01": 6.3, "io_sf": 6.1}
LOG = Path("runs/nf/train.log")
STEP = re.compile(r"epoch (\d+) step (\d+)/\d+ loss (\S+) elapsed ([\d.]+)s")


def log(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {line}\n")
    print(f"{stamp}  {line}", flush=True)


def kill(proc: subprocess.Popen) -> None:
    subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)


def train(cond: str, seed: int) -> bool:
    run_id = f"{cond}_seed{seed}"
    log_path = Path(f"runs/nf/train_{run_id}.log")
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    command = ["uv", "run", "python", "scripts/train.py", "--train", "data/train_v2b.jsonl",
               "--val", "data/val_v2.jsonl", "--encoding", "joint", "--epochs", "2",
               "--batch-size", "24", "--lr", "2e-5", "--head-lr", "2e-4",
               "--seed", str(seed), "--max-vram-fraction", "0.88",
               "--input-order", ORDER[cond], "--save", "--run-id", run_id]
    log(f"train start {run_id}: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(command, stdout=fh, stderr=subprocess.STDOUT, env=env)
    checked = False
    while proc.poll() is None:
        time.sleep(5)
        text = log_path.read_text(encoding="utf-8", errors="replace")
        steps = STEP.findall(text)
        for _, _, loss, _ in steps:
            value = float(loss)
            if math.isnan(value) or math.isinf(value):
                kill(proc)
                log(f"STOP {run_id}: loss {loss}")
                return False
        if not checked:
            first = [s for s in steps if s[0] == "0" and s[1] == "50"]
            if first:
                checked = True
                seconds = float(first[0][3])
                limit = 2 * REFERENCE_50[cond]
                log(f"{run_id} first 50 steps {seconds:.1f}s, limit {limit:.1f}s")
                if seconds > limit:
                    kill(proc)
                    log(f"STOP {run_id}: slower than 2x the earlier run (paging suspected)")
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


def evaluate(cond: str, seed: int) -> None:
    run_id = f"{cond}_seed{seed}"
    out = Path(f"runs/nf/eval_{run_id}.json")
    log(f"eval start {run_id}")
    with open(f"runs/nf/eval_{run_id}.log", "w", encoding="utf-8") as fh:
        code = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.eval_local_attention", "--checkpoints",
             f"runs/{run_id}/model.pt", "--out", str(out)],
            stdout=fh, stderr=subprocess.STDOUT,
        ).returncode
    if code != 0:
        raise SystemExit(f"evaluation of {run_id} failed: {code}")
    e = json.loads(out.read_text(encoding="utf-8"))[0]
    assert e["input_order"] == ORDER[cond] and e["local_attention"] == 128
    assert not e["inference_only_override"]
    log(f"eval done {run_id}: M1 {e['M1']['auroc']:.6f} M2 {e['M2']['auroc']:.6f} "
        f"M5 {e['M1_by_attribute']['implies_declining']['auroc']:.6f} "
        f"ends_with_question {e['M1_by_attribute']['ends_with_question']['auroc']:.6f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", choices=sorted(ORDER), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    args = parser.parse_args()
    longest = 0.0
    for seed in args.seeds:
        if longest and time.time() + longest > args.deadline_epoch:
            log(f"STOP deadline: {args.cond} seed {seed} would end past the wind-down "
                f"(longest run {longest:.0f}s)")
            return 0
        began = time.time()
        if not train(args.cond, seed):
            return 1
        evaluate(args.cond, seed)
        longest = max(longest, time.time() - began)
    log(f"{args.cond} seeds {args.seeds} complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
