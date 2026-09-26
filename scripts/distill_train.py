"""Students for docs/distill.md: v0.1's command plus distillation, seeds 0-7, batch 24.

    uv run python scripts/distill_train.py --deadline-epoch <unix time>

v0.1's command (question_first, batch 24, no accumulation, no `--local-attention`, the
usual VRAM cap) with `--distill-targets data/cache/distill/teacher_E_mix16.npz
--distill-alpha 0.5`. Run ids `runs/distill_seed<s>`, evaluation into
`runs/distill/eval_distill_seed<s>.json`. Stops, logged to `runs/distill/train.log`:

- a non-zero exit, including CUDA out of memory (no fallback to accumulation:
  docs/research_protocol.md §6);
- the logged loss is NaN / inf;
- the first 50 steps take more than 2x v0.1's 6.3 s (12.6 s); above 1.5x (9.45 s) it
  is only recorded;
- seed 0's M1m below 0.791169 (docs/research_protocol.md §5) stops the condition;
- a run that would end past the wind-down, judged by the longest run so far.
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
LOG = Path("runs/distill/train.log")
TARGETS = "data/cache/distill/teacher_E_mix16.npz"
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
    run_id = f"distill_seed{seed}"
    log_path = Path(f"runs/distill/train_{run_id}.log")
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    command = ["uv", "run", "python", "scripts/train.py", "--train", "data/train_v2b.jsonl",
               "--val", "data/val_v2.jsonl", "--encoding", "joint", "--epochs", "2",
               "--batch-size", "24", "--lr", "2e-5", "--head-lr", "2e-4",
               "--seed", str(seed), "--max-vram-fraction", "0.88",
               "--input-order", "question_first", "--distill-targets", TARGETS,
               "--distill-alpha", "0.5", "--save", "--run-id", run_id]
    log(f"train start {run_id}: {' '.join(command)}")
    with log_path.open("w", encoding="utf-8") as fh:
        proc = subprocess.Popen(command, stdout=fh, stderr=subprocess.STDOUT, env=env)
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
    report = Path(f"runs/{run_id}/report.md").read_text(encoding="utf-8")
    peak = re.findall(r"peak_alloc ([\d.]+)GiB peak_reserved ([\d.]+)GiB",
                      log_path.read_text(encoding="utf-8", errors="replace"))
    seconds = re.search(r"\| 学習 \| (\d+) 秒", report)
    stamp = re.search(r"実行日時: (\S+)", report)
    log(f"train done {run_id}: {seconds.group(1) if seconds else '?'} s training, "
        f"peak alloc/reserved {peak[-1] if peak else '?'} GiB, "
        f"report {stamp.group(1) if stamp else '?'}")
    return True


def evaluate(seed: int) -> dict:
    run_id = f"distill_seed{seed}"
    out = Path(f"runs/distill/eval_{run_id}.json")
    log(f"eval start {run_id}")
    with open(f"runs/distill/eval_{run_id}.log", "w", encoding="utf-8") as fh:
        code = subprocess.run(
            ["uv", "run", "python", "-m", "scripts.eval_local_attention", "--checkpoints",
             f"runs/{run_id}/model.pt", "--out", str(out)],
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
    for seed in range(8):
        if longest and time.time() + longest > args.deadline_epoch:
            log(f"STOP deadline: seed {seed} would end past the wind-down "
                f"(longest run {longest:.0f}s)")
            return 0
        began = time.time()
        if not train(seed):
            return 1
        entry = evaluate(seed)
        longest = max(longest, time.time() - began)
        if seed == 0:
            value = m1m(entry)
            if value < M1M_FLOOR:
                log(f"BROKEN distill: seed 0 M1m {value:.6f} < {M1M_FLOOR}")
                return 0
            log(f"not broken: seed 0 M1m {value:.6f} >= {M1M_FLOOR}")
    log("distill seeds 0-7 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
