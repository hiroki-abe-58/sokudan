"""Phase 4 of docs/length_2x2.md: train and evaluate qf-long / sf-long, 3 seeds each.

    uv run python scripts/l2x2_train_all.py

Order: qf-long s0, sf-long s0, then the breakage stop per condition (seed 0 M1 below
0.854142 -> that condition's other seeds are not trained), then the remaining seeds.

Each run is v0.1's command with the long training set and the condition's input order
(no `--local-attention`), under the same VRAM cap as every earlier run. While it
trains, the log is watched for step 50: if the seconds per step exceed twice the
short-data run in the same order (qf 6.3 s / 50, sf 6.1 s / 50, both measured
2026-09-24), the run is killed and the whole phase stops for a manual retry
decision. After training the checkpoint is evaluated with
`scripts/eval_local_attention.py` into `runs/l2x2/eval_<cond>_seed<s>.json`.

One line per event goes to `runs/l2x2/train_all.log`, with the time read at that moment.

**Memory fallback (applied from 2026-09-25 01:03).** qf-long seed 0 ran out of memory
under the 0.88 cap at batch 24 (the long rows reach 875 tokens). The pre-registered
retry -- batch 12 with gradient accumulation 2, once -- is used for every long run, so
both long conditions train the same way. The speed limit is then compared per example
(24 examples = one v0.1 step), which is the "per step" of the rule.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

CONDITIONS = {"qf_long": "question_first", "sf_long": "state_first"}
SHORT_50_STEPS = {"qf_long": 6.3, "sf_long": 6.1}
M1_FLOOR = 0.854142
TRAIN = "data/l2x2/train_long.jsonl"
LOG = Path("runs/l2x2/train_all.log")
STEP50 = re.compile(r"epoch 0 step 50/\d+ .* elapsed ([\d.]+)s")
BATCH, ACCUM = 12, 2  # the memory fallback; 24 examples per optimizer step as in v0.1


def log(line: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(f"{stamp}  {line}\n")
    print(f"{stamp}  {line}", flush=True)


def kill_tree(pid: int) -> None:
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)


def train(cond: str, seed: int) -> bool:
    run_id = f"l2x2_{cond}_seed{seed}"
    log_path = Path(f"runs/l2x2/train_{cond}_seed{seed}.log")
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    command = ["uv", "run", "python", "scripts/train.py", "--train", TRAIN,
               "--val", "data/val_v2.jsonl", "--encoding", "joint", "--epochs", "2",
               "--batch-size", str(BATCH), "--grad-accum", str(ACCUM),
               "--lr", "2e-5", "--head-lr", "2e-4",
               "--seed", str(seed), "--max-vram-fraction", "0.88",
               "--input-order", CONDITIONS[cond], "--save", "--run-id", run_id]
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
            per_example = seconds / (50 * BATCH)
            limit = 2 * SHORT_50_STEPS[cond] / (50 * 24)
            log(f"{run_id} first 50 micro-batches {seconds:.1f}s = {per_example * 24:.4f} s "
                f"per 24 examples, limit {limit * 24:.4f} s per 24 examples")
            seconds, limit = per_example, limit
            if seconds > limit:
                kill_tree(proc.pid)
                log(f"STOP {run_id}: slower than 2x the short run (paging suspected)")
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
    run_id = f"l2x2_{cond}_seed{seed}"
    out = Path(f"runs/l2x2/eval_{cond}_seed{seed}.json")
    log(f"eval start {run_id}")
    code = subprocess.run(
        ["uv", "run", "python", "-m", "scripts.eval_local_attention", "--checkpoints",
         f"runs/{run_id}/model.pt", "--out", str(out)],
        stdout=open(f"runs/l2x2/eval_{cond}_seed{seed}.log", "w", encoding="utf-8"),
        stderr=subprocess.STDOUT,
    ).returncode
    if code != 0:
        raise SystemExit(f"evaluation of {run_id} failed: {code}")
    entry = json.loads(out.read_text(encoding="utf-8"))[0]
    assert entry["input_order"] == CONDITIONS[cond] and entry["local_attention"] == 128
    assert not entry["inference_only_override"]
    log(f"eval done {run_id}: M1 {entry['M1']['auroc']:.6f} M2 {entry['M2']['auroc']:.6f} "
        f"M5 {entry['M1_by_attribute']['implies_declining']['auroc']:.6f}")
    return entry


def main() -> int:
    broken = set()
    for cond in CONDITIONS:
        if not train(cond, 0):
            return 1
        m1 = evaluate(cond, 0)["M1"]["auroc"]
        if m1 < M1_FLOOR:
            broken.add(cond)
            log(f"BROKEN {cond}: seed 0 M1 {m1:.6f} < {M1_FLOOR}; its seeds 1-2 are skipped")
        else:
            log(f"not broken {cond}: seed 0 M1 {m1:.6f} >= {M1_FLOOR}")
    for seed in (1, 2):
        for cond in CONDITIONS:
            if cond in broken:
                continue
            if not train(cond, seed):
                return 1
            evaluate(cond, seed)
    log("phase 4 complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
