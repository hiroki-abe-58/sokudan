"""One timed v0.1 cycle (train + external evaluation) for docs/perf_a.md.

    uv run python scripts/perf_run.py --run-id perf_a_seed0 --seed 0

v0.1's command (question_first, batch 24, the usual VRAM cap), then
`scripts/eval_local_attention.py --save-probs` on the checkpoint, started the moment the
training process exits (`Popen.wait`, no polling interval: A7). Every line the training
prints is stamped on arrival, and `nvidia-smi dmon -s u -d 1` samples SM utilisation for
the whole cycle. The breakdown uses the same four parts as the diagnosis
(the local performance diagnosis of 2026-09-26, §3.1; not in this repository):

- training loop: the two `epoch N done in Xs` values;
- val evaluation inside training: before training (from the `train ... steps` line to
  the `before training` line) and after each epoch (the epoch's wall time minus X);
- external evaluation: the evaluation process, start to exit;
- the rest (start-up, model load, saving, switching): the cycle minus the three above.

Writes `runs/perf/<run-id>.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

DONE = re.compile(r"epoch (\d+) done in (\d+)s")


def dmon(path: Path) -> subprocess.Popen:
    fh = path.open("w", encoding="utf-8")
    return subprocess.Popen(["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-o", "DT"],
                            stdout=fh, stderr=subprocess.STDOUT)


def read_dmon(path: Path) -> list[tuple[float, float]]:
    samples = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4 or line.startswith("#"):
            continue
        try:
            stamp = datetime.strptime(f"{parts[0]} {parts[1]}", "%Y%m%d %H:%M:%S").timestamp()
            samples.append((stamp, float(parts[3])))
        except ValueError:
            continue
    return samples


def mean_sm(samples: list[tuple[float, float]], lo: float, hi: float) -> float | None:
    values = [sm for t, sm in samples if lo <= t <= hi]
    return round(sum(values) / len(values), 1) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--extra", nargs="*", default=[], help="extra train.py arguments")
    args = parser.parse_args()
    out_dir = Path("runs/perf")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": "garbage_collection_threshold:0.8,max_split_size_mb:256"}
    monitor = dmon(out_dir / f"{args.run_id}.dmon.txt")
    stamps: list[tuple[float, str]] = []
    command = [sys.executable, "scripts/train.py", "--train", "data/train_v2b.jsonl",
               "--val", "data/val_v2.jsonl", "--encoding", "joint", "--epochs", "2",
               "--batch-size", "24", "--lr", "2e-5", "--head-lr", "2e-4",
               "--seed", str(args.seed), "--max-vram-fraction", "0.88",
               "--input-order", "question_first", "--save", "--run-id", args.run_id,
               *args.extra]
    cycle_start = time.time()
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, text=True, encoding="utf-8", errors="replace",
                            bufsize=1)
    log = (out_dir / f"{args.run_id}.train.log").open("w", encoding="utf-8")

    def pump() -> None:
        for line in proc.stdout:
            now = time.time()
            stamps.append((now, line.rstrip("\n")))
            log.write(f"{now:.3f} {line}")
            log.flush()

    reader = threading.Thread(target=pump)
    reader.start()
    code = proc.wait()
    reader.join()
    train_end = time.time()
    if code != 0:
        monitor.terminate()
        raise SystemExit(f"training failed: {code}")
    eval_out = out_dir / f"{args.run_id}.eval.json"
    eval_start = time.time()
    with (out_dir / f"{args.run_id}.eval.log").open("w", encoding="utf-8") as fh:
        code = subprocess.run([sys.executable, "-m", "scripts.eval_local_attention",
                               "--checkpoints", f"runs/{args.run_id}/model.pt",
                               "--out", str(eval_out), "--save-probs"],
                              stdout=fh, stderr=subprocess.STDOUT).returncode
    eval_end = time.time()
    time.sleep(2)
    monitor.terminate()
    if code != 0:
        raise SystemExit(f"evaluation failed: {code}")

    def when(pattern: str) -> float:
        return next(t for t, line in stamps if re.search(pattern, line))

    start_line = when(r"^train \d+ examples")
    before_line = when(r"before training:")
    epochs = [(t, int(m.group(1)), float(m.group(2)))
              for t, line in stamps if (m := DONE.search(line))]
    loop_seconds = sum(e[2] for e in epochs)
    val_seconds = [before_line - start_line]
    epoch_start = before_line
    windows = []
    for t, _, seconds in epochs:
        val_seconds.append(t - epoch_start - seconds)
        windows.append((epoch_start, epoch_start + seconds))
        epoch_start = t
    samples = read_dmon(out_dir / f"{args.run_id}.dmon.txt")
    loop_sm = [sm for t, sm in samples if any(lo <= t <= hi for lo, hi in windows)]
    total = eval_end - cycle_start
    external = eval_end - eval_start
    result = {
        "run_id": args.run_id, "seed": args.seed, "extra": args.extra,
        "cycle_seconds": round(total, 1),
        "training_loop_seconds": loop_seconds,
        "epoch_seconds": [e[2] for e in epochs],
        "val_eval_seconds": [round(v, 1) for v in val_seconds],
        "val_eval_total": round(sum(val_seconds), 1),
        "external_eval_seconds": round(external, 1),
        "rest_seconds": round(total - loop_seconds - sum(val_seconds) - external, 1),
        "train_process_seconds": round(train_end - cycle_start, 1),
        "sm_percent_training_loop": (round(sum(loop_sm) / len(loop_sm), 1)
                                     if loop_sm else None),
        "sm_samples_training_loop": len(loop_sm),
        "sm_percent_external_eval": mean_sm(samples, eval_start, eval_end),
        "eval": str(eval_out),
    }
    (out_dir / f"{args.run_id}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
