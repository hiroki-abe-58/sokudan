"""docs/latency.md: v0.2's time per request and batch throughput, on the GPU and the CPU.

    uv run python -m scripts.latency_v02 --device cuda
    uv run python -m scripts.latency_v02 --device cpu [--threads N]

- **Per request**: `sokudan.load(runs/release_candidate/model.pt)` and
  `agent.predict({"body": state}, bench_questions())` -- the three bench_ja questions
  (choice 4, score 3, bool) about one item -- for each of the 300 bench_ja items, one call
  at a time. The first `--warmup` calls (items 0.. in file order, then the 300 again from
  the start) are not timed. Each call is timed with `time.perf_counter` around `predict`
  (which returns host numbers, so the GPU work is finished). Median and p95 over 300.
- **Throughput**: the same 900 questions as rows of 32 (per question type, items in file
  order), through the same forward path as `predict` (JointCollator, fp32, no autocast),
  after one untimed pass; questions per second over the whole timed pass (CUDA
  synchronised).
- Nothing is scored: no label is read and no accuracy is computed.

Writes `runs/latency/v02_<device>.json`.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

import sokudan.config  # noqa: F401
from sokudan.eval.bench_ja import BenchItem, bench_questions
from sokudan.schema.question import parse_question
from sokudan.train.dataset import Example
from sokudan.train.loop import TrainConfig, build_collator, run_model

CHECKPOINT = "runs/release_candidate/model.pt"


def items() -> list[BenchItem]:
    return [BenchItem(**json.loads(line)) for line in
            Path("data/bench_ja.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    import sokudan

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cuda", "cpu"), required=True)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    bench = items()
    questions = bench_questions()
    agent = sokudan.load(CHECKPOINT, device=args.device, temperatures=None)  # as measured
    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)

    for item in bench[:args.warmup]:
        agent.predict({"body": item.state}, questions)
    per_call = []
    for item in bench:
        sync()
        started = time.perf_counter()
        agent.predict({"body": item.state}, questions)
        per_call.append(time.perf_counter() - started)

    # throughput: rows of 32 through the predict forward path
    collator = build_collator(agent.tokenizer, TrainConfig(device=args.device,
                                                           encoding=agent.encoding))
    specs = [("choice", parse_question(questions["department"])),
             ("score", parse_question(questions["urgency"])),
             ("bool", parse_question(questions["churn"]))]
    rows = [[Example(state=it.state, question=q, label=0, ordered=kind == "score",
                     doc_id=it.item_id, domain="bench_ja", attribute=kind, kind=kind)
             for it in bench] for kind, q in specs]

    def one_pass() -> int:
        n = 0
        with torch.no_grad():
            for group_rows in rows:
                for start in range(0, len(group_rows), 32):
                    chunk = group_rows[start:start + 32]
                    batch = collator(chunk).to(args.device)
                    run_model(agent.model, batch).probs.float().cpu()
                    n += len(chunk)
        return n

    one_pass()  # untimed
    sync()
    started = time.perf_counter()
    n = one_pass()
    sync()
    seconds = time.perf_counter() - started

    ms = np.array(per_call) * 1000
    res = {
        "checkpoint": CHECKPOINT, "device": args.device,
        "device_name": (torch.cuda.get_device_name(0) if args.device == "cuda"
                        else platform.processor()),
        "torch": torch.__version__, "threads": torch.get_num_threads(),
        "warmup_calls": args.warmup, "n_calls": len(per_call),
        "per_request_ms": {"median": float(np.median(ms)), "p95": float(np.percentile(ms, 95)),
                           "mean": float(ms.mean()), "min": float(ms.min()),
                           "max": float(ms.max())},
        "throughput": {"batch_size": 32, "questions": n, "seconds": seconds,
                       "questions_per_s": n / seconds},
        "questions_per_request": 3, "dtype": "float32 (predict path, no autocast)",
    }
    out = Path(f"runs/latency/v02_{args.device}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
