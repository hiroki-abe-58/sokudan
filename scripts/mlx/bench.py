"""`Agent.predict` latency, end to end, for one backend configuration (one process).

    python scripts/mlx/bench.py --model GeneLab/sokudan-ja-310m --backend mlx \
        --dtype float16 --out mlx_float16_round1.json

States: the README Quickstart state (short), the same sentence repeated to ~400 state
tokens (long), and to ~950 state tokens (1k: joint sequences of about 1000 tokens).
Questions: q1 = the Quickstart department (choice, 4 options); q3 = q1 + urgency (score,
3 levels) + refund (bool). Per cell: 3 warm-up calls, then 30 timed calls. Also records
the load average at start, peak RSS and, for MLX, `mx.get_peak_memory()`.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np

import sokudan

SHORT = "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。"
Q1 = {"department": {"type": "choice", "instructions": "この問い合わせはどの部署が担当すべきか",
                     "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                                  "営業": "料金・新規契約", "その他": "上記以外"}}}
Q3 = {**Q1,
      "urgency": {"type": "score", "instructions": "この問い合わせの緊急度",
                  "criteria": ["低", "中", "高"]},
      "refund": {"type": "noul", "instructions": "顧客は返金を求めているか"}}
WARMUP, N = 3, 30


def repeated(tokenizer, target: int) -> str:
    """SHORT repeated as consecutive sentences while it stays within `target` tokens."""
    text = SHORT
    while len(tokenizer(text + SHORT, add_special_tokens=False)["input_ids"]) <= target:
        text += SHORT
    return text


def peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak if platform.system() == "Darwin" else peak * 1024  # Linux reports KiB


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--backend", required=True, choices=["torch", "mlx"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    result = {"loadavg_start": os.getloadavg(), "config": vars(args) | {"out": str(args.out)}}
    start = time.perf_counter()
    agent = sokudan.load(args.model, backend=args.backend, device=args.device,
                         dtype=args.dtype)
    result["load_s"] = time.perf_counter() - start
    result["backend"] = {"name": agent.backend.name, "device": agent.device,
                         "dtype": agent.dtype}
    if agent.backend.name == "mlx":
        import mlx.core as mx

        result["mlx"] = {"version": mx.__version__,
                         "peak_memory_after_load": mx.get_peak_memory(),
                         "active_memory_after_load": mx.get_active_memory()}
        mx.reset_peak_memory()
    else:
        import torch

        result["torch"] = {"version": torch.__version__, "threads": torch.get_num_threads()}

    states = {"short": SHORT, "long": repeated(agent.tokenizer, 400),
              "1k": repeated(agent.tokenizer, 950)}
    cells = {}
    for sname, state in states.items():
        for qname, questions in (("q1", Q1), ("q3", Q3)):
            for _ in range(WARMUP):
                agent.predict(state, questions)
            times = []
            for _ in range(N):
                t0 = time.perf_counter()
                out = agent.predict(state, questions)
                times.append((time.perf_counter() - t0) * 1000)
            usage = out["usage"]
            cells[f"{sname}/{qname}"] = {
                "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95)),
                "state_tokens": usage["state_tokens"],
                "joint_tokens_mean": usage["state_tokens"]
                + usage["question_tokens"] / usage["backbone_passes"],
                "times_ms": times,
            }
    result["cells"] = cells
    result["peak_rss_bytes"] = peak_rss_bytes()
    if agent.backend.name == "mlx":
        import mlx.core as mx

        result["mlx"]["peak_memory_during_predict"] = mx.get_peak_memory()
    result["loadavg_end"] = os.getloadavg()
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("backend", "loadavg_start", "load_s")}),
          {c: round(v["median_ms"], 1) for c, v in cells.items()})


if __name__ == "__main__":
    main()
