"""Gate A (SOKUDAN_SPEC.md §4.1): can we use FlashAttention-2 on sm_120?

ModernBERT's unpadding path depends on flash-attn. If no sm_120 wheel builds, we
fall back to `attn_implementation="sdpa"` with padded batches -- and we are required
to *measure* that fallback's throughput before proceeding, not assume it.

Usage:
    uv run python scripts/gate_a_attention.py                  # measure only
    uv run python scripts/gate_a_attention.py --json out.json

Every number printed by this script is measured on this machine. Nothing is estimated.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

from sokudan.config import BACKBONE_MODEL_ID

# Sequence lengths that matter for this design (§5.2, §6.2):
#   128  -- ModernBERT local-attention window; a question block should fit near here
#   256  -- stated upper bound for the question side
#   512  -- a short state
#   1024 -- a realistic support-email state
SEQ_LENS = (128, 256, 512, 1024)
BATCH = 32
WARMUP = 3
ITERS = 10


@dataclass
class ThroughputRow:
    attn_implementation: str
    seq_len: int
    batch_size: int
    iters: int
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    sequences_per_sec: float
    tokens_per_sec: float
    peak_vram_mib: float


def flash_attn_status() -> dict[str, Any]:
    """Report whether flash-attn is importable, without pretending to know why not."""
    spec = importlib.util.find_spec("flash_attn")
    if spec is None:
        return {"importable": False, "version": None, "detail": "module not found"}
    try:
        import flash_attn  # noqa: PLC0415

        return {"importable": True, "version": getattr(flash_attn, "__version__", "unknown"),
                "detail": "import succeeded"}
    except Exception as exc:  # pragma: no cover - environment dependent
        return {"importable": False, "version": None,
                "detail": f"{type(exc).__name__}: {exc}"}


def measure(attn_implementation: str, seq_len: int) -> ThroughputRow:
    torch.manual_seed(0)
    model = AutoModel.from_pretrained(
        BACKBONE_MODEL_ID,
        attn_implementation=attn_implementation,
        dtype=torch.bfloat16,
    ).cuda().eval()

    tok = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    vocab = tok.vocab_size
    input_ids = torch.randint(0, vocab, (BATCH, seq_len), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        for _ in range(WARMUP):
            model(input_ids=input_ids, attention_mask=attention_mask)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        samples_ms: list[float] = []
        for _ in range(ITERS):
            start = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            torch.cuda.synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000.0)

    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    mean_ms = statistics.fmean(samples_ms)
    ordered = sorted(samples_ms)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]

    del model
    torch.cuda.empty_cache()

    return ThroughputRow(
        attn_implementation=attn_implementation,
        seq_len=seq_len,
        batch_size=BATCH,
        iters=ITERS,
        latency_ms_mean=round(mean_ms, 3),
        latency_ms_p50=round(p50, 3),
        latency_ms_p95=round(p95, 3),
        sequences_per_sec=round(BATCH / (mean_ms / 1000.0), 1),
        tokens_per_sec=round(BATCH * seq_len / (mean_ms / 1000.0), 1),
        peak_vram_mib=round(peak_mib, 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default=None, help="write raw results here")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device. Gate A cannot be evaluated.")
        return 1

    env = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(0),
        "device_capability": list(torch.cuda.get_device_capability(0)),
        "arch_list": torch.cuda.get_arch_list(),
        "backbone": BACKBONE_MODEL_ID,
    }
    print("== environment ==")
    for key, value in env.items():
        print(f"{key:20s} {value}")

    fa2 = flash_attn_status()
    print("\n== Gate A: flash-attn ==")
    print(f"importable: {fa2['importable']}  version: {fa2['version']}  detail: {fa2['detail']}")

    impls = ["sdpa", "eager"]
    if fa2["importable"]:
        impls.insert(0, "flash_attention_2")

    rows: list[ThroughputRow] = []
    failures: dict[str, str] = {}
    print("\n== throughput (measured) ==")
    header = (f"{'impl':>18} {'seq':>5} {'bs':>4} {'ms/mean':>9} {'ms/p95':>8} "
              f"{'seq/s':>9} {'tok/s':>11} {'VRAM MiB':>9}")
    print(header)
    print("-" * len(header))
    for impl in impls:
        for seq_len in SEQ_LENS:
            try:
                row = measure(impl, seq_len)
            except Exception as exc:
                failures[f"{impl}@{seq_len}"] = f"{type(exc).__name__}: {exc}"
                print(f"{impl:>18} {seq_len:>5} {'--':>4} {'FAILED':>9}  {type(exc).__name__}")
                continue
            rows.append(row)
            print(f"{row.attn_implementation:>18} {row.seq_len:>5} {row.batch_size:>4} "
                  f"{row.latency_ms_mean:>9.2f} {row.latency_ms_p95:>8.2f} "
                  f"{row.sequences_per_sec:>9.1f} {row.tokens_per_sec:>11.1f} "
                  f"{row.peak_vram_mib:>9.1f}")

    result = {"env": env, "flash_attn": fa2,
              "rows": [asdict(r) for r in rows], "failures": failures}
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.json}")

    print("\n== Gate A verdict ==")
    if fa2["importable"] and any(r.attn_implementation == "flash_attention_2" for r in rows):
        print("PASS: flash_attention_2 available and measured.")
    else:
        print("FALLBACK: flash-attn unavailable -> use attn_implementation='sdpa' "
              "with padded, length-bucketed batches (§6.2).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
