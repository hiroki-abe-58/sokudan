"""Raw head probabilities (before any temperature) on the parity set, one row per question.

    python scripts/mlx/dump_probs.py --items parity_set.json --out probs.npz \
        --model GeneLab/sokudan-ja-310m [--device cpu] [--backend mlx] [--dtype float16]
        [--mlx-variant 4bit-g32]

Each state's questions go through the model as one batch, the way `Agent.predict` sends
them, through `Agent._forward` -- the probabilities `predict` reads before temperatures
and rounding. The npz holds one array per question, keyed `"{item id}/{question id}"`.
Options left unset are not passed, so the script also runs against code that does not
have them yet. `--mlx-variant` builds the MLX backend directly: `float32`, `float16`, or
a quantized backbone `sokudan.load` does not offer -- `8bit-g64`, `8bit-g32`, `4bit-g64`,
`4bit-g32`: `nn.quantize(bits, group_size)` over the float16 backbone (every Linear and
the token embedding; the float32 heads are left alone).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import numpy as np

import sokudan
from sokudan.predict import _Prepared
from sokudan.schema.question import parse_questions


def no_grad():
    try:
        import torch
    except ImportError:
        return contextlib.nullcontext()
    return torch.no_grad()


QUANTIZED = {"8bit-g64": (8, 64), "8bit-g32": (8, 32), "4bit-g64": (4, 64),
             "4bit-g32": (4, 32)}


def mlx_variant_agent(model: str, variant: str) -> sokudan.Agent:
    import mlx.core as mx
    import mlx.nn as nn
    from transformers import AutoTokenizer

    from sokudan.backends.mlx import MLXBackend
    from sokudan.predict import Agent, locate_checkpoint

    quantized = QUANTIZED.get(variant)
    backend = MLXBackend.load(locate_checkpoint(model),
                              dtype="float16" if quantized else variant)
    if quantized:
        bits, group_size = quantized
        nn.quantize(backend.model.backbone, group_size=group_size, bits=bits)
        mx.eval(backend.model.parameters())
        backend.dtype = variant
    return Agent(backend, AutoTokenizer.from_pretrained(backend.backbone_id),
                 encoding=backend.encoding)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="GeneLab/sokudan-ja-310m")
    parser.add_argument("--device")
    parser.add_argument("--backend")
    parser.add_argument("--dtype")
    parser.add_argument("--mlx-variant")
    args = parser.parse_args()

    if args.mlx_variant:
        agent = mlx_variant_agent(args.model, args.mlx_variant)
    else:
        options = {k: v for k, v in (("device", args.device), ("backend", args.backend),
                                     ("dtype", args.dtype)) if v is not None}
        agent = sokudan.load(args.model, temperatures=None, **options)
    items = json.loads(args.items.read_text(encoding="utf-8"))
    rows: dict[str, np.ndarray] = {}
    start = time.perf_counter()
    with no_grad():
        for item in items:
            prepared = {qid: _Prepared(q, q.type, list(q.labels))
                        for qid, q in parse_questions(item["questions"]).items()}
            raw = agent._forward(item["state"], prepared)[0]
            for qid, probs in raw.items():
                rows[f"{item['id']}/{qid}"] = np.asarray(probs)
    elapsed = time.perf_counter() - start
    np.savez(args.out, **rows)
    described = {k: getattr(agent, k, None) for k in ("backend", "device", "dtype")}
    print(f"{args.out}: {len(rows)} questions, {elapsed:.1f} s, {described}")


if __name__ == "__main__":
    main()
