"""The MLX backend: the joint model on Apple silicon, without torch.

Reads a safetensors checkpoint directory (the Hub layout); a `.pt` file and the separate
encoding need the torch backend. `dtype` casts or quantizes the backbone only; the heads
stay float32 (`sokudan.backends.mlx.model`).
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from sokudan.backends import Backend, Batch, checkpoint_config
from sokudan.backends.mlx.model import JointModel, convert_weights
from sokudan.backends.mlx.modernbert import EncoderConfig
from sokudan.config import BACKBONE_MODEL_ID

QUANTIZED = {"8bit-g64": (8, 64), "8bit-g32": (8, 32), "4bit-g64": (4, 64),
             "4bit-g32": (4, 32)}
"""Quantized backbones: `nn.quantize(bits, group_size)` over a float16 backbone."""

VARIANTS = ("float32", "float16", *QUANTIZED)
"""Every backbone precision this module can build."""

DTYPES = ("float32",)
"""The ones `sokudan.load(backend="mlx", dtype=...)` accepts."""

DEFAULT_DTYPE = "float32"


def _set_precision(backbone: nn.Module, dtype: str) -> None:
    if dtype == "float32":
        return
    backbone.set_dtype(mx.float16)
    if dtype in QUANTIZED:
        bits, group_size = QUANTIZED[dtype]
        nn.quantize(backbone, group_size=group_size, bits=bits)


class MLXBackend(Backend):
    name = "mlx"
    encoding = "joint"

    def __init__(self, model: JointModel, *, dtype: str, backbone_id: str,
                 input_order: str) -> None:
        self.model = model
        self.dtype = dtype
        self.backbone_id = backbone_id
        self.input_order = input_order
        self.device = mx.default_device().type.name

    @classmethod
    def load(cls, path: Path, *, dtype: str | None = None) -> MLXBackend:
        """Build the joint model from a checkpoint directory; `dtype` from `VARIANTS`."""
        dtype = dtype or DEFAULT_DTYPE
        if dtype not in VARIANTS:
            raise ValueError(f"unknown MLX dtype {dtype!r}; known: {VARIANTS}")
        if path.is_file():
            raise ValueError(f"{path} is a torch checkpoint; the MLX backend reads a "
                             "directory with model.safetensors")
        config = checkpoint_config(path)
        encoding = config.get("encoding", "separate")
        if encoding != "joint":
            raise NotImplementedError(f"the MLX backend runs the joint encoding only, "
                                      f"this checkpoint is {encoding!r}")
        from transformers import AutoConfig

        backbone_id = config.get("backbone", BACKBONE_MODEL_ID)
        encoder = AutoConfig.from_pretrained(backbone_id).to_dict()
        # None keeps the pretrained window, as in `Backbone.load`.
        if config.get("local_attention") is not None:
            encoder["local_attention"] = int(config["local_attention"])

        model = JointModel(EncoderConfig.from_dict(encoder))
        weights = convert_weights(mx.load(str(path / "model.safetensors")))
        model.load_weights(list(weights.items()), strict=True)
        _set_precision(model.backbone, dtype)
        model.eval()
        mx.eval(model.parameters())
        return cls(model, dtype=dtype, backbone_id=backbone_id,
                   input_order=config.get("input_order", "question_first"))

    def probs(self, batch: Batch) -> np.ndarray:
        if batch.state_ids is not None:
            raise NotImplementedError("the MLX backend runs the joint encoding only")

        def put(array: np.ndarray) -> mx.array:
            return mx.array(array.astype(np.int32))

        back = (None if batch.marker_positions_back is None
                else put(batch.marker_positions_back))
        out = self.model(put(batch.input_ids), put(batch.attention_mask),
                         put(batch.marker_positions), put(batch.marker_mask),
                         mx.array(batch.ordered), back)
        return np.array(out)
