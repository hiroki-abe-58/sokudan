"""The PyTorch backend: the model sokudan was trained as, on cuda, mps or cpu."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sokudan.backends import Backend, Batch
from sokudan.config import BACKBONE_MODEL_ID


def resolve_device(device: str | None = "auto") -> str:
    """`"auto"` (or None): cuda, then mps, then cpu. Any other value is used as given."""
    if device not in (None, "auto"):
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def read_checkpoint(path: Path) -> dict:
    """`{"state_dict", "config"}` from a `.pt` file or a directory holding
    `model.safetensors` + `config.json` -- the shape `torch.load` returns for a `.pt`."""
    if path.is_file():
        return torch.load(str(path), map_location="cpu", weights_only=False)
    from safetensors.torch import load_file

    config_path = path / "config.json"
    config = (
        json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    )
    return {"state_dict": load_file(str(path / "model.safetensors")), "config": config}


class TorchBackend(Backend):
    name = "torch"
    dtype = "float32"

    def __init__(self, model: Any, device: str, *, encoding: str = "separate",
                 backbone_id: str = BACKBONE_MODEL_ID) -> None:
        self.model = model
        self.device = device
        self.encoding = encoding
        self.backbone_id = backbone_id
        # Not on the separate model; checkpoints that predate the option are v0.1's.
        self.input_order = getattr(model, "input_order", "question_first")

    @classmethod
    def load(cls, path: Path, *, device: str = "auto") -> TorchBackend:
        """Build the model a checkpoint describes and load its weights onto `device`."""
        from sokudan.model.sokudan import SokudanModel

        device = resolve_device(device)
        blob = read_checkpoint(path)
        config = blob.get("config", {})
        backbone_id = config.get("backbone", BACKBONE_MODEL_ID)
        encoding = config.get("encoding", "separate")

        # Absent in checkpoints that predate `--local-attention`; None keeps the
        # pretrained window, which is what those were trained at.
        local_attention = config.get("local_attention")

        if encoding == "joint":
            from sokudan.model.joint import SokudanJointModel

            model = SokudanJointModel.from_pretrained_backbone(
                backbone_id, local_attention=local_attention,
                input_order=config.get("input_order", "question_first"),
            )
        else:
            model = SokudanModel.from_pretrained_backbone(
                backbone_id, n_head_layers=config.get("n_head_layers", 2),
                local_attention=local_attention,
            )
        model.load_state_dict(blob["state_dict"])
        model.to(device).eval()
        return cls(model, device, encoding=encoding, backbone_id=backbone_id)

    @torch.no_grad()
    def probs(self, batch: Batch) -> np.ndarray:
        device = torch.device(self.device)

        def put(array: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(array).to(device)

        if batch.state_ids is None:
            extra = ({} if batch.marker_positions_back is None
                     else {"marker_positions_back": put(batch.marker_positions_back)})
            out = self.model(
                put(batch.input_ids), put(batch.attention_mask),
                put(batch.marker_positions), put(batch.marker_mask), put(batch.ordered),
                **extra,
            )
        else:
            out = self.model(
                put(batch.state_ids), put(batch.state_mask),
                put(batch.input_ids), put(batch.attention_mask),
                put(batch.marker_positions), put(batch.marker_mask), put(batch.ordered),
            )
        return out.probs.float().cpu().numpy()
