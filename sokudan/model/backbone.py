"""ModernBERT-Ja wrapper (SOKUDAN_SPEC.md §6.2).

One backbone, shared by the state side and the question side. Not two models: the
weights are the same, only the input differs. Keeping it shared is what makes the
question side's marker positions comparable to what the state encoder produces, and
it halves the parameters we have to train.

`attn_implementation` defaults to `sdpa` because Gate A could not build flash-attn on
this machine -- the system CUDA toolkit is 13.1 while torch is built against 12.8, so
`torch.utils.cpp_extension` refuses. Measured fallback throughput is in
`docs/gate_a.md`; sdpa beats eager by 1.13x to 2.18x over the sequence lengths that
matter here, and the numbers cleared the bar for the day's training budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from transformers import AutoConfig, AutoModel

from sokudan.config import BACKBONE_MODEL_ID


@dataclass(frozen=True)
class BackboneSpec:
    """The config values the head needs to match the backbone."""

    hidden_size: int
    num_attention_heads: int
    num_hidden_layers: int
    intermediate_size: int
    norm_eps: float
    local_attention: int
    global_attn_every_n_layers: int
    max_position_embeddings: int

    @classmethod
    def from_config(cls, config: object) -> BackboneSpec:
        return cls(
            hidden_size=int(config.hidden_size),
            num_attention_heads=int(config.num_attention_heads),
            num_hidden_layers=int(config.num_hidden_layers),
            intermediate_size=int(config.intermediate_size),
            norm_eps=float(getattr(config, "norm_eps", 1e-5)),
            local_attention=int(getattr(config, "local_attention", -1)),
            global_attn_every_n_layers=int(getattr(config, "global_attn_every_n_layers", 1)),
            max_position_embeddings=int(getattr(config, "max_position_embeddings", 8192)),
        )


class Backbone(nn.Module):
    """Encodes a batch of sequences to per-token hidden states."""

    def __init__(self, model: nn.Module, spec: BackboneSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec

    @classmethod
    def load(
        cls,
        model_id: str = BACKBONE_MODEL_ID,
        *,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.float32,
    ) -> Backbone:
        model = AutoModel.from_pretrained(
            model_id, attn_implementation=attn_implementation, dtype=dtype
        )
        return cls(model, BackboneSpec.from_config(model.config))

    @classmethod
    def spec_only(cls, model_id: str = BACKBONE_MODEL_ID) -> BackboneSpec:
        """Read the config without downloading weights. Useful in tests."""
        return BackboneSpec.from_config(AutoConfig.from_pretrained(model_id))

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    @property
    def layers(self) -> nn.ModuleList:
        """The encoder layers, for the head's copy-init (§6.2)."""
        return self.model.layers

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
