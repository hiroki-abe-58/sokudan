"""sokudan's joint model in MLX: the vendored ModernBERT, the marker scorer, the ordinal head.

A port of `sokudan.model.joint.SokudanJointModel.forward`, `sokudan.model.head.masked_softmax`
and `sokudan.model.ordinal.OrdinalHead` for inference. The heads run in float32 whatever
the backbone's dtype: the marker states are cast before the scorer, and the five head
tensors are never cast or quantized.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from sokudan.backends.mlx.modernbert import EncoderConfig, ModernBert

# sokudan.model.ordinal.LOGIT_CLAMP
LOGIT_CLAMP = 15.0

BACKBONE_PREFIX = "backbone.model."
"""The torch checkpoint nests the transformers ModernBertModel under `backbone.model`."""


def convert_weights(state: dict[str, mx.array]) -> dict[str, mx.array]:
    """A sokudan joint checkpoint's tensors under this module's parameter names.

    `backbone.model.X` -> `backbone.X` (152 tensors for modernbert-ja-310m); `scorer.weight`,
    `ordinal.cut.{weight,bias}` and `ordinal.location.{weight,bias}` keep their names.
    """
    return {("backbone." + name[len(BACKBONE_PREFIX):]
             if name.startswith(BACKBONE_PREFIX) else name): value
            for name, value in state.items()}


def masked_softmax(logits: mx.array, marker_mask: mx.array) -> mx.array:
    """Softmax over each row's real options; 0 on padding."""
    mask = marker_mask.astype(mx.bool_)
    masked = mx.where(mask, logits, mx.finfo(logits.dtype).min)
    return mx.softmax(masked, axis=-1) * mask.astype(logits.dtype)


def _uniform_base_thresholds(marker_mask: mx.array) -> mx.array:
    """Cut points whose sigmoid is the uniform survival curve `(K-1-k)/K`."""
    levels_per_row = mx.maximum(marker_mask.sum(axis=1, keepdims=True), 2.0)
    k_index = mx.arange(marker_mask.shape[1], dtype=marker_mask.dtype)[None]
    ratio = (levels_per_row - 1.0 - k_index) / levels_per_row
    eps = mx.finfo(marker_mask.dtype).eps
    ratio = mx.clip(ratio, eps, 1.0 - eps)
    return mx.log(ratio) - mx.log1p(-ratio)


class OrdinalHead(nn.Module):
    """Cumulative link over K ordered levels, with K read off the markers."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.cut = nn.Linear(hidden_size, 1)
        self.location = nn.Linear(hidden_size, 1)

    def __call__(self, marker_states: mx.array, marker_mask: mx.array) -> mx.array:
        """`(B, K, H)` float32 states, `(B, K)` float32 mask -> `(B, K)` probabilities."""
        n_levels = marker_states.shape[1]
        denominator = mx.maximum(marker_mask.sum(axis=1, keepdims=True), 1.0)
        pooled = (marker_states * marker_mask[..., None]).sum(axis=1) / denominator
        location = self.location(pooled).squeeze(-1)

        spacing = nn.softplus(self.cut(marker_states).squeeze(-1)) * marker_mask
        base = _uniform_base_thresholds(marker_mask)
        thresholds = base + location[:, None] - mx.cumsum(spacing, axis=1)
        survival = mx.sigmoid(mx.clip(thresholds, -LOGIT_CLAMP, LOGIT_CLAMP))

        k_index = mx.arange(n_levels)[None]
        levels_per_row = marker_mask.sum(axis=1, keepdims=True)
        survival = mx.where(k_index >= levels_per_row - 1, 0.0, survival)

        ones = mx.ones((survival.shape[0], 1), dtype=survival.dtype)
        lower = mx.concatenate([ones, survival[:, :-1]], axis=1)
        return mx.maximum((lower - survival) * marker_mask, 0.0)


class JointModel(nn.Module):
    """Backbone + marker scorer + ordinal head; `(N, M)` probabilities per batch."""

    def __init__(self, config: EncoderConfig) -> None:
        super().__init__()
        self.backbone = ModernBert(config)
        self.scorer = nn.Linear(config.hidden_size, 1, bias=False)
        self.ordinal = OrdinalHead(config.hidden_size)

    def __call__(self, input_ids: mx.array, attention_mask: mx.array,
                 marker_positions: mx.array, marker_mask: mx.array, ordered: mx.array,
                 marker_positions_back: mx.array | None = None) -> mx.array:
        hidden = self.backbone(input_ids, attention_mask)
        rows = mx.arange(hidden.shape[0])[:, None]
        states = hidden[rows, marker_positions]
        if marker_positions_back is not None:
            states = (states + hidden[rows, marker_positions_back]) / 2
        states = states.astype(mx.float32)
        mask = marker_mask.astype(mx.float32)

        logits = self.scorer(states).squeeze(-1)
        probs = masked_softmax(logits, mask)
        # The ordinal head is per row, so computing it for every row and keeping the
        # `score` rows gives those rows exactly what the torch model's index_copy does.
        return mx.where(ordered[:, None], self.ordinal(states, mask), probs)
