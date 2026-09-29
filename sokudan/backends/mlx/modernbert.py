# ruff: noqa: E501
"""ModernBERT encoder in MLX (inference only). Vendored, unmodified below this docstring.

Source: laya-mlx 0.2.0 (PyPI wheel `laya_mlx-0.2.0`, https://github.com/mizorewww/laya-mlx),
`laya_mlx/model.py` lines 13-154 (`EncoderConfig` through `ModernBert`); the file's sha256
in that wheel is 24ef50d4e53e0b12528cffd46cc463c0b8fbea8f052e80f8bf40a68aa692c7ea.
Copyright 2026 laya-mlx contributors. Licensed under the Apache License, Version 2.0
(see NOTICE). laya-mlx follows the ModernBERT architecture in Hugging Face Transformers.

Only the encoder is taken. Laya's decision head, prompt construction and agent are not;
sokudan's own heads are in `sokudan.backends.mlx.model`.
"""

from dataclasses import dataclass, fields

import mlx.core as mx
import mlx.nn as nn


@dataclass
class EncoderConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    model_type: str = "modernbert"
    norm_eps: float = 1e-5
    norm_bias: bool = False
    attention_bias: bool = False
    mlp_bias: bool = False
    hidden_activation: str = "gelu"
    local_attention: int = 128
    global_attn_every_n_layers: int = 3
    global_rope_theta: float = 160000.0
    local_rope_theta: float = 10000.0
    max_position_embeddings: int = 8192
    layer_types: list[str] | None = None
    rope_parameters: dict | None = None

    @classmethod
    def from_dict(cls, value: dict):
        names = {f.name for f in fields(cls)}
        cfg = cls(**{k: v for k, v in value.items() if k in names})
        if cfg.model_type != "modernbert":
            raise ValueError(f"Unsupported encoder: {cfg.model_type!r}; expected modernbert")
        if cfg.hidden_activation != "gelu":
            raise ValueError(f"Unsupported encoder activation: {cfg.hidden_activation!r}")
        if cfg.hidden_size % cfg.num_attention_heads or cfg.head_dim % 2:
            raise ValueError("ModernBERT requires an even, integral attention head dimension")
        if cfg.layer_types is None:
            cfg.layer_types = [
                "full_attention" if i % cfg.global_attn_every_n_layers == 0 else "sliding_attention"
                for i in range(cfg.num_hidden_layers)
            ]
        if len(cfg.layer_types) != cfg.num_hidden_layers or set(cfg.layer_types) - {
            "full_attention",
            "sliding_attention",
        }:
            raise ValueError("Invalid ModernBERT layer_types")
        for kind in set(cfg.layer_types):
            params = (cfg.rope_parameters or {}).get(kind, {})
            if params.get("rope_type", "default") != "default":
                raise ValueError("Only default (unscaled) ModernBERT RoPE is supported")
        return cfg

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def rope_base(self, kind):
        fallback = self.global_rope_theta if kind == "full_attention" else self.local_rope_theta
        return float((self.rope_parameters or {}).get(kind, {}).get("rope_theta", fallback))


class Embeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def __call__(self, ids):
        return self.norm(self.tok_embeddings(ids))


class EncoderAttention(nn.Module):
    def __init__(self, cfg, kind):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.head_dim = cfg.head_dim
        self.base = cfg.rope_base(kind)
        self.Wqkv = nn.Linear(cfg.hidden_size, 3 * cfg.hidden_size, bias=cfg.attention_bias)
        self.Wo = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=cfg.attention_bias)

    def __call__(self, x, mask):
        b, length, _ = x.shape
        qkv = self.Wqkv(x).reshape(b, length, 3, self.num_heads, self.head_dim)
        q, k, v = [qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3)]
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.base, scale=1.0, offset=0)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
        return self.Wo(out.transpose(0, 2, 1, 3).reshape(b, length, -1))


class EncoderMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.Wi = nn.Linear(cfg.hidden_size, 2 * cfg.intermediate_size, bias=cfg.mlp_bias)
        self.Wo = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=cfg.mlp_bias)

    def __call__(self, x):
        value, gate = mx.split(self.Wi(x), 2, axis=-1)
        return self.Wo(nn.gelu(value) * gate)


class EncoderLayer(nn.Module):
    def __init__(self, cfg, index):
        super().__init__()
        self.attention_type = cfg.layer_types[index]
        self.attn_norm = (
            nn.Identity()
            if index == 0
            else nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        )
        self.attn = EncoderAttention(cfg, self.attention_type)
        self.mlp_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)
        self.mlp = EncoderMLP(cfg)

    def __call__(self, x, mask):
        x = x + self.attn(self.attn_norm(x), mask)
        return x + self.mlp(self.mlp_norm(x))


def attention_masks(attention_mask, window):
    """Boolean key masks, with inclusive local distance <= local_attention // 2.

    Padded queries can see valid keys to avoid all-masked softmax rows. They are
    never used as keys or pooled outputs, so valid-token results are unchanged.
    """
    valid = attention_mask.astype(mx.bool_)
    full = valid[:, None, None, :]
    positions = mx.arange(valid.shape[1])
    local = mx.abs(positions[:, None] - positions[None, :]) <= window // 2
    local = (local[None, None] | ~valid[:, None, :, None]) & full
    return {"full_attention": full, "sliding_attention": local}


class ModernBert(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embeddings = Embeddings(cfg)
        self.layers = [EncoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        self.final_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.norm_eps, bias=cfg.norm_bias)

    def __call__(self, input_ids, attention_mask):
        x = self.embeddings(input_ids)
        masks = attention_masks(attention_mask, self.config.local_attention)
        for layer in self.layers:
            x = layer(x, masks[layer.attention_type])
        return self.final_norm(x)
