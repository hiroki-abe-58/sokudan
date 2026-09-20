"""Cross-attention decision head (SOKUDAN_SPEC.md §6.2).

    question --[backbone]--> H_q      (B, L_q, H)   N questions in the batch dim
    state    --[backbone]--> H_state  (B, L_s, H)   once per request, broadcast
                                  |
                        [head: self-attn over H_q, cross-attn into H_state] x n_layers
                                  |
                        read the marker positions -> scorer

v1 packed every question behind the state in one sequence with a block-diagonal mask.
§6.1 discards that: `modernbert-ja-310m` has `local_attention: 128` and
`global_attn_every_n_layers: 3`, so a question block sitting at position 3000 cannot
see the state at all in two layers out of three, and its behaviour changes with where
it lands in the sequence. Splitting the two sides keeps each sequence short, so the
local/global alternation runs under the conditions it was pretrained with, and makes
order invariance structural: questions never see each other.

The three initialisation choices in §6.2, and why each one earns its place:

* **Self-attention and MLP weights are copied from the backbone's last layers.**
  The markers are `<mask>` tokens; random attention on top of them throws away the
  MLM prior sitting on exactly the positions we are about to read.
* **The cross-attention output projection is zero.** At step zero the head is the
  identity on the question representation and the state contributes nothing, so
  training starts from a working model and mixes the state in as gradients arrive.
  Same idea as an adapter or ControlNet.
* **The scorer starts near zero**, so the first softmax is near uniform. A confident
  wrong start makes a proper scoring rule's gradient violent.

**No rotary embedding inside the head.** The backbone has already applied RoPE, so
`H_q` is position-aware before the head sees it; the head refines those
representations rather than re-encoding a raw sequence. This does mean the copied
attention weights run without the positional signal they were trained beside, so
they are an initialisation rather than an exact continuation -- which is all §6.2
claims for them. Recorded in `docs/architecture.md` as an ablation candidate.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

NEG_INF = torch.finfo(torch.float32).min


def _pad_mask_to_bias(mask: Tensor | None, dtype: torch.dtype) -> Tensor | None:
    """`(B, L)` 1/0 padding mask -> `(B, 1, 1, L)` additive attention bias."""
    if mask is None:
        return None
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias = bias.masked_fill(mask == 0, torch.finfo(dtype).min)
    return bias[:, None, None, :]


class _SelfAttention(nn.Module):
    """Fused-QKV self-attention shaped exactly like `ModernBertAttention`.

    Same module names and shapes (`Wqkv`, `Wo`, no bias) so `load_from_backbone_layer`
    is a real `copy_`, not an approximate remap that silently transposes something.
    """

    def __init__(self, hidden_size: int, n_heads: int) -> None:
        super().__init__()
        if hidden_size % n_heads:
            raise ValueError(f"hidden_size {hidden_size} is not divisible by {n_heads} heads")
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.Wqkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: Tensor, pad_bias: Tensor | None) -> Tensor:
        batch, length, _ = x.shape
        qkv = self.Wqkv(x).view(batch, length, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        context = F.scaled_dot_product_attention(query, key, value, attn_mask=pad_bias)
        return self.Wo(context.transpose(1, 2).reshape(batch, length, self.hidden_size))


class _CrossAttention(nn.Module):
    """Questions attend into the state. Output projection starts at zero."""

    def __init__(self, hidden_size: int, n_heads: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.head_dim = hidden_size // n_heads
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wkv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.Wo = nn.Linear(hidden_size, hidden_size, bias=False)

        nn.init.normal_(self.Wq.weight, std=0.02)
        nn.init.normal_(self.Wkv.weight, std=0.02)
        nn.init.zeros_(self.Wo.weight)  # identity at step zero -- see module docstring

    def forward(self, x: Tensor, state: Tensor, state_bias: Tensor | None) -> Tensor:
        batch, length, _ = x.shape
        source_len = state.shape[1]

        query = self.Wq(x).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        kv = self.Wkv(state).view(batch, source_len, 2, self.n_heads, self.head_dim)
        key, value = kv.permute(2, 0, 3, 1, 4).unbind(0)

        context = F.scaled_dot_product_attention(query, key, value, attn_mask=state_bias)
        return self.Wo(context.transpose(1, 2).reshape(batch, length, self.hidden_size))


class _GeGLUMLP(nn.Module):
    """`ModernBertMLP`: one fused projection split into value and gate."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.Wi = nn.Linear(hidden_size, 2 * intermediate_size, bias=False)
        self.act = nn.GELU()
        self.Wo = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        value, gate = self.Wi(x).chunk(2, dim=-1)
        return self.Wo(self.act(value) * gate)


class DecisionLayer(nn.Module):
    """Pre-norm self-attention, then cross-attention into the state, then MLP."""

    def __init__(
        self,
        hidden_size: int,
        n_heads: int,
        intermediate_size: int,
        *,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.attn = _SelfAttention(hidden_size, n_heads)
        self.cross_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.cross_attn = _CrossAttention(hidden_size, n_heads)
        self.mlp_norm = nn.LayerNorm(hidden_size, eps=norm_eps)
        self.mlp = _GeGLUMLP(hidden_size, intermediate_size)

    def forward(
        self,
        x: Tensor,
        pad_bias: Tensor | None,
        state: Tensor,
        state_bias: Tensor | None,
    ) -> Tensor:
        x = x + self.attn(self.attn_norm(x), pad_bias)
        x = x + self.cross_attn(self.cross_norm(x), state, state_bias)
        return x + self.mlp(self.mlp_norm(x))

    @torch.no_grad()
    def load_from_backbone_layer(self, layer: nn.Module) -> None:
        """Copy self-attention, MLP and their norms from a `ModernBertEncoderLayer`.

        Cross-attention is left alone: it has no counterpart in the backbone, and its
        output projection must stay zero so the head starts as the identity.

        `attn_norm` on the backbone's layer 0 is an `Identity`, so a caller taking the
        *first* layers would silently copy nothing; the default takes the last ones.
        """
        self.attn.Wqkv.weight.copy_(layer.attn.Wqkv.weight)
        self.attn.Wo.weight.copy_(layer.attn.Wo.weight)
        self.mlp.Wi.weight.copy_(layer.mlp.Wi.weight)
        self.mlp.Wo.weight.copy_(layer.mlp.Wo.weight)

        for target, source in ((self.attn_norm, layer.attn_norm), (self.mlp_norm, layer.mlp_norm)):
            if isinstance(source, nn.LayerNorm):
                target.weight.copy_(source.weight)
                if source.bias is not None and target.bias is not None:
                    target.bias.copy_(source.bias)
            else:
                # Identity: the backbone applies no norm here, so neither do we.
                nn.init.ones_(target.weight)
                nn.init.zeros_(target.bias)


class DecisionHead(nn.Module):
    """`n_layers` decision layers plus the marker scorer.

    §14.3 fixes `n_layers` at 2 for the sprint; the 2-vs-4 ablation is deferred.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        n_layers: int = 2,
        n_heads: int = 12,
        intermediate_size: int = 3072,
        norm_eps: float = 1e-5,
        scorer_init_std: float = 0.002,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("the head needs at least one layer")
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList(
            DecisionLayer(hidden_size, n_heads, intermediate_size, norm_eps=norm_eps)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=norm_eps)

        # Small, not zero: an exactly zero scorer gives every option an identical
        # logit *and* an identical gradient, so the options never differentiate.
        self.scorer = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.scorer.weight, std=scorer_init_std)

    def forward(
        self,
        question_states: Tensor,
        state_states: Tensor,
        *,
        question_mask: Tensor | None = None,
        state_mask: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            question_states: `(B, L_q, H)` from the backbone.
            state_states: `(B, L_s, H)`. A single state is broadcast across the batch
                by passing `(1, L_s, H)`.
            question_mask: `(B, L_q)` 1/0.
            state_mask: `(B, L_s)` or `(1, L_s)` 1/0.

        Returns:
            `(B, L_q, H)` refined question representations.
        """
        if question_states.dim() != 3 or state_states.dim() != 3:
            raise ValueError("question_states and state_states must both be (B, L, H)")

        batch = question_states.shape[0]
        if state_states.shape[0] == 1 and batch > 1:
            state_states = state_states.expand(batch, -1, -1)
            if state_mask is not None and state_mask.shape[0] == 1:
                state_mask = state_mask.expand(batch, -1)
        elif state_states.shape[0] != batch:
            raise ValueError(
                f"state batch {state_states.shape[0]} matches neither the question "
                f"batch {batch} nor 1 (broadcast)"
            )

        dtype = question_states.dtype
        pad_bias = _pad_mask_to_bias(question_mask, dtype)
        state_bias = _pad_mask_to_bias(state_mask, dtype)

        hidden = question_states
        for layer in self.layers:
            hidden = layer(hidden, pad_bias, state_states, state_bias)
        return self.final_norm(hidden)

    def score_markers(self, hidden: Tensor, marker_positions: Tensor) -> Tensor:
        """Gather marker representations and score them.

        Args:
            hidden: `(B, L_q, H)` from `forward`.
            marker_positions: `(B, M)` indices into `L_q`.

        Returns:
            `(B, M)` logits, one per option.
        """
        if marker_positions.dim() != 2:
            raise ValueError(f"expected (B, M) marker positions, got {marker_positions.shape}")
        gathered = self.gather_markers(hidden, marker_positions)
        return self.scorer(gathered).squeeze(-1)

    @staticmethod
    def gather_markers(hidden: Tensor, marker_positions: Tensor) -> Tensor:
        """`(B, L, H)` + `(B, M)` -> `(B, M, H)`.

        Bounds are checked here rather than left to `gather`, because §5.2 warns that
        a marker position drifting away from where the encoder put it breaks
        everything silently. A raw gather reports "index 24 is out of bounds for
        dimension 1" with no hint that a marker is what went missing.
        """
        length = hidden.shape[1]
        if marker_positions.numel() and int(marker_positions.max()) >= length:
            raise ValueError(
                f"marker position {int(marker_positions.max())} is outside a question of "
                f"length {length}. Marker positions must come from `encode_question`, "
                f"and padding must be applied to both the sequence and the positions."
            )
        if marker_positions.numel() and int(marker_positions.min()) < 0:
            raise ValueError("marker positions must be non-negative")
        index = marker_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        return hidden.gather(dim=1, index=index)

    @torch.no_grad()
    def init_from_backbone(self, backbone: nn.Module) -> None:
        """Copy the backbone's **last** `n_layers` into the head, in order (§6.2)."""
        layers = getattr(backbone, "layers", None)
        if layers is None:
            raise ValueError("backbone exposes no `.layers` to copy from")
        if len(layers) < len(self.layers):
            raise ValueError(
                f"backbone has {len(layers)} layers, fewer than the head's {len(self.layers)}"
            )
        for head_layer, source in zip(self.layers, layers[-len(self.layers):], strict=True):
            head_layer.load_from_backbone_layer(source)


def masked_softmax(logits: Tensor, marker_mask: Tensor | None = None) -> Tensor:
    """Softmax within a question, over its real options only.

    `choice` and `bool` share this: a `bool` is a two-option question internally, so
    there is no second implementation of the binary case (§2).
    """
    if marker_mask is None:
        return logits.softmax(dim=-1)
    marker_mask = marker_mask.to(torch.bool)
    masked = logits.masked_fill(~marker_mask, torch.finfo(logits.dtype).min)
    probs = masked.softmax(dim=-1)
    return probs * marker_mask.to(probs.dtype)
