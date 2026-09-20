"""Dynamic-K ordinal head: cumulative link with a monotone CDF (SOKUDAN_SPEC.md §6.3).

A `score` question is not a multiclass question that happens to be sorted, and K is
decided per request rather than per model. Standard CORAL fixes K at construction
time, which is incompatible with a schema that arrives at inference time, so the
parameterisation reads its cut points off the markers instead:

    b_k = softplus(w · h_k)            >= 0, one per level marker
    a   = v · h_pool                   one shared location term
    P(y > k) = sigmoid(a - Σ_{j<=k} b_j)
    p_k = P(y > k-1) - P(y > k)

The cumulative sum is non-decreasing because every `b_k` is non-negative, so the
survival function is non-increasing and the CDF is monotone **by construction** --
not by penalty, not by a sorting trick afterwards. The same weights serve K=2 and
K=7, and adding a level does not disturb the existing ones.

Why this matters here specifically: `docs/baseline_ja.md` §6.2 measured
`laya-multilingual` selecting the first presented option 0-1 times out of 300 across
five schema variants. A head whose level probabilities come from a monotone CDF over
per-level cut points cannot develop a dead slot the way an unconstrained softmax over
marker logits can, because every level's mass is a difference of two adjacent points
on one shared curve.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

# Keeps sigmoid out of its saturated tails, where gradients vanish and float32 rounds
# neighbouring cut points onto each other.
LOGIT_CLAMP = 15.0


def _uniform_base_thresholds(marker_mask: Tensor, *, dtype: torch.dtype) -> Tensor:
    """Thresholds whose sigmoid is the uniform survival curve `P(y > k) = (K-1-k)/K`.

    The reason this exists is that K arrives with the request. An exactly uniform
    start needs cut points at `logit((K-1-k)/K)`, which depend on K, so a head with
    K-independent parameters cannot produce one from its weights alone -- with a
    constant learned spacing the initial distribution comes out at roughly 0.67 on
    the first level for K=3, and worse as K grows.

    Splitting the thresholds into a closed-form uniform base plus a learned monotone
    correction fixes that for every K at once: at initialisation the correction is
    ~0 and the head emits exactly `1/K` per level whatever K is, which is what §6.2
    asks for so that a proper scoring rule's gradient starts tame. It is the same
    idea as zero-initialising the cross-attention output projection -- begin at the
    neutral function, let gradients move away from it.

    The base is non-increasing in k, so adding it preserves the monotone CDF.
    """
    levels_per_row = marker_mask.sum(dim=1, keepdim=True).clamp(min=2.0)  # (B, 1)
    k_index = torch.arange(
        marker_mask.shape[1], device=marker_mask.device, dtype=dtype
    ).unsqueeze(0)

    ratio = (levels_per_row - 1.0 - k_index) / levels_per_row
    # Only k <= K-2 is ever read: k = K-1 and padding are overwritten with 0 survival.
    eps = torch.finfo(dtype).eps
    ratio = ratio.clamp(min=eps, max=1.0 - eps)
    return torch.log(ratio) - torch.log1p(-ratio)


class OrdinalHead(nn.Module):
    """Marker representations -> a monotone distribution over K ordered levels.

    Args:
        hidden_size: width of the marker representations.
        init_scale: std of the initial projections. 0.0 (the default) removes any
            random tilt from the starting distribution.
        cut_bias: initial bias of the spacing projection. Sets how spread the
            initial CDF is, and -- because softplus' is small in its left tail --
            how much gradient this parameter receives at all. See the comment in
            `__init__`; -5.0 is exactly uniform and untrainable, -1.0 is
            near-uniform and trains.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        init_scale: float = 0.0,
        cut_bias: float = -1.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cut = nn.Linear(hidden_size, 1)       # w: per-level spacing
        self.location = nn.Linear(hidden_size, 1)  # v: shared location

        # Both projections start at exactly zero so the head emits exactly 1/K for
        # every K on step one (see `_uniform_base_thresholds`). A `Linear(H, 1)` has
        # no symmetry to break -- its weight gradient is `dL/dout * h`, which differs
        # per input feature -- so zero init costs nothing and buys an exact,
        # reproducible starting point rather than a random tilt of roughly 0.55 in
        # the location term. `init_scale` remains for ablating that choice.
        if init_scale == 0.0:
            nn.init.zeros_(self.cut.weight)
            nn.init.zeros_(self.location.weight)
        else:
            nn.init.normal_(self.cut.weight, std=init_scale)
            nn.init.normal_(self.location.weight, std=init_scale)
        nn.init.zeros_(self.location.bias)
        # `cut_bias` trades exactness of the uniform start against trainability, and
        # the trade is real rather than theoretical. -5.0 makes softplus(-5) = 0.0067
        # and gives an *exactly* uniform start -- but softplus' at -5 is also 0.0067,
        # so the gradient reaching this parameter is attenuated ~150x. An AdamW step
        # moves a parameter by roughly the learning rate, so over a few hundred steps
        # this one does not move at all.
        #
        # A frozen spacing is worse than slow learning. With every b_k ~ 0 the
        # thresholds collapse to `base_k + a`, a one-parameter family that slides mass
        # between the bottom and top levels but **cannot peak on a middle level** --
        # for a 3-level scale it cannot say "confidently level 1".
        #
        # -1.0 gives softplus(-1) = 0.313 and a gradient scale of 0.269, roughly 40x
        # better, at the cost of a start that is near-uniform rather than exactly
        # uniform. §6.2 asks for near-uniform so the proper scoring rule's gradient
        # starts tame, and that intent survives; exactness does not.
        nn.init.constant_(self.cut.bias, cut_bias)

    def forward(
        self,
        marker_states: Tensor,
        marker_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Args:
            marker_states: `(B, K, H)` -- one representation per level marker, lowest
                level first.
            marker_mask: `(B, K)` of 1 for a real level and 0 for padding, so a batch
                may mix questions with different K. `None` means every level is real.

        Returns:
            `(probs, survival)` where `probs` is `(B, K)` summing to 1 over real
            levels and `survival[:, k]` is `P(y > k)`.
        """
        if marker_states.dim() != 3:
            raise ValueError(f"expected (B, K, H) marker states, got {marker_states.shape}")
        batch, n_levels, hidden = marker_states.shape
        if hidden != self.hidden_size:
            raise ValueError(f"expected hidden size {self.hidden_size}, got {hidden}")
        if n_levels < 2:
            raise ValueError("an ordinal question needs at least 2 levels")

        if marker_mask is None:
            marker_mask = marker_states.new_ones(batch, n_levels)
        marker_mask = marker_mask.to(marker_states.dtype)
        if marker_mask.shape != (batch, n_levels):
            raise ValueError(f"marker_mask must be (B, K), got {marker_mask.shape}")

        # Location term: mean over the real markers only.
        denominator = marker_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (marker_states * marker_mask.unsqueeze(-1)).sum(dim=1) / denominator
        location = self.location(pooled).squeeze(-1)  # (B,)

        # Non-negative spacing per level. Padding contributes zero width so it cannot
        # push real levels apart.
        spacing = F.softplus(self.cut(marker_states).squeeze(-1)) * marker_mask  # (B, K)

        # thresholds[:, k] = base_k + a - sum_{j<=k} b_j.
        # Both `base` and `-cumsum(b)` are non-increasing in k, so their sum is too:
        # monotonicity survives the added base.
        base = _uniform_base_thresholds(marker_mask, dtype=marker_states.dtype)
        thresholds = base + location.unsqueeze(1) - torch.cumsum(spacing, dim=1)
        survival = torch.sigmoid(thresholds.clamp(-LOGIT_CLAMP, LOGIT_CLAMP))  # P(y > k)

        # P(y > K-1) is 0 by definition: there is no level above the top one. Learning
        # it would leave `sum_k p_k = 1 - P(y > K-1)` short of 1 and force a rescale
        # that breaks the telescoping. Padded levels sit above the top one, so 0 too.
        k_index = torch.arange(n_levels, device=marker_states.device).unsqueeze(0)
        levels_per_row = marker_mask.sum(dim=1, keepdim=True)
        survival = torch.where(k_index >= levels_per_row - 1, survival.new_zeros(()), survival)

        # p_0 = 1 - P(y > 0); p_k = P(y > k-1) - P(y > k). Telescopes to exactly 1
        # over the real levels, so no renormalisation is needed.
        ones = survival.new_ones(batch, 1)
        lower = torch.cat([ones, survival[:, :-1]], dim=1)
        probs = ((lower - survival) * marker_mask).clamp(min=0.0)
        return probs, survival


def expected_level(probs: Tensor) -> Tensor:
    """`sum_k k * p_k`. §6.3 requires returning this *and* the distribution."""
    levels = torch.arange(probs.shape[-1], device=probs.device, dtype=probs.dtype)
    return (probs * levels).sum(dim=-1)


def ranked_probability_score(
    probs: Tensor,
    targets: Tensor,
    marker_mask: Tensor | None = None,
) -> Tensor:
    """RPS, the training loss for `score` (§6.3, §8 Stage 1).

    Cross-entropy penalises "off by one" and "off by four" identically, which is
    wrong for an ordinal scale. RPS compares cumulative distributions, so the penalty
    grows with the distance of the miss.

    Normalised by `K - 1` so batches mixing different K are comparable.

    Args:
        probs: `(B, K)` summing to 1 over real levels.
        targets: `(B,)` integer gold level.
        marker_mask: `(B, K)`; levels past each row's real K are excluded.
    """
    if probs.dim() != 2:
        raise ValueError(f"expected (B, K) probs, got {probs.shape}")
    batch, n_levels = probs.shape
    if targets.shape != (batch,):
        raise ValueError(f"expected ({batch},) targets, got {targets.shape}")

    if marker_mask is None:
        marker_mask = probs.new_ones(batch, n_levels)
    marker_mask = marker_mask.to(probs.dtype)

    k_per_row = marker_mask.sum(dim=1)
    if (k_per_row < 2).any():
        raise ValueError("every row needs at least 2 real levels")
    if (targets < 0).any() or (targets >= k_per_row.long()).any():
        raise ValueError("a target falls outside its row's real levels")

    onehot = F.one_hot(targets, num_classes=n_levels).to(probs.dtype)
    predicted_cdf = torch.cumsum(probs * marker_mask, dim=1)
    target_cdf = torch.cumsum(onehot, dim=1)

    squared = ((predicted_cdf - target_cdf) ** 2) * marker_mask
    return (squared.sum(dim=1) / (k_per_row - 1.0)).mean()
