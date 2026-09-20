"""Stage 1 loss (SOKUDAN_SPEC.md §8).

§8 specifies KL to the teacher's distribution, with RPS for `score`. §7.4 then removes
the teacher: every label today is gold, drawn from the generation condition. KL to a
one-hot target *is* cross-entropy, so that is what this computes -- the same objective
§8 asks for, under the labels §7.4 chose.

`score` keeps RPS. Cross-entropy charges the same price for missing by one level and
missing by four, which is exactly the property an ordinal scale must not have (§6.3),
and `docs/baseline_ja.md` §6.2 is what a model that ignores level distance looks like
in practice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from sokudan.model.ordinal import ranked_probability_score

EPS = 1e-9


@dataclass
class LossBreakdown:
    """Total plus per-primitive parts, so a regression can be localised."""

    total: Tensor
    choice_like: float
    ordinal: float
    n_choice_like: int
    n_ordinal: int

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.total.detach()),
            "loss_choice_like": self.choice_like,
            "loss_ordinal": self.ordinal,
            "n_choice_like": float(self.n_choice_like),
            "n_ordinal": float(self.n_ordinal),
        }


def stage1_loss(
    probs: Tensor,
    labels: Tensor,
    ordered: Tensor,
    marker_mask: Tensor,
    *,
    ordinal_weight: float = 1.0,
) -> LossBreakdown:
    """Cross-entropy for `choice`/`bool`, RPS for `score`.

    Args:
        probs: `(B, M)` from the model, already normalised over real options.
        labels: `(B,)` gold option index.
        ordered: `(B,)` bool, True for `score` rows.
        marker_mask: `(B, M)`.
        ordinal_weight: relative weight of the RPS term. 1.0 today; exposed because
            the two terms are on different scales and the balance is worth ablating.

    The mean is taken over each subset and then combined by count, so the result does
    not shift when a batch happens to hold more of one primitive than another.
    """
    if probs.shape != marker_mask.shape:
        raise ValueError(f"probs {tuple(probs.shape)} vs mask {tuple(marker_mask.shape)}")

    ordered = ordered.to(torch.bool)
    device = probs.device
    zero = torch.zeros((), device=device, dtype=probs.dtype)

    choice_rows = (~ordered).nonzero(as_tuple=True)[0]
    ordinal_rows = ordered.nonzero(as_tuple=True)[0]

    choice_loss = zero
    if choice_rows.numel():
        picked = probs[choice_rows].gather(1, labels[choice_rows].unsqueeze(1)).squeeze(1)
        choice_loss = -torch.log(picked.clamp_min(EPS)).mean()

    ordinal_loss = zero
    if ordinal_rows.numel():
        ordinal_loss = ranked_probability_score(
            probs[ordinal_rows],
            labels[ordinal_rows],
            marker_mask[ordinal_rows].to(probs.dtype),
        )

    n_choice = int(choice_rows.numel())
    n_ordinal = int(ordinal_rows.numel())
    total_count = max(n_choice + n_ordinal, 1)
    total = (
        choice_loss * (n_choice / total_count)
        + ordinal_weight * ordinal_loss * (n_ordinal / total_count)
    )

    return LossBreakdown(
        total=total,
        choice_like=float(choice_loss.detach()),
        ordinal=float(ordinal_loss.detach()),
        n_choice_like=n_choice,
        n_ordinal=n_ordinal,
    )
