"""Joint encoding, the Laya arrangement (s2c).

    [CLS] instructions [SEP] options+markers [SEP] state [SEP]  ->  backbone  ->
    marker positions  ->  scorer  ->  softmax / ordinal

No cross-attention head. The backbone's own attention does the mixing, and the
marker hidden states go straight to the scorer.

**Why this exists.** §6.1 rejected joint encoding on a stated mechanism:
`modernbert-ja-310m` runs `local_attention: 128` with global attention every third
layer, so a marker and a distant state token cannot see one another in two layers out
of three, and behaviour shifts with where in the sequence things land. s2a then
failed for an unrelated reason -- dilution, with every validation metric below s1's --
so the mechanism deserves a measurement instead of an argument. Running both arms on
the same corpus, the same mixture and the same schedule is the only way to say which
of the two explanations the numbers support.

**What it costs.** The state is re-encoded once per question rather than once per
request, so N questions cost N backbone passes over `len(state) + len(question)`
tokens instead of one pass over the state plus N short passes. That is the §6.2
trade-off, stated rather than hidden: if joint wins on accuracy, it loses the single
structural latency claim the project makes.

The ordinal head is the same `OrdinalHead` the separate arm uses, with the same
dynamic K. Only the path from tokens to marker states differs, which is what makes
the comparison mean something.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sokudan.model.backbone import Backbone
from sokudan.model.head import DecisionHead, masked_softmax
from sokudan.model.ordinal import OrdinalHead, expected_level
from sokudan.model.sokudan import SokudanOutput


@dataclass(frozen=True)
class JointBatch:
    """What the joint arm needs. One sequence per (question, state) pair."""

    input_ids: Tensor
    attention_mask: Tensor
    marker_positions: Tensor
    marker_mask: Tensor
    ordered: Tensor


class SokudanJointModel(nn.Module):
    """Backbone + marker scorer + the shared ordinal head. No decision head."""

    def __init__(self, backbone: Backbone, *, scorer_init_std: float = 0.002) -> None:
        super().__init__()
        spec = backbone.spec
        self.backbone = backbone
        self.spec = spec
        # Same shape and same initialisation scale as `DecisionHead.scorer`. Small
        # rather than zero for the same reason: an exactly zero scorer gives every
        # option an identical logit *and* an identical gradient.
        self.scorer = nn.Linear(spec.hidden_size, 1, bias=False)
        nn.init.normal_(self.scorer.weight, std=scorer_init_std)
        self.ordinal = OrdinalHead(spec.hidden_size)

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_id: str | None = None,
        *,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.float32,
    ) -> SokudanJointModel:
        from sokudan.config import BACKBONE_MODEL_ID

        backbone = Backbone.load(
            model_id or BACKBONE_MODEL_ID,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        return cls(backbone)

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        marker_positions: Tensor,
        marker_mask: Tensor,
        ordered: Tensor,
    ) -> SokudanOutput:
        """
        Args:
            input_ids / attention_mask: `(N, L)` joint sequences.
            marker_positions: `(N, M)` indices into `L`; padded rows repeat a valid
                index and `marker_mask` decides which count.
            marker_mask: `(N, M)` 1 for a real option.
            ordered: `(N,)` bool, True for `score` rows.

        Returns the same `SokudanOutput` the separate arm returns, so every metric,
        loss and diagnostic downstream is shared between the two.
        """
        n_questions, n_markers = marker_positions.shape
        if marker_mask.shape != marker_positions.shape:
            raise ValueError(
                f"marker_mask {tuple(marker_mask.shape)} does not match "
                f"marker_positions {tuple(marker_positions.shape)}"
            )
        if ordered.shape != (n_questions,):
            raise ValueError(f"ordered must be ({n_questions},), got {tuple(ordered.shape)}")

        hidden = self.backbone(input_ids, attention_mask)
        # Reused from the separate arm: the bounds check there explains that a marker
        # drifting from where the encoder put it fails silently otherwise.
        marker_states = DecisionHead.gather_markers(hidden, marker_positions)
        logits = self.scorer(marker_states).squeeze(-1)

        ordered = ordered.to(torch.bool)
        mask_float = marker_mask.to(marker_states.dtype)
        probs = masked_softmax(logits, marker_mask)

        expectation: Tensor | None = None
        survival: Tensor | None = None

        if bool(ordered.any()):
            if n_markers < 2:
                raise ValueError("an ordinal question needs at least 2 option slots")
            rows = ordered.nonzero(as_tuple=True)[0]
            ordinal_probs, ordinal_survival = self.ordinal(
                marker_states[rows], mask_float[rows]
            )
            probs = probs.index_copy(0, rows, ordinal_probs)
            expectation = torch.full(
                (n_questions,), float("nan"), dtype=probs.dtype, device=probs.device
            ).index_copy(0, rows, expected_level(ordinal_probs))
            survival = torch.full_like(probs, float("nan")).index_copy(
                0, rows, ordinal_survival
            )

        return SokudanOutput(
            probs=probs,
            marker_logits=logits,
            ordered=ordered,
            expected_level=expectation,
            survival=survival,
        )

    def parameter_counts(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "backbone": count(self.backbone),
            "scorer": count(self.scorer),
            "ordinal": count(self.ordinal),
            "total": count(self),
        }
