"""The assembled model (SOKUDAN_SPEC.md §6.2).

`model` does forward and nothing else (§2, single responsibility). Calibration lives
in `sokudan.calibration`, tokenisation in `sokudan.encoding`, losses in `train`.

The shape of a request is the whole point of the design:

    state    -> backbone once          (1, L_s, H)
    questions-> backbone as a batch    (N, L_q, H)
    head: N questions cross-attend into the one state
    markers  -> scorer                 (N, M)
    choice/bool -> softmax within the question
    score       -> dynamic-K cumulative link (§6.3)

The state is encoded **once per request** no matter how many questions come with it.
That is a structural difference from re-encoding the state per question -- but it is
a claim about speed, and §9 says not to make it in public until both sides have been
measured on the same machine. Nothing here asserts it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from sokudan.model.backbone import Backbone, BackboneSpec
from sokudan.model.head import DecisionHead, masked_softmax
from sokudan.model.ordinal import OrdinalHead, expected_level


@dataclass
class SokudanOutput:
    """Per-question distributions, plus the ordinal extras where they apply."""

    probs: Tensor
    """`(N, M)`; each row sums to 1 over its real options, 0 on padding."""
    marker_logits: Tensor
    """`(N, M)` raw scorer outputs. Meaningless for ordinal rows -- see `ordered`."""
    ordered: Tensor
    """`(N,)` bool; True where the row was scored by the ordinal head."""
    expected_level: Tensor | None
    """`(N,)` expected level for ordinal rows, NaN elsewhere. §6.3 wants both."""
    survival: Tensor | None
    """`(N, M)` `P(y > k)` for ordinal rows, NaN elsewhere. For debugging monotonicity."""


class SokudanModel(nn.Module):
    """Shared backbone + cross-attention decision head + two output parameterisations."""

    def __init__(
        self,
        backbone: Backbone,
        *,
        n_head_layers: int = 2,
        init_head_from_backbone: bool = True,
    ) -> None:
        super().__init__()
        spec = backbone.spec
        self.backbone = backbone
        self.spec = spec
        self.head = DecisionHead(
            spec.hidden_size,
            n_layers=n_head_layers,
            n_heads=spec.num_attention_heads,
            intermediate_size=spec.intermediate_size,
            norm_eps=spec.norm_eps,
        )
        self.ordinal = OrdinalHead(spec.hidden_size)
        if init_head_from_backbone:
            self.head.init_from_backbone(backbone)

    @classmethod
    def from_pretrained_backbone(
        cls,
        model_id: str | None = None,
        *,
        n_head_layers: int = 2,
        attn_implementation: str = "sdpa",
        dtype: torch.dtype = torch.float32,
    ) -> SokudanModel:
        from sokudan.config import BACKBONE_MODEL_ID

        backbone = Backbone.load(
            model_id or BACKBONE_MODEL_ID,
            attn_implementation=attn_implementation,
            dtype=dtype,
        )
        return cls(backbone, n_head_layers=n_head_layers)

    @property
    def hidden_size(self) -> int:
        return self.spec.hidden_size

    def encode_state(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """`(B_s, L_s)` -> `(B_s, L_s, H)`. Called once per request."""
        return self.backbone(input_ids, attention_mask)

    def encode_questions(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        """`(N, L_q)` -> `(N, L_q, H)`. Cacheable across requests (§5.2)."""
        return self.backbone(input_ids, attention_mask)

    def forward(
        self,
        state_input_ids: Tensor,
        state_attention_mask: Tensor,
        question_input_ids: Tensor,
        question_attention_mask: Tensor,
        marker_positions: Tensor,
        marker_mask: Tensor,
        ordered: Tensor,
    ) -> SokudanOutput:
        """
        Args:
            state_input_ids / state_attention_mask: `(1, L_s)` to broadcast one state
                across every question, or `(N, L_s)` for a per-question state.
            question_input_ids / question_attention_mask: `(N, L_q)`.
            marker_positions: `(N, M)` indices into `L_q`, padded rows repeating any
                valid index -- `marker_mask` is what decides which count.
            marker_mask: `(N, M)` 1 for a real option.
            ordered: `(N,)` bool, True for `score` questions.
        """
        state_hidden = self.encode_state(state_input_ids, state_attention_mask)
        question_hidden = self.encode_questions(question_input_ids, question_attention_mask)
        return self.decide(
            state_hidden,
            state_attention_mask,
            question_hidden,
            question_attention_mask,
            marker_positions,
            marker_mask,
            ordered,
        )

    def decide(
        self,
        state_hidden: Tensor,
        state_attention_mask: Tensor,
        question_hidden: Tensor,
        question_attention_mask: Tensor,
        marker_positions: Tensor,
        marker_mask: Tensor,
        ordered: Tensor,
    ) -> SokudanOutput:
        """Everything after the backbone. Split out so cached `H_q` can be reused."""
        n_questions, n_markers = marker_positions.shape
        if marker_mask.shape != marker_positions.shape:
            raise ValueError(
                f"marker_mask {tuple(marker_mask.shape)} does not match "
                f"marker_positions {tuple(marker_positions.shape)}"
            )
        if ordered.shape != (n_questions,):
            raise ValueError(f"ordered must be ({n_questions},), got {tuple(ordered.shape)}")

        refined = self.head(
            question_hidden,
            state_hidden,
            question_mask=question_attention_mask,
            state_mask=state_attention_mask,
        )
        marker_states = self.head.gather_markers(refined, marker_positions)  # (N, M, H)
        logits = self.head.scorer(marker_states).squeeze(-1)                 # (N, M)

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
            # `index_copy` rather than in-place assignment on a view, so the graph
            # stays intact for the softmax rows that are not being replaced.
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
        """For the model card. Reported, never estimated."""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "backbone": count(self.backbone),
            "head": count(self.head),
            "ordinal": count(self.ordinal),
            "total": count(self),
        }


def build_for_spec(spec: BackboneSpec, **kwargs: object) -> DecisionHead:
    """A head sized for a backbone, without loading the backbone's weights."""
    return DecisionHead(
        spec.hidden_size,
        n_heads=spec.num_attention_heads,
        intermediate_size=spec.intermediate_size,
        norm_eps=spec.norm_eps,
        **kwargs,  # type: ignore[arg-type]
    )
