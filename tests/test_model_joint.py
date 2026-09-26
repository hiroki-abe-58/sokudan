"""Joint-arm model tests.

`sokudan/model/joint.py` is the arm v0.1 actually ships (`docs/ablations.md` §3c
measured it beating the separate arm 0.506 -> 0.872 on held-out), and it had no
tests: `scripts/coverage_report.py` reported it at 0%. Everything it shares with the
separate arm is covered by `tests/test_model.py`; what is tested here is the part
that is only in this file -- the marker scorer, the ordinal routing by row, and the
`use_ordinal=False` ablation switch.

The same tiny randomly-initialised ModernBERT as `tests/test_model.py` is used. The
properties are structural and do not depend on width.
"""

from __future__ import annotations

import pytest
import torch

from sokudan.model.backbone import Backbone, BackboneSpec
from sokudan.model.joint import JointBatch, SokudanJointModel

TINY = BackboneSpec(
    hidden_size=32,
    num_attention_heads=4,
    num_hidden_layers=4,
    intermediate_size=64,
    norm_eps=1e-5,
    local_attention=8,
    global_attn_every_n_layers=2,
    max_position_embeddings=512,
)


def _tiny_backbone() -> Backbone:
    from transformers import AutoModel, ModernBertConfig

    torch.manual_seed(0)
    config = ModernBertConfig(
        vocab_size=256,
        hidden_size=TINY.hidden_size,
        num_attention_heads=TINY.num_attention_heads,
        num_hidden_layers=TINY.num_hidden_layers,
        intermediate_size=TINY.intermediate_size,
        local_attention=TINY.local_attention,
        global_attn_every_n_layers=TINY.global_attn_every_n_layers,
        max_position_embeddings=TINY.max_position_embeddings,
        pad_token_id=0, bos_token_id=1, eos_token_id=2,
        cls_token_id=3, sep_token_id=4,
    )
    return Backbone(AutoModel.from_config(config), BackboneSpec.from_config(config))


@pytest.fixture
def joint_model() -> SokudanJointModel:
    return SokudanJointModel(_tiny_backbone())


def make_joint_batch(specs: list[tuple[int, bool]], *, seq_len: int = 48) -> JointBatch:
    """Build a `JointBatch` from `[(n_options, ordered), ...]`.

    Padded marker slots repeat a valid index rather than pointing past the sequence,
    which is what the collator does -- `marker_mask`, not the index, is what makes a
    slot count.
    """
    n = len(specs)
    max_options = max(k for k, _ in specs)
    assert 2 * max_options + 2 <= seq_len
    input_ids = torch.randint(5, 200, (n, seq_len))
    attention_mask = torch.ones(n, seq_len, dtype=torch.long)
    positions = torch.zeros(n, max_options, dtype=torch.long)
    mask = torch.zeros(n, max_options, dtype=torch.long)
    for row, (k, _) in enumerate(specs):
        for slot in range(max_options):
            positions[row, slot] = 2 * (slot if slot < k else k - 1) + 1
        mask[row, :k] = 1
    ordered = torch.tensor([flag for _, flag in specs], dtype=torch.bool)
    return JointBatch(input_ids, attention_mask, positions, mask, ordered)


def run(model: SokudanJointModel, batch: JointBatch):
    return model(batch.input_ids, batch.attention_mask, batch.marker_positions,
                 batch.marker_mask, batch.ordered)


def test_probabilities_sum_to_one_over_the_real_options(joint_model):
    batch = make_joint_batch([(2, False), (5, False), (3, True), (4, True)])
    out = run(joint_model, batch)
    kept = out.probs * batch.marker_mask
    assert torch.allclose(kept.sum(dim=1), torch.ones(4), atol=1e-5)


def test_padded_slots_get_no_probability(joint_model):
    # Rows 0 and 2 have fewer options than the widest row, so their trailing slots
    # are padding. A padded slot that took probability would silently steal it from
    # a real option at serving time, where the answer is argmax over the row.
    batch = make_joint_batch([(2, False), (5, False), (3, True)])
    out = run(joint_model, batch)
    padding = out.probs * (1 - batch.marker_mask)
    assert float(padding.abs().max().detach()) == pytest.approx(0.0, abs=1e-6)


def test_ordinal_rows_get_a_monotone_survival_curve(joint_model):
    batch = make_joint_batch([(4, True), (3, False), (5, True)])
    out = run(joint_model, batch)
    rows = batch.ordered.nonzero(as_tuple=True)[0]
    survival = out.survival[rows]
    counts = batch.marker_mask[rows].sum(dim=1)
    for row, k in zip(survival, counts.tolist(), strict=True):
        steps = row[:k]
        assert torch.all(steps[:-1] >= steps[1:] - 1e-6), steps
        assert float(steps[-1].detach()) == pytest.approx(0.0, abs=1e-6)


def test_non_ordinal_rows_keep_nan_expectation(joint_model):
    # `expected_level` is only defined where the levels are ordered. A number here
    # would be read downstream as a meaningful level for a `choice` question.
    batch = make_joint_batch([(3, False), (4, True)])
    out = run(joint_model, batch)
    assert torch.isnan(out.expected_level[0])
    assert not torch.isnan(out.expected_level[1])


def test_expectation_lies_inside_the_level_range(joint_model):
    batch = make_joint_batch([(5, True), (3, True)])
    out = run(joint_model, batch)
    counts = batch.marker_mask.sum(dim=1)
    for value, k in zip(out.expected_level.tolist(), counts.tolist(), strict=True):
        assert 0.0 <= value <= k - 1


def test_no_ordinal_rows_leaves_expectation_and_survival_unset(joint_model):
    batch = make_joint_batch([(3, False), (2, False)])
    out = run(joint_model, batch)
    assert out.expected_level is None and out.survival is None


def test_use_ordinal_false_routes_score_rows_through_the_softmax():
    # Ablation (d) in docs/ablations.md. With the ordinal head switched off the
    # `score` row must come out of the same within-question softmax as `choice`,
    # not out of the cumulative link.
    backbone = _tiny_backbone()
    torch.manual_seed(1)
    plain = SokudanJointModel(backbone, use_ordinal=False)
    batch = make_joint_batch([(4, True)])
    out = run(plain, batch)
    assert out.expected_level is None and out.survival is None
    from sokudan.model.head import masked_softmax
    assert torch.allclose(out.probs, masked_softmax(out.marker_logits, batch.marker_mask))


def test_shapes_follow_the_batch(joint_model):
    for specs in ([(2, False)], [(3, True), (3, True)], [(2, False), (5, True), (4, False)]):
        batch = make_joint_batch(specs)
        out = run(joint_model, batch)
        assert out.probs.shape == batch.marker_positions.shape
        assert out.marker_logits.shape == batch.marker_positions.shape
        assert out.ordered.shape == (len(specs),)


def test_mismatched_marker_mask_is_rejected(joint_model):
    batch = make_joint_batch([(3, False)])
    with pytest.raises(ValueError, match="marker_mask"):
        joint_model(batch.input_ids, batch.attention_mask, batch.marker_positions,
                    batch.marker_mask[:, :-1], batch.ordered)


def test_mismatched_ordered_length_is_rejected(joint_model):
    batch = make_joint_batch([(3, False), (3, True)])
    with pytest.raises(ValueError, match="ordered"):
        joint_model(batch.input_ids, batch.attention_mask, batch.marker_positions,
                    batch.marker_mask, batch.ordered[:1])


def test_a_one_option_ordinal_row_is_rejected(joint_model):
    # A cumulative link over a single level has no threshold to place, and silently
    # returning 1.0 would hide a schema that lost its options upstream.
    batch = make_joint_batch([(1, True)], seq_len=8)
    with pytest.raises(ValueError, match="at least 2"):
        run(joint_model, batch)


def test_parameter_counts_add_up(joint_model):
    counts = joint_model.parameter_counts()
    # There is no decision head on this arm; the total is the backbone plus the two
    # small modules, and a regression that reintroduced one would show up here.
    assert counts["total"] == counts["backbone"] + counts["scorer"] + counts["ordinal"]
    assert counts["scorer"] == TINY.hidden_size  # (hidden, 1), no bias
    assert joint_model.hidden_size == TINY.hidden_size


def test_gradients_reach_the_backbone_and_the_scorer(joint_model):
    batch = make_joint_batch([(3, False), (4, True)])
    out = run(joint_model, batch)
    # Not `probs.sum()`: each row sums to one by construction, so its gradient is
    # exactly zero and the test would pass on a model with the scorer detached.
    # An NLL against a fixed target depends on the values.
    loss = -torch.log(out.probs[:, 0] + 1e-9).sum()
    loss.backward()
    assert joint_model.scorer.weight.grad is not None
    assert float(joint_model.scorer.weight.grad.abs().sum().detach()) > 0.0
    embedding = joint_model.backbone.model.get_input_embeddings().weight
    assert embedding.grad is not None
    assert float(embedding.grad.abs().sum().detach()) > 0.0
