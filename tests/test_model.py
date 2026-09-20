"""Model tests (SOKUDAN_SPEC.md §6.5 completion criteria).

    "ランダム初期化で forward が通り、sum(p) == 1、質問数・選択肢数を変えても
     形状が正しく、score の CDF が単調。"

Most of these run on a deliberately tiny backbone so the suite stays fast; the
properties under test are structural and do not depend on width. One test loads the
real 310M backbone to confirm the head's copy-init targets modules that actually
exist, and is marked `slow`.
"""

from __future__ import annotations

import pytest
import torch

from sokudan.model.backbone import Backbone, BackboneSpec
from sokudan.model.head import DecisionHead, masked_softmax
from sokudan.model.ordinal import OrdinalHead, expected_level, ranked_probability_score
from sokudan.model.sokudan import SokudanModel

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


@pytest.fixture
def tiny_model() -> SokudanModel:
    """A randomly initialised model on a tiny ModernBERT."""
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
        # ModernBertConfig's defaults point at the 50k-token vocabulary it ships
        # with; left alone they index past a 256-token embedding table.
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        cls_token_id=3,
        sep_token_id=4,
    )
    backbone = Backbone(AutoModel.from_config(config), BackboneSpec.from_config(config))
    return SokudanModel(backbone, n_head_layers=2)


def make_batch(specs: list[tuple[int, bool]], *, seq_len: int | None = None,
               state_len: int = 32):
    """Build a batch from `[(n_options, ordered), ...]`."""
    n = len(specs)
    max_options = max(k for k, _ in specs)
    # Markers sit at 2, 4, 6, ...; the sequence has to be long enough to hold them.
    if seq_len is None:
        seq_len = 2 * max_options + 4
    question_ids = torch.randint(0, 200, (n, seq_len))
    question_mask = torch.ones(n, seq_len, dtype=torch.long)
    state_ids = torch.randint(0, 200, (1, state_len))
    state_mask = torch.ones(1, state_len, dtype=torch.long)

    positions = torch.zeros(n, max_options, dtype=torch.long)
    marker_mask = torch.zeros(n, max_options, dtype=torch.long)
    for row, (k, _) in enumerate(specs):
        for j in range(k):
            positions[row, j] = 2 + j * 2
            marker_mask[row, j] = 1
        for j in range(k, max_options):
            positions[row, j] = 2  # padded slots point somewhere valid; the mask decides
    ordered = torch.tensor([o for _, o in specs])
    return (state_ids, state_mask, question_ids, question_mask, positions, marker_mask, ordered)


# --------------------------------------------------------------------------
# §6.5: forward runs, sum(p) == 1, shapes follow the schema
# --------------------------------------------------------------------------

def test_forward_runs_from_random_init(tiny_model) -> None:
    with torch.no_grad():
        out = tiny_model(*make_batch([(4, False)]))
    assert out.probs.shape == (1, 4)
    assert torch.isfinite(out.probs).all()


@pytest.mark.parametrize("n_questions", [1, 2, 5, 16])
def test_shape_follows_the_number_of_questions(tiny_model, n_questions) -> None:
    specs = [(3, False)] * n_questions
    with torch.no_grad():
        out = tiny_model(*make_batch(specs))
    assert out.probs.shape == (n_questions, 3)
    assert out.ordered.shape == (n_questions,)


@pytest.mark.parametrize("n_options", [2, 3, 5, 12, 20])
def test_shape_follows_the_number_of_options(tiny_model, n_options) -> None:
    with torch.no_grad():
        out = tiny_model(*make_batch([(n_options, False)]))
    assert out.probs.shape == (1, n_options)
    assert out.probs.sum(-1).item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("n_levels", [2, 3, 4, 5, 7])
def test_ordinal_shape_follows_k(tiny_model, n_levels) -> None:
    """§6.3: K is decided per request, so one head must serve every K."""
    with torch.no_grad():
        out = tiny_model(*make_batch([(n_levels, True)]))
    assert out.probs.shape == (1, n_levels)
    assert out.probs.sum(-1).item() == pytest.approx(1.0, abs=1e-5)


def test_every_row_sums_to_one_in_a_mixed_batch(tiny_model) -> None:
    specs = [(4, False), (3, True), (2, False), (7, True), (2, True)]
    with torch.no_grad():
        out = tiny_model(*make_batch(specs))
    torch.testing.assert_close(
        out.probs.sum(-1), torch.ones(len(specs)), atol=1e-5, rtol=0
    )


def test_padded_option_slots_carry_no_probability(tiny_model) -> None:
    specs = [(4, False), (2, True), (3, False)]
    batch = make_batch(specs)
    with torch.no_grad():
        out = tiny_model(*batch)
    marker_mask = batch[5]
    assert float((out.probs * (1 - marker_mask)).abs().sum()) == 0.0


def test_expected_level_is_reported_only_for_ordinal_rows(tiny_model) -> None:
    """§6.3: return the expectation *and* the distribution, not just one."""
    specs = [(4, False), (3, True)]
    with torch.no_grad():
        out = tiny_model(*make_batch(specs))
    assert out.expected_level is not None
    assert torch.isnan(out.expected_level[0])
    assert torch.isfinite(out.expected_level[1])


def test_a_batch_with_no_ordinal_rows_skips_the_ordinal_head(tiny_model) -> None:
    with torch.no_grad():
        out = tiny_model(*make_batch([(3, False), (2, False)]))
    assert out.expected_level is None
    assert out.survival is None


# --------------------------------------------------------------------------
# §6.5: the score CDF is monotone
# --------------------------------------------------------------------------

def test_survival_is_non_increasing_at_init(tiny_model) -> None:
    with torch.no_grad():
        out = tiny_model(*make_batch([(7, True)]))
    survival = out.survival[0]
    assert bool((survival[:-1] >= survival[1:] - 1e-6).all())


@pytest.mark.parametrize("seed", range(5))
def test_survival_is_non_increasing_under_arbitrary_weights(seed) -> None:
    """Monotone by construction means it holds for *any* weights, not just at init."""
    torch.manual_seed(seed)
    head = OrdinalHead(16)
    for parameter in head.parameters():
        torch.nn.init.normal_(parameter, std=3.0)
    _, survival = head(torch.randn(32, 6, 16) * 5)
    assert bool((survival[:, :-1] >= survival[:, 1:] - 1e-6).all())


@pytest.mark.parametrize("seed", range(5))
def test_ordinal_probabilities_are_non_negative_under_arbitrary_weights(seed) -> None:
    torch.manual_seed(seed)
    head = OrdinalHead(16)
    for parameter in head.parameters():
        torch.nn.init.normal_(parameter, std=3.0)
    probs, _ = head(torch.randn(32, 6, 16) * 5)
    assert bool((probs >= 0).all())
    torch.testing.assert_close(probs.sum(-1), torch.ones(32), atol=1e-5, rtol=0)


def test_top_survival_is_zero_because_no_level_exceeds_the_last() -> None:
    torch.manual_seed(0)
    _, survival = OrdinalHead(16)(torch.randn(4, 5, 16))
    assert float(survival[:, -1].abs().max()) == 0.0


@pytest.mark.parametrize("n_levels", [2, 3, 4, 5, 7])
def test_ordinal_head_starts_near_uniform_for_every_k(n_levels) -> None:
    """§6.2: a near-uniform start keeps the proper-scoring-rule gradient tame.

    Near-uniform, not exactly uniform. Exactness needs `softplus(cut_bias) ~ 0`,
    which also drives `softplus'` to ~0 and freezes the parameter -- measured at
    0.001 of movement across a 930-step epoch. The default trades exactness for a
    trainable spacing; what §6.2 actually asks for is that the model not start
    confident, and that still holds.
    """
    torch.manual_seed(0)
    probs, _ = OrdinalHead(32)(torch.randn(8, n_levels, 32))
    assert float((probs - 1.0 / n_levels).abs().max()) < 0.15
    # The real requirement: nowhere near confident.
    assert float(probs.max()) < max(0.65, 2.0 / n_levels)


@pytest.mark.parametrize("n_levels", [2, 3, 5, 7])
def test_an_exactly_uniform_start_is_still_available_but_untrainable(n_levels) -> None:
    """Documents the trade-off the default makes, so it is not rediscovered."""
    torch.manual_seed(0)
    head = OrdinalHead(32, cut_bias=-5.0)
    probs, _ = head(torch.randn(8, n_levels, 32))
    assert float((probs - 1.0 / n_levels).abs().max()) < 0.01   # exactly uniform
    # ...and this is why it does not train: the gradient is attenuated ~150x.
    assert float(torch.sigmoid(head.cut.bias)) < 0.01


def test_ordinal_head_handles_mixed_k_in_one_batch() -> None:
    torch.manual_seed(0)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.float)
    probs, _ = OrdinalHead(16)(torch.randn(3, 5, 16), mask)
    torch.testing.assert_close(probs.sum(-1), torch.ones(3), atol=1e-5, rtol=0)
    assert float((probs * (1 - mask)).abs().sum()) == 0.0


def test_ordinal_head_rejects_a_single_level() -> None:
    with pytest.raises(ValueError, match="at least 2 levels"):
        OrdinalHead(16)(torch.randn(2, 1, 16))


def test_expected_level_matches_a_hand_computation() -> None:
    probs = torch.tensor([[0.5, 0.0, 0.5], [0.0, 1.0, 0.0]])
    torch.testing.assert_close(expected_level(probs), torch.tensor([1.0, 1.0]))


# --------------------------------------------------------------------------
# RPS: the reason `score` is not scored with cross-entropy (§6.3)
# --------------------------------------------------------------------------

def test_rps_punishes_a_distant_miss_more_than_an_adjacent_one() -> None:
    adjacent = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]])
    distant = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
    gold = torch.tensor([2])
    assert ranked_probability_score(adjacent, gold) < ranked_probability_score(distant, gold)


def test_rps_of_a_perfect_prediction_is_zero() -> None:
    probs = torch.tensor([[0.0, 0.0, 1.0]])
    assert float(ranked_probability_score(probs, torch.tensor([2]))) == pytest.approx(0.0)


def test_rps_normalisation_makes_different_k_comparable() -> None:
    for k in (3, 5, 7):
        probs = torch.zeros(1, k)
        probs[0, 0] = 1.0
        value = ranked_probability_score(probs, torch.tensor([k - 1]))
        assert float(value) == pytest.approx(1.0)


def test_rps_respects_the_marker_mask() -> None:
    probs = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    assert float(ranked_probability_score(probs, torch.tensor([2]), mask)) == pytest.approx(0.0)


def test_rps_rejects_a_target_outside_its_rows_levels() -> None:
    probs = torch.tensor([[0.5, 0.5, 0.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    with pytest.raises(ValueError, match="outside its row"):
        ranked_probability_score(probs, torch.tensor([2]), mask)


def test_rps_is_differentiable() -> None:
    torch.manual_seed(0)
    head = OrdinalHead(16)
    probs, _ = head(torch.randn(8, 4, 16))
    ranked_probability_score(probs, torch.randint(0, 4, (8,))).backward()
    assert head.cut.weight.grad is not None
    assert torch.isfinite(head.cut.weight.grad).all()


# --------------------------------------------------------------------------
# head behaviour (§6.2)
# --------------------------------------------------------------------------

def test_cross_attention_output_projection_starts_at_zero() -> None:
    head = DecisionHead(32, n_layers=2, n_heads=4, intermediate_size=64)
    for layer in head.layers:
        assert float(layer.cross_attn.Wo.weight.abs().max()) == 0.0


def test_the_state_cannot_change_the_output_at_initialisation() -> None:
    """The point of the zero init: the head starts as the identity on the question."""
    torch.manual_seed(0)
    head = DecisionHead(32, n_layers=2, n_heads=4, intermediate_size=64)
    questions = torch.randn(3, 12, 32)
    with torch.no_grad():
        a = head(questions, torch.randn(1, 20, 32))
        b = head(questions, torch.randn(1, 20, 32) * 100)
    torch.testing.assert_close(a, b)


def test_the_state_does_change_the_output_once_cross_attention_is_non_zero() -> None:
    """...and the path is live, not dead: a trained Wo must matter."""
    torch.manual_seed(0)
    head = DecisionHead(32, n_layers=2, n_heads=4, intermediate_size=64)
    for layer in head.layers:
        torch.nn.init.normal_(layer.cross_attn.Wo.weight, std=0.5)
    questions = torch.randn(3, 12, 32)
    with torch.no_grad():
        a = head(questions, torch.randn(1, 20, 32))
        b = head(questions, torch.randn(1, 20, 32) * 5)
    assert float((a - b).abs().max()) > 1e-4


def test_gradients_reach_the_cross_attention_output_projection() -> None:
    """A zero-initialised projection that never gets gradient would stay dead."""
    torch.manual_seed(0)
    head = DecisionHead(32, n_layers=1, n_heads=4, intermediate_size=64)
    out = head(torch.randn(2, 10, 32), torch.randn(1, 16, 32))
    out.sum().backward()
    grad = head.layers[0].cross_attn.Wo.weight.grad
    assert grad is not None and float(grad.abs().max()) > 0.0


def test_a_single_state_broadcasts_across_the_question_batch(tiny_model) -> None:
    """§6.2: one state, N questions -- the state is encoded once per request."""
    torch.manual_seed(0)
    head = DecisionHead(32, n_layers=1, n_heads=4, intermediate_size=64)
    out = head(torch.randn(5, 10, 32), torch.randn(1, 16, 32))
    assert out.shape == (5, 10, 32)


def test_a_mismatched_state_batch_is_refused() -> None:
    head = DecisionHead(32, n_layers=1, n_heads=4, intermediate_size=64)
    with pytest.raises(ValueError, match="matches neither"):
        head(torch.randn(5, 10, 32), torch.randn(3, 16, 32))


def test_questions_do_not_see_each_other(tiny_model) -> None:
    """§6.2: order invariance is structural because questions are separate rows."""
    torch.manual_seed(0)
    head = DecisionHead(32, n_layers=2, n_heads=4, intermediate_size=64)
    for layer in head.layers:
        torch.nn.init.normal_(layer.cross_attn.Wo.weight, std=0.3)
    questions = torch.randn(4, 10, 32)
    state = torch.randn(1, 16, 32)
    with torch.no_grad():
        together = head(questions, state)
        alone = torch.cat([head(questions[i: i + 1], state) for i in range(4)], dim=0)
    torch.testing.assert_close(together, alone, atol=1e-5, rtol=1e-4)


def test_reordering_questions_permutes_the_output_identically(tiny_model) -> None:
    torch.manual_seed(0)
    order = torch.tensor([2, 0, 3, 1])
    specs = [(3, False), (4, False), (2, True), (5, True)]
    batch = make_batch(specs)
    with torch.no_grad():
        out = tiny_model(*batch)
        permuted = tiny_model(
            batch[0], batch[1], batch[2][order], batch[3][order],
            batch[4][order], batch[5][order], batch[6][order],
        )
    torch.testing.assert_close(out.probs[order], permuted.probs, atol=1e-5, rtol=1e-4)


def test_scorer_is_not_exactly_zero() -> None:
    """A zero scorer gives every option the same logit and the same gradient."""
    head = DecisionHead(32, n_layers=1, n_heads=4, intermediate_size=64)
    assert float(head.scorer.weight.abs().max()) > 0.0


def test_masked_softmax_ignores_padded_options() -> None:
    logits = torch.tensor([[1.0, 2.0, 50.0]])
    mask = torch.tensor([[1, 1, 0]])
    probs = masked_softmax(logits, mask)
    assert float(probs[0, 2]) == 0.0
    assert float(probs.sum()) == pytest.approx(1.0)


def test_masked_softmax_matches_plain_softmax_when_nothing_is_padded() -> None:
    logits = torch.randn(4, 5)
    torch.testing.assert_close(masked_softmax(logits, torch.ones(4, 5, dtype=torch.long)),
                               logits.softmax(-1))


def test_padding_in_the_question_sequence_does_not_change_real_positions() -> None:
    """Right-padding a question must not move its marker representations."""
    torch.manual_seed(0)
    head = DecisionHead(32, n_layers=2, n_heads=4, intermediate_size=64)
    for layer in head.layers:
        torch.nn.init.normal_(layer.cross_attn.Wo.weight, std=0.3)
    real_len = 10
    question = torch.randn(1, real_len, 32)
    padded = torch.cat([question, torch.randn(1, 6, 32)], dim=1)
    state = torch.randn(1, 16, 32)
    with torch.no_grad():
        short = head(question, state, question_mask=torch.ones(1, real_len, dtype=torch.long))
        long = head(
            padded, state,
            question_mask=torch.cat(
                [torch.ones(1, real_len, dtype=torch.long), torch.zeros(1, 6, dtype=torch.long)],
                dim=1,
            ),
        )
    torch.testing.assert_close(short, long[:, :real_len], atol=1e-4, rtol=1e-3)


# --------------------------------------------------------------------------
# the real backbone
# --------------------------------------------------------------------------

@pytest.mark.slow
def test_head_copy_init_targets_real_modules_on_the_actual_backbone() -> None:
    """Guards against the head drifting from `ModernBertEncoderLayer`'s structure."""
    try:
        backbone = Backbone.load()
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"backbone weights unavailable: {type(exc).__name__}: {exc}")

    model = SokudanModel(backbone, n_head_layers=2)
    for head_layer, source in zip(model.head.layers, backbone.layers[-2:], strict=True):
        assert torch.equal(head_layer.attn.Wqkv.weight, source.attn.Wqkv.weight)
        assert torch.equal(head_layer.attn.Wo.weight, source.attn.Wo.weight)
        assert torch.equal(head_layer.mlp.Wi.weight, source.mlp.Wi.weight)
        assert torch.equal(head_layer.mlp.Wo.weight, source.mlp.Wo.weight)
    assert model.hidden_size == 768
    assert model.parameter_counts()["total"] > 300_000_000


def test_an_out_of_range_marker_position_is_reported_clearly() -> None:
    """§5.2: a drifting marker position must fail loudly, not gather garbage."""
    head = DecisionHead(32, n_layers=1, n_heads=4, intermediate_size=64)
    with pytest.raises(ValueError, match="outside a question of length"):
        head.gather_markers(torch.randn(1, 10, 32), torch.tensor([[2, 99]]))


def test_a_negative_marker_position_is_refused() -> None:
    head = DecisionHead(32, n_layers=1, n_heads=4, intermediate_size=64)
    with pytest.raises(ValueError, match="non-negative"):
        head.gather_markers(torch.randn(1, 10, 32), torch.tensor([[2, -1]]))
