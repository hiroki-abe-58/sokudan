"""`input_order="sandwich"` (docs/sandwich.md, Phase 2).

(a) question_first and state_first are byte-identical to the encoder before sandwich
    existed (`tests/fixtures/joint_golden.json` from 709abea, and
    `tests/fixtures/joint_golden_state_first.json` from ca7a601).
(b) sandwich's token multiset = question_first's + one question block (with its
    separator).
(c) Front and back markers are all <mask>, pair slot i with slot i, and there are as
    many as options / levels.
(d) With identical hidden states at the front and back positions, the averaged marker
    states equal the single-position ones; and a model's output with back == front
    equals its output without the back positions.
(e) A sandwich checkpoint reloads as sandwich.
(f) No row of the training data or of any evaluation set is truncated in sandwich.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from sokudan.encoding.question import INPUT_ORDERS, encode_joint
from sokudan.encoding.special_tokens import SpecialTokenLayout
from sokudan.model.joint import gather_marker_states
from sokudan.schema.question import parse_question
from sokudan.train.dataset import Example, JointCollator

FIXTURES = Path(__file__).parent / "fixtures"


def _golden(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _examples(golden: dict) -> list[Example]:
    return [Example.from_row({"state": s["state"], "question": s["question"], "label": 0,
                              "kind": s["kind"]}) for s in golden["samples"]]


def test_orders():
    assert INPUT_ORDERS == ("question_first", "state_first", "sandwich")


@pytest.mark.parametrize("fixture,order", [("joint_golden.json", "question_first"),
                                           ("joint_golden_state_first.json", "state_first")])
def test_a_single_block_orders_unchanged(tokenizer, fixture, order):
    golden = _golden(fixture)
    for sample in golden["samples"]:
        enc = encode_joint(parse_question(sample["question"]), sample["state"], tokenizer,
                           input_order=order)
        assert enc.input_ids == sample["input_ids"]
        assert enc.attention_mask == sample["attention_mask"]
        assert enc.marker_positions == sample["marker_positions"]
        assert enc.marker_positions_back is None
    batch = JointCollator(tokenizer, input_order=order)(_examples(golden))
    for name in ("input_ids", "attention_mask", "marker_positions", "marker_mask"):
        assert getattr(batch, name).tolist() == golden["collated"][name], name
    assert batch.marker_positions_back is None


@pytest.mark.slow
@pytest.mark.parametrize("fixture,order", [("joint_golden.json", "question_first"),
                                           ("joint_golden_state_first.json", "state_first")])
def test_a_single_block_orders_unchanged_on_every_row(tokenizer, fixture, order):
    from scripts.joint_golden import fingerprint

    assert fingerprint(tokenizer, order)["file_sha256"] == _golden(fixture)["file_sha256"]


def test_b_sandwich_adds_exactly_one_question_block(tokenizer):
    layout = SpecialTokenLayout.from_tokenizer(tokenizer)
    for sample in _golden("joint_golden.json")["samples"]:
        question = parse_question(sample["question"])
        qf = encode_joint(question, sample["state"], tokenizer)
        sw = encode_joint(question, sample["state"], tokenizer, input_order="sandwich")
        block_len = qf.n_question_tokens - len(layout.prefix) - len(layout.separator)
        block = qf.input_ids[len(layout.prefix): len(layout.prefix) + block_len]
        assert Counter(sw.input_ids) == Counter(qf.input_ids) + Counter(block + layout.separator)
        # The two blocks are the same tokens.
        back_start = len(sw.input_ids) - len(layout.suffix) - block_len
        assert sw.input_ids[back_start: back_start + block_len] == block
        assert sw.n_state_tokens == qf.n_state_tokens


def test_c_marker_pairs(tokenizer):
    mask = tokenizer.mask_token_id
    examples = _examples(_golden("joint_golden.json"))
    for e in examples:
        enc = encode_joint(e.question, e.state, tokenizer, input_order="sandwich")
        n = len(e.question.labels)
        assert len(enc.marker_positions) == len(enc.marker_positions_back) == n
        assert all(enc.input_ids[p] == mask for p in enc.marker_positions)
        assert all(enc.input_ids[p] == mask for p in enc.marker_positions_back)
        gaps = {b - f for f, b in zip(enc.marker_positions, enc.marker_positions_back,
                                      strict=True)}
        assert len(gaps) == 1 and gaps.pop() > 0          # slot i pairs with slot i
        assert enc.marker_positions == sorted(enc.marker_positions)
    batch = JointCollator(tokenizer, input_order="sandwich")(examples)
    for row, e in enumerate(examples):
        n = int(batch.marker_mask[row].sum())
        assert n == len(e.question.labels)
        assert (batch.input_ids[row, batch.marker_positions[row, :n]] == mask).all()
        assert (batch.input_ids[row, batch.marker_positions_back[row, :n]] == mask).all()


def test_d_identical_pairs_average_to_the_single_path():
    torch.manual_seed(0)
    half = torch.randn(3, 10, 8)
    hidden = torch.cat([half, half], dim=1)  # position p + 10 carries p's state
    front = torch.tensor([[2, 4, 6], [1, 3, 3], [5, 7, 9]])
    back = front + 10
    single = gather_marker_states(hidden, front)
    averaged = gather_marker_states(hidden, front, back)
    assert torch.allclose(single, averaged, atol=0, rtol=0)
    with pytest.raises(ValueError):
        gather_marker_states(hidden, front, back[:, :2])


def test_d_model_with_back_equal_front_matches_single(tokenizer):
    from tests.test_grad_accum import _tiny_model

    model = _tiny_model(tokenizer).eval()
    examples = _examples(_golden("joint_golden.json"))
    batch = JointCollator(tokenizer)(examples)
    with torch.no_grad():
        plain = model(batch.input_ids, batch.attention_mask, batch.marker_positions,
                      batch.marker_mask, batch.ordered)
        model.input_order = "sandwich"
        same = model(batch.input_ids, batch.attention_mask, batch.marker_positions,
                     batch.marker_mask, batch.ordered,
                     marker_positions_back=batch.marker_positions)
        with pytest.raises(ValueError):
            model(batch.input_ids, batch.attention_mask, batch.marker_positions,
                  batch.marker_mask, batch.ordered)
    assert torch.equal(plain.probs, same.probs)


@pytest.mark.slow
def test_e_sandwich_survives_save_and_reload(tmp_path):
    import importlib.util

    from sokudan.model.joint import SokudanJointModel

    root = Path(__file__).resolve().parent.parent

    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        model = SokudanJointModel.from_pretrained_backbone(input_order="sandwich")
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"backbone unavailable: {type(exc).__name__}: {exc}")
    train, eval_heldout = load("train"), load("eval_heldout")
    path = train.save_checkpoint(model, tmp_path / "model.pt", head_layers=2, encoding="joint")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["config"]["input_order"] == "sandwich"
    reloaded, encoding, _ = eval_heldout.load_checkpoint(str(path))
    assert encoding == "joint" and reloaded.input_order == "sandwich"
    assert reloaded.backbone.spec.local_attention == 128


@pytest.mark.slow
def test_f_no_row_is_truncated_in_sandwich(tokenizer):
    from scripts.input_order_checks import check, data_sets

    for name, rows in data_sets().items():
        result = check(rows, tokenizer)
        assert result["sandwich"]["truncated"] == 0, name
        assert result["sandwich"]["marker_not_on_mask"] == 0, name
        assert result["sandwich"]["marker_count_mismatch"] == 0, name
