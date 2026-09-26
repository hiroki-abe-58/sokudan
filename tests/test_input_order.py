"""`input_order` for the joint arm (docs/input_order.md, Phase 1).

(a) The default order is byte-identical to the encoder before `input_order` existed:
    `tests/fixtures/joint_golden.json` was written by `scripts/joint_golden.py` on
    commit 709abea, before the change.
(b) `state_first` is a permutation of `question_first`: same token multiset.
(c) In `state_first`, every marker position the model reads holds `<mask>`, and there
    are exactly as many as the question has options (choice) or levels (score).
(d) A `state_first` checkpoint reloads as `state_first`; one without the key reloads
    as `question_first`; a batch in the wrong order is refused.
(e) No row of any set the experiment uses is truncated, in either order.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import torch

from sokudan.encoding.question import INPUT_ORDERS, encode_joint
from sokudan.schema.question import parse_question
from sokudan.train.dataset import Example, JointCollator

FIXTURE = Path(__file__).parent / "fixtures" / "joint_golden.json"


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _examples(golden: dict) -> list[Example]:
    return [Example.from_row({"state": s["state"], "question": s["question"], "label": 0,
                              "kind": s["kind"]}) for s in golden["samples"]]


def test_fixture_covers_every_kind(golden):
    assert Counter(s["kind"] for s in golden["samples"]) == {"choice": 3, "score": 3,
                                                            "bool": 3}


def test_a_default_order_matches_the_pre_change_encoder(golden, tokenizer):
    for sample in golden["samples"]:
        question = parse_question(sample["question"])
        enc = encode_joint(question, sample["state"], tokenizer)
        assert enc.input_ids == sample["input_ids"], sample["kind"]
        assert enc.attention_mask == sample["attention_mask"], sample["kind"]
        assert enc.marker_positions == sample["marker_positions"], sample["kind"]
        explicit = encode_joint(question, sample["state"], tokenizer,
                                input_order="question_first")
        assert explicit == enc


def test_a_default_truncation_matches_the_pre_change_encoder(golden, tokenizer):
    for sample in golden["samples"]:
        question = parse_question(sample["question"])
        full = encode_joint(question, sample["state"], tokenizer)
        limit = len(full.input_ids) - min(20, full.n_state_tokens - 1)
        small = encode_joint(question, sample["state"], tokenizer, max_tokens=limit)
        stored = sample["truncated_minus20"]
        assert small.input_ids == stored["input_ids"]
        assert small.marker_positions == stored["marker_positions"]
        assert small.truncated == stored["truncated"]


def test_a_default_collated_batch_matches(golden, tokenizer):
    batch = JointCollator(tokenizer)(_examples(golden))
    assert batch.input_order == "question_first"
    for name in ("input_ids", "attention_mask", "marker_positions", "marker_mask"):
        assert getattr(batch, name).tolist() == golden["collated"][name], name


@pytest.mark.slow
def test_a_default_order_matches_on_every_row_of_every_file(golden, tokenizer):
    from scripts.joint_golden import fingerprint

    assert fingerprint(tokenizer)["file_sha256"] == golden["file_sha256"]


def test_b_state_first_is_a_permutation(golden, tokenizer):
    for sample in golden["samples"]:
        question = parse_question(sample["question"])
        a = encode_joint(question, sample["state"], tokenizer)
        b = encode_joint(question, sample["state"], tokenizer, input_order="state_first")
        assert Counter(a.input_ids) == Counter(b.input_ids), sample["kind"]
        assert len(a.input_ids) == len(b.input_ids)
        assert a.input_ids != b.input_ids  # it did move something
        assert (a.n_state_tokens, a.n_question_tokens) == (b.n_state_tokens,
                                                           b.n_question_tokens)


def test_b_state_first_layout(golden, tokenizer):
    from sokudan.encoding.special_tokens import SpecialTokenLayout

    layout = SpecialTokenLayout.from_tokenizer(tokenizer)
    sample = golden["samples"][0]
    question = parse_question(sample["question"])
    state_ids = list(tokenizer(sample["state"], add_special_tokens=False)["input_ids"])
    b = encode_joint(question, sample["state"], tokenizer, input_order="state_first")
    start = len(layout.prefix)
    assert b.input_ids[:start] == layout.prefix
    assert b.input_ids[start:start + len(state_ids)] == state_ids
    after = start + len(state_ids)
    assert b.input_ids[after:after + len(layout.separator)] == layout.separator
    assert b.input_ids[-len(layout.suffix):] == layout.suffix
    # The last marker is the last token before the suffix: the markers are now at the
    # far end, right after the question block.
    assert b.marker_positions[-1] == len(b.input_ids) - len(layout.suffix) - 1


def test_c_state_first_markers_are_masks_and_count_the_options(golden, tokenizer):
    mask = tokenizer.mask_token_id
    examples = _examples(golden)
    for example in examples:
        enc = encode_joint(example.question, example.state, tokenizer,
                           input_order="state_first")
        assert [enc.input_ids[p] for p in enc.marker_positions] == [mask] * len(
            example.question.labels), example.kind
    batch = JointCollator(tokenizer, input_order="state_first")(examples)
    assert batch.input_order == "state_first"
    for row, example in enumerate(examples):
        n = int(batch.marker_mask[row].sum())
        assert n == len(example.question.labels)
        positions = batch.marker_positions[row, :n]
        assert (batch.input_ids[row, positions] == mask).all()


def test_unknown_order_is_rejected(golden, tokenizer):
    sample = golden["samples"][0]
    with pytest.raises(ValueError):
        encode_joint(parse_question(sample["question"]), sample["state"], tokenizer,
                     input_order="interleaved")


def test_separate_arm_refuses_an_input_order(tokenizer):
    from sokudan.train.loop import TrainConfig, build_collator

    with pytest.raises(ValueError):
        build_collator(tokenizer, TrainConfig(encoding="separate",
                                              input_order="state_first"))
    assert INPUT_ORDERS[:2] == ("question_first", "state_first")


def test_d_run_model_refuses_a_batch_in_the_other_order(golden, tokenizer):
    from sokudan.train.loop import run_model

    class Stub(torch.nn.Module):
        input_order = "state_first"

        def forward(self, *args):
            return "called"

    batch = JointCollator(tokenizer)(_examples(golden))
    with pytest.raises(ValueError):
        run_model(Stub(), batch)
    good = JointCollator(tokenizer, input_order="state_first")(_examples(golden))
    assert run_model(Stub(), good) == "called"


@pytest.mark.slow
def test_d_state_first_survives_save_and_reload(tmp_path):
    import importlib.util

    from sokudan.model.joint import SokudanJointModel

    root = Path(__file__).resolve().parent.parent

    def load(name: str):
        spec = importlib.util.spec_from_file_location(name, root / "scripts" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    try:
        model = SokudanJointModel.from_pretrained_backbone(input_order="state_first")
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"backbone unavailable: {type(exc).__name__}: {exc}")
    train, eval_heldout = load("train"), load("eval_heldout")
    path = train.save_checkpoint(model, tmp_path / "model.pt", head_layers=2,
                                 encoding="joint")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["config"]["input_order"] == "state_first"
    assert blob["config"]["local_attention"] == 128

    reloaded, encoding, _ = eval_heldout.load_checkpoint(str(path))
    assert encoding == "joint"
    assert reloaded.input_order == "state_first"
    assert reloaded.backbone.spec.local_attention == 128

    del blob["config"]["input_order"]
    legacy = tmp_path / "legacy.pt"
    torch.save(blob, legacy)
    old, _, _ = eval_heldout.load_checkpoint(str(legacy))
    assert old.input_order == "question_first"


@pytest.mark.slow
def test_e_no_row_is_truncated_in_either_order(tokenizer):
    from scripts.input_order_checks import check, data_sets

    for name, rows in data_sets().items():
        result = check(rows, tokenizer)
        for order in INPUT_ORDERS:
            assert result[order]["truncated"] == 0, (name, order)
            assert result[order]["marker_not_on_mask"] == 0, (name, order)
            assert result[order]["marker_count_mismatch"] == 0, (name, order)
        assert result["multiset_differs"] == 0, name
