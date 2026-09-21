"""Round-trip properties for the joint encoder (s2c).

The separate encoder is tested in `test_encoding.py`. Every property there that is
about *markers* has to hold here too, because the two modes share `marker_texts`, the
special-token layout and the downstream scorer -- a marker that lands in the wrong
place is the failure §5.2 warns is silent, and it would be silent in either mode.

What is deliberately different is tested as well: the joint sequence contains the
state, and the question part does not move when the state changes.
"""

from __future__ import annotations

import pytest

from sokudan.encoding.question import (
    QuestionTooLongError,
    encode_joint,
    encode_question,
)
from sokudan.schema.question import parse_question

CHOICE = {
    "type": "choice",
    "instructions": "この問い合わせはどの部署が担当すべきか",
    "criteria": {"請求": "支払い・返金", "技術": "不具合・障害", "営業": "料金・新規契約"},
}
SCORE = {
    "type": "score",
    "instructions": "この依頼の緊急度は",
    "criteria": ["急がない", "早めに", "業務が止まっている"],
}
BOOL = {"type": "bool", "instructions": "解約を示唆しているか"}

STATE = "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。"


@pytest.fixture(params=[CHOICE, SCORE, BOOL], ids=["choice", "score", "bool"])
def question(request):
    return parse_question(request.param)


def test_every_marker_position_holds_a_mask_token(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    for position in encoded.marker_positions:
        assert encoded.input_ids[position] == tokenizer.mask_token_id


def test_marker_count_matches_the_option_count(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    assert encoded.n_markers == len(question.labels)
    assert len(encoded.marker_positions) == len(question.labels)


def test_the_only_mask_tokens_are_the_markers(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    masks = [i for i, t in enumerate(encoded.input_ids) if t == tokenizer.mask_token_id]
    assert masks == encoded.marker_positions


def test_marker_positions_are_strictly_increasing(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    assert encoded.marker_positions == sorted(set(encoded.marker_positions))


def test_markers_sit_in_the_question_part_before_the_state(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    assert max(encoded.marker_positions) < encoded.n_question_tokens


def test_the_joint_sequence_contains_the_state(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    decoded = tokenizer.decode(encoded.input_ids)
    assert "二回引き落とされて" in decoded
    # The separate encoder must not; that is the difference being measured.
    assert "二回引き落とされて" not in tokenizer.decode(
        encode_question(question, tokenizer).input_ids
    )


def test_decoded_sequence_contains_instructions_and_every_option(question, tokenizer) -> None:
    decoded = tokenizer.decode(encode_joint(question, STATE, tokenizer).input_ids)
    assert question.instructions in decoded
    for label in question.labels:
        assert label in decoded


def test_changing_the_state_does_not_move_the_markers(question, tokenizer) -> None:
    short = encode_joint(question, "短い本文。", tokenizer)
    long = encode_joint(question, STATE * 3, tokenizer)
    assert short.marker_positions == long.marker_positions
    assert short.n_question_tokens == long.n_question_tokens


def test_reordering_options_moves_the_markers_with_them(tokenizer) -> None:
    forward = parse_question(CHOICE)
    reversed_criteria = dict(reversed(list(CHOICE["criteria"].items())))
    backward = parse_question({**CHOICE, "criteria": reversed_criteria})

    a = encode_joint(forward, STATE, tokenizer)
    b = encode_joint(backward, STATE, tokenizer)
    assert a.n_markers == b.n_markers
    # The option whose marker is first differs, which is exactly what §7.2 shuffles.
    assert a.input_ids[: a.marker_positions[0]] != b.input_ids[: b.marker_positions[0]]


def test_encoding_is_deterministic(question, tokenizer) -> None:
    first = encode_joint(question, STATE, tokenizer)
    second = encode_joint(question, STATE, tokenizer)
    assert first == second


def test_attention_mask_matches_the_sequence_length(question, tokenizer) -> None:
    encoded = encode_joint(question, STATE, tokenizer)
    assert len(encoded.attention_mask) == len(encoded.input_ids)
    assert set(encoded.attention_mask) == {1}


def test_empty_state_is_legal(question, tokenizer) -> None:
    encoded = encode_joint(question, "", tokenizer)
    assert encoded.n_state_tokens == 0
    assert encoded.n_markers == len(question.labels)


def test_a_long_state_is_truncated_and_says_so(question, tokenizer) -> None:
    encoded = encode_joint(question, "あ" * 5000, tokenizer, max_tokens=256)
    assert encoded.truncated
    assert len(encoded.input_ids) <= 256
    # Truncation takes the state, never a marker.
    assert encoded.n_markers == len(question.labels)
    for position in encoded.marker_positions:
        assert encoded.input_ids[position] == tokenizer.mask_token_id


def test_a_short_state_is_not_flagged_as_truncated(question, tokenizer) -> None:
    assert not encode_joint(question, STATE, tokenizer).truncated


def test_a_question_that_cannot_fit_raises_rather_than_dropping_an_option(tokenizer) -> None:
    wide = parse_question({
        "type": "choice",
        "instructions": "どれか",
        "criteria": {f"選択肢{i}": "説明" * 20 for i in range(30)},
    })
    with pytest.raises(QuestionTooLongError):
        encode_joint(wide, STATE, tokenizer, max_tokens=64)


def test_both_modes_agree_on_how_many_markers_a_question_has(question, tokenizer) -> None:
    separate = encode_question(question, tokenizer)
    joint = encode_joint(question, STATE, tokenizer)
    assert separate.n_markers == joint.n_markers
    assert separate.ordered == joint.ordered


def test_both_modes_put_the_same_option_first(tokenizer) -> None:
    """The gold label indexes options, so the two modes must order them identically."""
    question = parse_question(CHOICE)
    separate = encode_question(question, tokenizer)
    joint = encode_joint(question, STATE, tokenizer)
    for index in range(separate.n_markers):
        before_separate = separate.input_ids[
            (separate.marker_positions[index - 1] + 1 if index else 0):
            separate.marker_positions[index]
        ]
        before_joint = joint.input_ids[
            (joint.marker_positions[index - 1] + 1 if index else 0):
            joint.marker_positions[index]
        ]
        assert tokenizer.decode(before_separate).strip().endswith(
            tokenizer.decode(before_joint).strip()[-4:]
        )
