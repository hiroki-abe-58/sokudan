"""Encoding round-trip and marker positions (SOKUDAN_SPEC.md §5.2, §1-3).

The completion criteria for Phase 1 are named in §5.2: schema validation, encoding
round-trip, marker positions, and the boundary cases (one option, a state over 8192
tokens, emoji, half-width kana, the empty string).

These run against the real backbone tokenizer. A stub would agree with whatever
assumption the encoder makes about special tokens, and that assumption is exactly
what went wrong once already: `[CLS]`/`[SEP]` in §5.2 are role names, but this
tokenizer's `cls_token_id`/`sep_token_id` are tokens its post-processor never emits.
"""

from __future__ import annotations

import pytest

from sokudan.encoding.question import (
    MAX_QUESTION_TOKENS,
    QuestionEncoderCache,
    QuestionTooLongError,
    encode_question,
    question_cache_key,
)
from sokudan.encoding.special_tokens import SpecialTokenLayout, require_mask_token_id
from sokudan.encoding.state import MAX_STATE_TOKENS, encode_state, encode_states
from sokudan.schema.question import BoolQuestion, ChoiceQuestion, ScoreQuestion, parse_question

CHOICE = ChoiceQuestion(
    instructions="この問い合わせはどの部署が担当すべきか",
    criteria={"請求": "支払い・返金", "技術": "不具合・障害", "営業": "料金・新規契約"},
)
SCORE = ScoreQuestion(
    instructions="この依頼の緊急度は",
    criteria=["急がない", "早めに", "業務が止まっている"],
)
BOOL = BoolQuestion(instructions="解約を示唆しているか")

ALL_QUESTIONS = [CHOICE, SCORE, BOOL]


# --------------------------------------------------------------------------
# special token layout
# --------------------------------------------------------------------------

def test_layout_is_probed_not_taken_from_attribute_names(tokenizer) -> None:
    """The regression guard. `cls_token_id` is 6 here but the template emits 1."""
    layout = SpecialTokenLayout.from_tokenizer(tokenizer)
    single = list(tokenizer("あ", add_special_tokens=True)["input_ids"])
    bare = list(tokenizer("あ", add_special_tokens=False)["input_ids"])
    assert single == [*layout.prefix, *bare, *layout.suffix]


def test_layout_separator_matches_the_tokenizers_pair_template(tokenizer) -> None:
    layout = SpecialTokenLayout.from_tokenizer(tokenizer)
    pair = list(tokenizer("あ", "い", add_special_tokens=True)["input_ids"])
    a = list(tokenizer("あ", add_special_tokens=False)["input_ids"])
    b = list(tokenizer("い", add_special_tokens=False)["input_ids"])
    assert pair == [*layout.prefix, *a, *layout.separator, *b, *layout.suffix]


def test_mask_token_id_comes_from_the_tokenizer(tokenizer) -> None:
    assert require_mask_token_id(tokenizer) == tokenizer.mask_token_id


# --------------------------------------------------------------------------
# marker positions -- if these drift, everything downstream is silently wrong
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", ALL_QUESTIONS, ids=lambda q: q.type)
def test_every_marker_position_holds_a_mask_token(question, tokenizer) -> None:
    encoded = encode_question(question, tokenizer)
    for position in encoded.marker_positions:
        assert encoded.input_ids[position] == tokenizer.mask_token_id


@pytest.mark.parametrize("question", ALL_QUESTIONS, ids=lambda q: q.type)
def test_marker_count_matches_the_option_count(question, tokenizer) -> None:
    from sokudan.schema.question import n_markers

    encoded = encode_question(question, tokenizer)
    assert encoded.n_markers == n_markers(question)
    assert len(encoded.marker_positions) == n_markers(question)


def test_marker_positions_are_strictly_increasing(tokenizer) -> None:
    positions = encode_question(CHOICE, tokenizer).marker_positions
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


def test_the_only_mask_tokens_are_the_markers(tokenizer) -> None:
    """Nothing else in the sequence may look like a marker."""
    encoded = encode_question(CHOICE, tokenizer)
    mask_positions = [
        i for i, token in enumerate(encoded.input_ids) if token == tokenizer.mask_token_id
    ]
    assert mask_positions == encoded.marker_positions


def test_an_option_containing_the_literal_mask_string_does_not_add_markers(tokenizer) -> None:
    """Why the encoder concatenates id runs instead of searching a rendered string."""
    question = ChoiceQuestion(
        instructions="どれか",
        criteria={"通常": "普通の説明", "罠": f"説明に {tokenizer.mask_token} が入っている"},
    )
    encoded = encode_question(question, tokenizer)
    assert encoded.n_markers == 2
    for position in encoded.marker_positions:
        assert encoded.input_ids[position] == tokenizer.mask_token_id


def test_question_encoding_does_not_contain_the_state(tokenizer) -> None:
    """§5.2's central change from v1: the question side never sees the state."""
    encoded = encode_question(CHOICE, tokenizer)
    decoded = tokenizer.decode(encoded.input_ids)
    assert "問い合わせ" in decoded  # the instructions are there
    assert "二重" not in decoded    # nothing from any particular state is


# --------------------------------------------------------------------------
# round-trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("question", ALL_QUESTIONS, ids=lambda q: q.type)
def test_decoded_question_contains_instructions_and_every_option(question, tokenizer) -> None:
    from sokudan.schema.question import marker_texts

    decoded = tokenizer.decode(encode_question(question, tokenizer).input_ids)
    assert question.instructions in decoded
    for text in marker_texts(question):
        # Labels survive the round-trip; the `label: description` join may re-space.
        assert text.split(":")[0].strip() in decoded


@pytest.mark.parametrize("question", ALL_QUESTIONS, ids=lambda q: q.type)
def test_encoding_is_deterministic(question, tokenizer) -> None:
    first = encode_question(question, tokenizer)
    second = encode_question(question, tokenizer)
    assert first.input_ids == second.input_ids
    assert first.marker_positions == second.marker_positions
    assert first.cache_key == second.cache_key


def test_attention_mask_matches_the_sequence_length(tokenizer) -> None:
    encoded = encode_question(CHOICE, tokenizer)
    assert len(encoded.attention_mask) == len(encoded.input_ids)
    assert set(encoded.attention_mask) == {1}


# --------------------------------------------------------------------------
# option order -- the property `docs/baseline_ja.md` §6.2 shows Laya failing
# --------------------------------------------------------------------------

def test_reordering_options_moves_the_markers_with_them(tokenizer) -> None:
    """Shuffled schemas must stay encodable and stay consistent (§7.2)."""
    forward = ChoiceQuestion(instructions="x", criteria={"a": "A", "b": "B", "c": "C"})
    backward = ChoiceQuestion(instructions="x", criteria={"c": "C", "b": "B", "a": "A"})

    enc_f = encode_question(forward, tokenizer)
    enc_b = encode_question(backward, tokenizer)

    assert enc_f.n_markers == enc_b.n_markers == 3
    assert enc_f.input_ids != enc_b.input_ids          # a different schema, really
    assert len(enc_f.input_ids) == len(enc_b.input_ids)  # same budget either way
    for position in enc_b.marker_positions:
        assert enc_b.input_ids[position] == tokenizer.mask_token_id


def test_reordering_options_changes_the_cache_key(tokenizer) -> None:
    """Otherwise a shuffled schema would silently reuse the wrong encoding."""
    forward = ChoiceQuestion(instructions="x", criteria={"a": "A", "b": "B"})
    backward = ChoiceQuestion(instructions="x", criteria={"b": "B", "a": "A"})
    assert question_cache_key(forward, tokenizer) != question_cache_key(backward, tokenizer)


def test_score_of_different_k_uses_the_same_encoder(tokenizer) -> None:
    """§6.3: K is decided per request, so nothing may hardcode a level count."""
    for k in range(2, 8):
        question = ScoreQuestion(instructions="程度は", criteria=[f"L{i}" for i in range(k)])
        encoded = encode_question(question, tokenizer)
        assert encoded.n_markers == k
        assert encoded.ordered


# --------------------------------------------------------------------------
# boundary cases named in §5.2
# --------------------------------------------------------------------------

def test_single_option_choice_encodes(tokenizer) -> None:
    encoded = encode_question(ChoiceQuestion(instructions="確認", criteria={"唯一": "常に"}),
                              tokenizer)
    assert encoded.n_markers == 1
    assert encoded.input_ids[encoded.marker_positions[0]] == tokenizer.mask_token_id


def test_empty_state_encodes_to_the_special_tokens_only(tokenizer) -> None:
    layout = SpecialTokenLayout.from_tokenizer(tokenizer)
    encoded = encode_state("", tokenizer)
    assert encoded.input_ids == [*layout.prefix, *layout.suffix]
    assert not encoded.truncated


@pytest.mark.parametrize(
    "state",
    [
        "🙇‍♂️ご対応ありがとうございます🙏 至急おねがいします‼️",   # emoji, including ZWJ
        "ﾊﾝｶｸｶﾅﾃﾞｽ ｾｲｷｭｳｶﾞ ﾆｼﾞｭｳﾆ ﾅｯﾃｲﾏｽ",                       # half-width kana
        "　　全角スペースと\t制御文字\r\nが混ざった本文　",            # odd whitespace
        "A" * 5000,                                              # long ASCII run
        "あ" * 3000,                                             # long CJK run
    ],
    ids=["emoji", "halfwidth_kana", "whitespace", "long_ascii", "long_cjk"],
)
def test_odd_states_encode_without_raising(state, tokenizer) -> None:
    encoded = encode_state(state, tokenizer)
    assert encoded.n_tokens >= 2
    assert len(encoded.attention_mask) == len(encoded.input_ids)


def test_state_over_the_limit_is_truncated_and_says_so(tokenizer) -> None:
    """§5.2 boundary case: a state past 8192 tokens. Truncation must be reported."""
    encoded = encode_state("この文章は長いです。" * 6000, tokenizer)
    assert encoded.truncated is True
    assert encoded.n_tokens == MAX_STATE_TOKENS


def test_short_state_is_not_flagged_as_truncated(tokenizer) -> None:
    assert encode_state("短い本文です。", tokenizer).truncated is False


def test_state_max_tokens_is_respected(tokenizer) -> None:
    encoded = encode_state("あ" * 500, tokenizer, max_tokens=32)
    assert encoded.n_tokens == 32
    assert encoded.truncated is True


def test_state_max_tokens_must_leave_room_for_the_special_tokens(tokenizer) -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        encode_state("あ", tokenizer, max_tokens=1)


def test_encode_states_batches(tokenizer) -> None:
    encoded = encode_states(["一件目", "", "三件目"], tokenizer)
    assert len(encoded) == 3
    assert all(e.n_tokens >= 2 for e in encoded)


def test_emoji_and_kana_survive_a_question_round_trip(tokenizer) -> None:
    question = ChoiceQuestion(
        instructions="どの区分か🙏",
        criteria={"ﾊﾝｶｸ": "半角カナの説明", "絵文字": "🎌の説明"},
    )
    encoded = encode_question(question, tokenizer)
    assert encoded.n_markers == 2
    for position in encoded.marker_positions:
        assert encoded.input_ids[position] == tokenizer.mask_token_id


# --------------------------------------------------------------------------
# over-long questions must fail loudly, never lose a marker
# --------------------------------------------------------------------------

def test_an_over_long_question_raises_rather_than_dropping_an_option(tokenizer) -> None:
    question = ChoiceQuestion(
        instructions="どれか",
        criteria={f"選択肢{i}": "非常に長い説明文。" * 40 for i in range(20)},
    )
    with pytest.raises(QuestionTooLongError, match="Markers are never truncated"):
        encode_question(question, tokenizer)


def test_question_within_the_budget_is_accepted(tokenizer) -> None:
    encoded = encode_question(CHOICE, tokenizer)
    assert len(encoded.input_ids) <= MAX_QUESTION_TOKENS


def test_typical_question_fits_the_256_token_expectation(tokenizer) -> None:
    """§5.2 expects the question side to stay inside ~256 tokens in normal use."""
    for question in ALL_QUESTIONS:
        assert len(encode_question(question, tokenizer).input_ids) <= 256


# --------------------------------------------------------------------------
# cache (§5.2: question-side encodings are reusable across requests)
# --------------------------------------------------------------------------

def test_cache_returns_an_identical_encoding_on_a_hit(tokenizer) -> None:
    cache = QuestionEncoderCache()
    first = cache.get(CHOICE, tokenizer)
    second = cache.get(CHOICE, tokenizer)
    assert first is second
    assert cache.hits == 1
    assert cache.misses == 1


def test_cache_distinguishes_different_questions(tokenizer) -> None:
    cache = QuestionEncoderCache()
    cache.get(CHOICE, tokenizer)
    cache.get(SCORE, tokenizer)
    cache.get(BOOL, tokenizer)
    assert len(cache) == 3
    assert cache.misses == 3


def test_cache_key_is_insensitive_to_how_the_question_was_built(tokenizer) -> None:
    """A dict from the wire and an equal model object must hit the same entry."""
    from_wire = parse_question({
        "type": "choice",
        "instructions": CHOICE.instructions,
        "criteria": dict(CHOICE.criteria),
    })
    assert question_cache_key(from_wire, tokenizer) == question_cache_key(CHOICE, tokenizer)


def test_cache_respects_its_bound(tokenizer) -> None:
    cache = QuestionEncoderCache(max_entries=2)
    for i in range(5):
        cache.get(ChoiceQuestion(instructions=f"質問{i}", criteria={"a": "A"}), tokenizer)
    assert len(cache) == 2
    assert cache.misses == 5


def test_cache_clear_resets_counters(tokenizer) -> None:
    cache = QuestionEncoderCache()
    cache.get(CHOICE, tokenizer)
    cache.clear()
    assert len(cache) == 0 and cache.hits == 0 and cache.misses == 0


def test_cache_rejects_a_useless_bound() -> None:
    with pytest.raises(ValueError, match="positive"):
        QuestionEncoderCache(max_entries=0)
