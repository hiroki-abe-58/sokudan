"""Schema validation and the question-type registry (SOKUDAN_SPEC.md §5.1)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sokudan.schema.question import (
    BoolQuestion,
    ChoiceQuestion,
    ScoreQuestion,
    get_handler,
    is_ordered,
    marker_texts,
    n_markers,
    parse_question,
    parse_questions,
    register_question_type,
    registered_type_names,
)

# --------------------------------------------------------------------------
# parsing and aliases
# --------------------------------------------------------------------------

def test_parses_each_primitive() -> None:
    assert isinstance(
        parse_question({"type": "choice", "instructions": "どの部署か",
                        "criteria": {"請求": "支払い", "技術": "不具合"}}),
        ChoiceQuestion,
    )
    assert isinstance(
        parse_question({"type": "score", "instructions": "緊急度は",
                        "criteria": ["低", "中", "高"]}),
        ScoreQuestion,
    )
    assert isinstance(parse_question({"type": "bool", "instructions": "解約か"}), BoolQuestion)


def test_noul_is_an_alias_for_bool() -> None:
    """§5.1: the wire format may say `noul`; nothing downstream should have to know."""
    question = parse_question({"type": "noul", "instructions": "解約を示唆しているか"})
    assert isinstance(question, BoolQuestion)
    assert question.type == "bool"


def test_noul_and_bool_are_the_same_registered_handler() -> None:
    assert get_handler("noul") is get_handler("bool")


def test_unknown_type_names_the_known_ones() -> None:
    with pytest.raises(ValueError, match="unknown question type"):
        parse_question({"type": "sentiment", "instructions": "x"})


def test_missing_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="no 'type' field"):
        parse_question({"instructions": "x"})


def test_parse_questions_reports_which_question_failed() -> None:
    with pytest.raises(ValueError, match="urgency"):
        parse_questions({
            "department": {"type": "choice", "instructions": "a", "criteria": {"x": "y"}},
            "urgency": {"type": "score", "instructions": "b", "criteria": ["only-one"]},
        })


def test_blank_question_id_is_rejected() -> None:
    with pytest.raises(ValueError, match="question ids"):
        parse_questions({"  ": {"type": "bool", "instructions": "x"}})


def test_parse_question_passes_through_an_existing_model() -> None:
    question = BoolQuestion(instructions="解約か")
    assert parse_question(question) is question


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------

def test_blank_instructions_rejected() -> None:
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="   ", criteria={"a": "b"})


def test_score_needs_at_least_two_levels() -> None:
    """An ordinal scale with one level is not a scale."""
    with pytest.raises(ValidationError):
        ScoreQuestion(instructions="緊急度は", criteria=["唯一"])


def test_score_levels_must_be_distinct() -> None:
    with pytest.raises(ValidationError, match="distinct"):
        ScoreQuestion(instructions="緊急度は", criteria=["低", "低", "高"])


def test_score_levels_must_not_be_blank() -> None:
    with pytest.raises(ValidationError, match="blank"):
        ScoreQuestion(instructions="緊急度は", criteria=["低", "  ", "高"])


def test_bool_labels_must_differ() -> None:
    with pytest.raises(ValidationError, match="differ"):
        BoolQuestion(instructions="解約か", false_label="はい", true_label="はい")


def test_choice_rejects_a_blank_option_label() -> None:
    with pytest.raises(ValidationError, match="blank"):
        ChoiceQuestion(instructions="どれか", criteria={" ": "説明"})


def test_extra_fields_are_rejected() -> None:
    """A typo in a schema field should fail loudly, not be ignored."""
    with pytest.raises(ValidationError):
        ChoiceQuestion(instructions="x", criteria={"a": "b"}, critera={"typo": "!"})


def test_questions_are_frozen() -> None:
    """Questions are cache keys; a mutable one would corrupt the encoder cache."""
    question = BoolQuestion(instructions="解約か")
    with pytest.raises(ValidationError):
        question.instructions = "別の質問"


# --------------------------------------------------------------------------
# edge case from §5: a single-option choice
# --------------------------------------------------------------------------

def test_single_option_choice_is_accepted_and_has_one_marker() -> None:
    question = ChoiceQuestion(instructions="確認", criteria={"唯一": "常にこれ"})
    assert question.n_options == 1
    assert n_markers(question) == 1


def test_wide_choice_is_allowed() -> None:
    """20 is the recommended ceiling, not a validation rule (§5.1)."""
    criteria = {f"opt{i}": f"説明{i}" for i in range(25)}
    assert ChoiceQuestion(instructions="どれか", criteria=criteria).n_options == 25


# --------------------------------------------------------------------------
# marker texts and ordering
# --------------------------------------------------------------------------

def test_choice_marker_text_joins_label_and_description() -> None:
    question = ChoiceQuestion(instructions="x", criteria={"請求": "支払い・返金"})
    assert marker_texts(question) == ["請求: 支払い・返金"]


def test_choice_marker_text_omits_an_empty_description() -> None:
    question = ChoiceQuestion(instructions="x", criteria={"請求": ""})
    assert marker_texts(question) == ["請求"]


def test_score_marker_texts_keep_the_given_order() -> None:
    question = ScoreQuestion(instructions="x", criteria=["低", "中", "高"])
    assert marker_texts(question) == ["低", "中", "高"]


def test_bool_marker_texts_are_false_then_true() -> None:
    """Column order has to match `binary_to_probs`, which puts True in column 1."""
    assert marker_texts(BoolQuestion(instructions="x")) == ["いいえ", "はい"]


def test_only_score_is_ordered() -> None:
    assert is_ordered(ScoreQuestion(instructions="x", criteria=["a", "b"]))
    assert not is_ordered(ChoiceQuestion(instructions="x", criteria={"a": "b"}))
    assert not is_ordered(BoolQuestion(instructions="x"))


def test_choice_preserves_option_order_as_written() -> None:
    """Order carries no meaning, but it must round-trip: training shuffles it (§7.2)."""
    criteria = {"営業": "c", "請求": "a", "技術": "b"}
    assert ChoiceQuestion(instructions="x", criteria=criteria).labels == list(criteria)


# --------------------------------------------------------------------------
# open/closed: a new type must not require editing existing code (§2)
# --------------------------------------------------------------------------

def test_a_new_question_type_can_be_added_without_touching_existing_code() -> None:
    from typing import ClassVar, Literal

    from sokudan.schema.question import Question

    class RankQuestion(Question):
        type: Literal["rank_demo"] = "rank_demo"
        items: list[str]

    class _RankHandler:
        type_name: ClassVar[str] = "rank_demo"
        aliases: ClassVar[tuple[str, ...]] = ()
        model: ClassVar[type[Question]] = RankQuestion
        ordered: ClassVar[bool] = True

        @staticmethod
        def marker_texts(question: Question) -> list[str]:
            return list(question.items)

    register_question_type(_RankHandler())
    try:
        assert "rank_demo" in registered_type_names()
        question = parse_question(
            {"type": "rank_demo", "instructions": "並べよ", "items": ["a", "b", "c"]}
        )
        assert isinstance(question, RankQuestion)
        # The generic helpers work on it without knowing it exists.
        assert marker_texts(question) == ["a", "b", "c"]
        assert n_markers(question) == 3
        assert is_ordered(question)
    finally:
        from sokudan.schema.question import _REGISTRY

        _REGISTRY.pop("rank_demo", None)


def test_registering_a_duplicate_type_is_refused() -> None:
    class _Dupe:
        type_name = "choice"
        aliases = ()
        model = ChoiceQuestion
        ordered = False

        @staticmethod
        def marker_texts(question):  # pragma: no cover - never reached
            return []

    with pytest.raises(ValueError, match="already registered"):
        register_question_type(_Dupe())


def test_the_three_documented_types_plus_the_alias_are_registered() -> None:
    assert registered_type_names() == ["bool", "choice", "noul", "score"]
