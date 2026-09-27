"""The `/v1/systemone` wire format, and how it maps onto `Agent.predict`.

The shape is read from TypeSafe's public API reference (docs.typesafe.ai/api) and
checked against the request/response examples published in the Lev, kev and Laya
READMEs. `docs/systemone_wire_format.md` has the table of what was read, what sokudan
does with it, and where those sources disagree. No TypeSafe code is used or imported.

This module is pure: no FastAPI, no torch. It turns a wire request into the dict that
`Agent.predict` already accepts, and turns `Agent.predict`'s result into wire answers.
`predict` itself is not changed -- the server adds translation, not semantics.

Three mappings carry a decision that the wire format leaves open:

**Structured text.** `instructions` and every criteria entry may be a string, an
object or an array. sokudan reads text, so an object is rendered as `key: value`
lines and an array as one line per item; nested values are written as JSON. `state`
is *not* rendered here -- it reaches `predict` unchanged, and `predict` renders a
mapping as `key: value` lines, which is not the input the model was trained on
(README, "pass the state as a string"). Every response says which path was taken.

**noul criteria.** sokudan's `bool` question has two marker texts (`いいえ` / `はい`).
A `criteria.true` / `criteria.false` description is appended to the matching one
(`はい: <description>`), the same way a choice option's description is rendered.
The effect on accuracy has **not been measured**.

**confidence.** `predict` reports the top probability. The wire format's
`confidence` is a different statistic: for a choice, `(p_max - 1/K) / (1 - 1/K)`
(the formula in the confidence explorer on docs.typesafe.ai/confidence, and the one
kev's and Laya's READMEs attribute to Jev); for a score, `max(0, 1 - E|level - mode| / D)`
as kev's README documents it. Both are computed here from the returned
probabilities, so a threshold written against the wire definition means the same
thing against this server.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10
MAX_QUESTIONS = int(os.environ.get("SOKUDAN_MAX_QUESTIONS", "64"))

# The two marker texts sokudan's `bool` question uses by default
# (`sokudan.schema.question.BoolQuestion`).
NOUL_FALSE_LABEL = "いいえ"
NOUL_TRUE_LABEL = "はい"

Entry = str | dict[str, Any] | list[Any]
"""A text-bearing field on the wire: a string, a JSON object, or a JSON array."""


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


class _WireQuestion(BaseModel):
    # Unknown keys are ignored rather than rejected: clients add their own (Laya's
    # `labels`, for one), and refusing them would break a client for no benefit.
    model_config = ConfigDict(extra="ignore")

    instructions: Entry

    @field_validator("instructions")
    @classmethod
    def _instructions_have_text(cls, value: Entry) -> Entry:
        if not render_entry(value).strip():
            raise ValueError("instructions must not be empty")
        return value


class NoulCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    true: Entry | None = None
    false: Entry | None = None


class NoulQuestion(_WireQuestion):
    # `bool` is sokudan's own name for the same type; accepted so that requests
    # written for sokudan's Python API work unchanged.
    type: Literal["noul", "bool"]
    criteria: NoulCriteria | None = None


class ChoiceQuestion(_WireQuestion):
    type: Literal["choice"]
    criteria: dict[str, Entry | None]

    @field_validator("criteria")
    @classmethod
    def _option_count(cls, value: dict[str, Entry | None]) -> dict[str, Entry | None]:
        if not value:
            raise ValueError("a choice question needs at least one option in criteria")
        if len(value) > MAX_CHOICE_OPTIONS:
            raise ValueError(
                f"at most {MAX_CHOICE_OPTIONS} options per choice question, got {len(value)}"
            )
        for label in value:
            if not label.strip():
                raise ValueError("option names must not be blank")
        return value


class ScoreQuestion(_WireQuestion):
    type: Literal["score"]
    criteria: list[Entry | None]

    @field_validator("criteria")
    @classmethod
    def _levels(cls, value: list[Entry | None]) -> list[Entry | None]:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"a score question needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, "
                f"got {len(value)}"
            )
        for index, level in enumerate(value):
            # sokudan reads a level through its text; an undescribed level has
            # nothing to read. Rejected rather than scored as a blank slot.
            if level is None or not render_entry(level).strip():
                raise ValueError(f"level {index} has no description; every level needs one")
        return value


WireQuestion = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion, Field(discriminator="type")
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    # Required but nullable on the wire (the SDK types allow null). sokudan needs
    # text to read, so null is rejected in the validator below with a reason.
    state: Entry | None
    questions: dict[str, WireQuestion]
    # Required by TypeSafe's reference, omitted in Laya's example. Accepted and
    # ignored: the server answers with the model it loaded and says which.
    model: str | None = None

    @field_validator("state")
    @classmethod
    def _state_present(cls, value: Entry | None) -> Entry:
        if value is None:
            raise ValueError("state is null; sokudan needs text (a string, object or array)")
        return value

    @field_validator("questions")
    @classmethod
    def _question_budget(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not value:
            raise ValueError("questions is empty; send at least one question")
        if len(value) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per request, got {len(value)}")
        for key in value:
            if not key.strip():
                raise ValueError("question ids must not be blank")
        return value


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _inline(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def render_entry(value: Entry | None) -> str:
    """Text for a structured `instructions` / criteria entry. None renders empty."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"{key}: {_inline(item)}" for key, item in value.items())
    if isinstance(value, list):
        return "\n".join(_inline(item) for item in value)
    return _inline(value)


def state_format(state: Entry) -> str:
    """How `predict` will read this state. Reported in every response."""
    if isinstance(state, str):
        return "text"
    if isinstance(state, dict):
        return "object_as_key_value_lines"
    return "array_as_lines"


def to_sokudan_question(question: NoulQuestion | ChoiceQuestion | ScoreQuestion) -> dict:
    """The dict `Agent.predict` accepts for one wire question."""
    instructions = render_entry(question.instructions)
    if isinstance(question, ChoiceQuestion):
        return {
            "type": "choice",
            "instructions": instructions,
            # Order is kept exactly as sent: it decides which marker is which
            # option. A null description becomes "", which renders the bare label.
            "criteria": {label: render_entry(description)
                         for label, description in question.criteria.items()},
        }
    if isinstance(question, ScoreQuestion):
        return {
            "type": "score",
            "instructions": instructions,
            "criteria": [render_entry(level) for level in question.criteria],
        }
    payload: dict[str, Any] = {"type": "bool", "instructions": instructions}
    if question.criteria is not None:
        if question.criteria.false is not None:
            payload["false_label"] = f"{NOUL_FALSE_LABEL}: {render_entry(question.criteria.false)}"
        if question.criteria.true is not None:
            payload["true_label"] = f"{NOUL_TRUE_LABEL}: {render_entry(question.criteria.true)}"
    return payload


def to_sokudan_questions(request: SystemOneRequest) -> dict[str, dict]:
    return {key: to_sokudan_question(question) for key, question in request.questions.items()}


# ---------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------


def choice_confidence(probabilities: list[float]) -> float:
    """`(p_max - 1/K) / (1 - 1/K)`, clipped to [0, 1]. A single option is 1."""
    k = len(probabilities)
    if k <= 1:
        return 1.0
    value = (k * max(probabilities) - 1) / (k - 1)
    return round(min(1.0, max(0.0, value)), 4)


def score_confidence(probabilities: list[float]) -> float:
    """`max(0, 1 - E|level - mode| / D)`.

    `mode` is the most likely level and `D` the mean distance of a uniform
    distribution over the levels from its middle (2/3 for three levels): all mass on
    one level gives 1, a uniform or wider spread gives 0.
    """
    k = len(probabilities)
    if k <= 1:
        return 1.0
    mode = max(range(k), key=lambda i: probabilities[i])
    spread = sum(p * abs(i - mode) for i, p in enumerate(probabilities))
    middle = (k - 1) / 2
    uniform = sum(abs(i - middle) for i in range(k)) / k
    return round(min(1.0, max(0.0, 1 - spread / uniform)), 4)


def to_wire_answer(question: NoulQuestion | ChoiceQuestion | ScoreQuestion,
                   answer: dict[str, Any]) -> dict[str, Any]:
    """One `predict` answer in the wire shape, field for field."""
    if isinstance(question, NoulQuestion):
        return {"type": "noul", "noul": answer["noul"]}
    if isinstance(question, ChoiceQuestion):
        probabilities = {label: answer["probabilities"][label] for label in question.criteria}
        return {
            "type": "choice",
            "choice": answer["choice"],
            "probabilities": probabilities,
            "confidence": choice_confidence(list(probabilities.values())),
        }
    k = len(question.criteria)
    probabilities = {str(i): answer["probabilities"][str(i)] for i in range(k)}
    return {
        "type": "score",
        "score": round(float(answer["score"]), 4),
        # The legend echoes each level as the client sent it, object or string.
        "legend": {str(i): level for i, level in enumerate(question.criteria)},
        "probabilities": probabilities,
        "confidence": score_confidence(list(probabilities.values())),
    }


def input_tokens(usage: dict[str, Any], encoding: str | None) -> int:
    """Tokens the backbone read for the request, from `predict`'s usage block.

    With joint encoding (v0.1 / v0.2) the state is encoded once per question, so it
    is counted once per backbone pass; with separate encoding, once.
    """
    state = int(usage.get("state_tokens", 0))
    questions = int(usage.get("question_tokens", 0))
    state_passes = int(usage.get("backbone_passes", 1)) if encoding == "joint" else 1
    return questions + state * state_passes


def to_wire_response(request: SystemOneRequest, result: dict[str, Any], *,
                     model_name: str, extension: dict[str, Any]) -> dict[str, Any]:
    usage = result.get("usage", {})
    return {
        "model": model_name,
        "answers": {
            key: to_wire_answer(question, result["answers"][key])
            for key, question in request.questions.items()
        },
        "usage": {"input_tokens": input_tokens(usage, result.get("encoding")),
                  "output_tokens": 0},
        # Everything sokudan-specific lives under one key, so `usage` stays exactly
        # the two fields the wire format defines.
        "sokudan": {
            "state_format": state_format(request.state),
            "state_tokens": usage.get("state_tokens"),
            "state_truncated": usage.get("state_truncated"),
            "backbone_passes": usage.get("backbone_passes"),
            **extension,
        },
    }
