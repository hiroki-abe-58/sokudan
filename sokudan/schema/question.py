"""Question primitives and their registry (SOKUDAN_SPEC.md §5.1).

Three primitives ship today: `choice`, `score`, `bool` (alias `noul`). Adding a
fourth must not require editing any of the code here or in `encoding` -- that is the
open/closed requirement in §2. A new type registers a handler and everything
downstream (encoding, the head, the metrics) reads it through `QuestionHandler`.

The handler, not the model class, is the extension point. It answers three questions
the rest of the system asks:

* how many marker slots does this question need, and what text precedes each one
* are those slots ordered (a `score`) or exchangeable (a `choice`)
* what does a distribution over those slots mean as an answer
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Expected range for `choice`. Not a hard limit -- see `ChoiceQuestion` -- because the
# right answer above it is a different schema design, not a validation error.
CHOICE_OPTIONS_TYPICAL_MAX = 20


class Question(BaseModel):
    """Base for every question type."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    instructions: str = Field(min_length=1)

    @field_validator("instructions")
    @classmethod
    def _instructions_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("instructions must not be blank")
        return value


class ChoiceQuestion(Question):
    """Mutually exclusive options.

    `criteria` maps an option label to a description of when it applies. Order is
    part of the schema as written, but it carries no meaning: the model must not
    read anything into an option's position. Training randomises the order for
    exactly this reason (§7.2), and `docs/baseline_ja.md` §6.2 shows what happens to
    a model that fails it -- `laya-multilingual` never selects the first slot of a
    `score` question, across five schema variants.

    2-20 options is the expected range. Above roughly 20, options start competing
    for the question side's token budget and each label gets too few tokens to be
    read properly. The fix is not a bigger budget: split the decision into a coarse
    `choice` followed by a fine `choice` within the chosen group.
    """

    type: Literal["choice"] = "choice"
    criteria: dict[str, str] = Field(min_length=1)

    @field_validator("criteria")
    @classmethod
    def _labels_are_usable(cls, value: dict[str, str]) -> dict[str, str]:
        for label in value:
            if not label.strip():
                raise ValueError("option labels must not be blank")
        return value

    @property
    def labels(self) -> list[str]:
        return list(self.criteria)

    @property
    def n_options(self) -> int:
        return len(self.criteria)


class ScoreQuestion(Question):
    """An ordinal scale. **Not** a multiclass question that happens to be sorted.

    `criteria` is the ordered list of level names, lowest first. The ordering is
    load-bearing: `sokudan` parameterises this with a cumulative link so that the
    predicted CDF is monotone by construction, and scores it with RPS, which
    punishes a two-level miss more than a one-level miss (§6.3).

    K is fixed per request, not per model, so anything downstream that hardcodes a
    number of levels is a bug.
    """

    type: Literal["score"] = "score"
    criteria: list[str] = Field(min_length=2)

    @field_validator("criteria")
    @classmethod
    def _levels_are_usable(cls, value: list[str]) -> list[str]:
        if any(not level.strip() for level in value):
            raise ValueError("level names must not be blank")
        if len(set(value)) != len(value):
            raise ValueError("level names must be distinct")
        return value

    @property
    def labels(self) -> list[str]:
        return list(self.criteria)

    @property
    def n_levels(self) -> int:
        return len(self.criteria)


class BoolQuestion(Question):
    """A calibrated P(true). Accepts `noul` as a type alias for wire compatibility.

    Internally this is a two-slot question so that it shares the marker/scorer path
    with `choice` rather than having a second implementation. The answer reports
    P(true) alone, which is what the wire format expects.
    """

    type: Literal["bool"] = "bool"
    # Rendered before the two markers. Kept as data, not literals in the encoder, so
    # that schema randomisation can swap them without touching encoding code.
    false_label: str = "いいえ"
    true_label: str = "はい"

    @model_validator(mode="after")
    def _labels_differ(self) -> BoolQuestion:
        if self.false_label.strip() == self.true_label.strip():
            raise ValueError("false_label and true_label must differ")
        if not self.false_label.strip() or not self.true_label.strip():
            raise ValueError("bool labels must not be blank")
        return self

    @property
    def labels(self) -> list[str]:
        return [self.false_label, self.true_label]


AnyQuestion = Annotated[
    ChoiceQuestion | ScoreQuestion | BoolQuestion,
    Field(discriminator="type"),
]


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@runtime_checkable
class QuestionHandler(Protocol):
    """What the rest of the system needs to know about a question type."""

    type_name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]]
    model: ClassVar[type[Question]]
    ordered: ClassVar[bool]
    """True when the marker slots form an ordinal scale (§6.3 applies)."""

    @staticmethod
    def marker_texts(question: Question) -> list[str]:
        """Text rendered immediately before each marker, one entry per slot."""
        ...


class _ChoiceHandler:
    type_name: ClassVar[str] = "choice"
    aliases: ClassVar[tuple[str, ...]] = ()
    model: ClassVar[type[Question]] = ChoiceQuestion
    ordered: ClassVar[bool] = False

    @staticmethod
    def marker_texts(question: Question) -> list[str]:
        assert isinstance(question, ChoiceQuestion)
        return [
            f"{label}: {description}" if description.strip() else label
            for label, description in question.criteria.items()
        ]


class _ScoreHandler:
    type_name: ClassVar[str] = "score"
    aliases: ClassVar[tuple[str, ...]] = ()
    model: ClassVar[type[Question]] = ScoreQuestion
    ordered: ClassVar[bool] = True

    @staticmethod
    def marker_texts(question: Question) -> list[str]:
        assert isinstance(question, ScoreQuestion)
        return list(question.criteria)


class _BoolHandler:
    type_name: ClassVar[str] = "bool"
    aliases: ClassVar[tuple[str, ...]] = ("noul",)
    model: ClassVar[type[Question]] = BoolQuestion
    ordered: ClassVar[bool] = False

    @staticmethod
    def marker_texts(question: Question) -> list[str]:
        assert isinstance(question, BoolQuestion)
        return [question.false_label, question.true_label]


_REGISTRY: dict[str, QuestionHandler] = {}


def register_question_type(handler: QuestionHandler) -> QuestionHandler:
    """Register a handler under its type name and every alias."""
    for name in (handler.type_name, *handler.aliases):
        if name in _REGISTRY:
            raise ValueError(f"question type {name!r} is already registered")
        _REGISTRY[name] = handler
    return handler


for _handler in (_ChoiceHandler(), _ScoreHandler(), _BoolHandler()):
    register_question_type(_handler)


def get_handler(type_name: str) -> QuestionHandler:
    try:
        return _REGISTRY[type_name]
    except KeyError:
        known = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"unknown question type {type_name!r}; known types: {known}") from None


def registered_type_names() -> list[str]:
    return sorted(_REGISTRY)


def parse_question(payload: dict[str, Any] | Question) -> Question:
    """Build a question from a wire dict, resolving type aliases.

    A single place where `noul` becomes `bool`, so nothing downstream has to know
    the alias exists.
    """
    if isinstance(payload, Question):
        return payload
    if "type" not in payload:
        raise ValueError("question payload has no 'type' field")

    data = dict(payload)
    handler = get_handler(str(data["type"]))
    data["type"] = handler.type_name
    return handler.model.model_validate(data)


def parse_questions(payload: dict[str, dict[str, Any] | Question]) -> dict[str, Question]:
    """Parse a whole `{question_id: question}` mapping."""
    parsed: dict[str, Question] = {}
    for question_id, raw in payload.items():
        if not str(question_id).strip():
            raise ValueError("question ids must not be blank")
        try:
            parsed[question_id] = parse_question(raw)
        except ValueError as exc:
            raise ValueError(f"question {question_id!r}: {exc}") from exc
    return parsed


def marker_texts(question: Question) -> list[str]:
    """Marker slot texts for any registered question type."""
    return get_handler(question.type).marker_texts(question)


def n_markers(question: Question) -> int:
    return len(marker_texts(question))


def is_ordered(question: Question) -> bool:
    return get_handler(question.type).ordered
