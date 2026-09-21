"""Question encoding. **The only place a question becomes token ids** (§1-3, §5.2).

    [CLS] {instructions} [SEP]
      {option_1_label}: {option_1_desc} <mask>
      {option_2_label}: {option_2_desc} <mask>
      ...
    [SEP]

Three decisions, all from §5.2, all load-bearing:

**The state is not in here.** That is the difference from v1. The question side stays
short (usually well under 256 tokens), so ModernBERT's 128-token local attention
windows and its every-third-layer global attention operate on it under the same
conditions they were pretrained with. The state reaches the question through the
cross-attention head instead (§6.2), which also means a state is encoded once per
request no matter how many questions there are.

**Markers reuse the pretrained `<mask>` token.** Adding a new special token would
force an embedding resize and throw away the MLM prior sitting on exactly the
position we are about to read. `tokenizer.mask_token_id` is read from the tokenizer;
it is never hardcoded.

**`marker_positions` is returned, not recomputed.** Training and inference reading
different positions would break everything silently, which is why the round-trip is
tested rather than assumed.

Encoded questions are cacheable: the result depends only on the question and the
tokenizer, so a repeated schema in production can reuse `H_q` (§5.2, §10).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from sokudan.encoding.special_tokens import SpecialTokenLayout, require_mask_token_id
from sokudan.schema.question import Question, is_ordered, marker_texts

MAX_QUESTION_TOKENS = 512


@dataclass(frozen=True)
class EncodedQuestion:
    input_ids: list[int]
    attention_mask: list[int]
    marker_positions: list[int]
    """Index into `input_ids` of each option's marker, in the question's own order."""
    n_markers: int
    ordered: bool
    truncated: bool
    cache_key: str

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("input_ids and attention_mask must be the same length")
        if len(self.marker_positions) != self.n_markers:
            raise ValueError(
                f"expected {self.n_markers} marker positions, "
                f"got {len(self.marker_positions)}"
            )
        for position in self.marker_positions:
            if not 0 <= position < len(self.input_ids):
                raise ValueError(f"marker position {position} is outside the sequence")


class QuestionTooLongError(ValueError):
    """The question and its options do not fit, so some markers would be lost.

    Truncating a question side is not a graceful degradation: a dropped marker means
    an option silently stops being an option. Callers must shorten descriptions or
    split the question (§5.1).
    """


def question_cache_key(question: Question, tokenizer: PreTrainedTokenizerBase) -> str:
    """Stable key over the question *and* the tokenizer that will encode it.

    **`sort_keys` must stay off for the question body.** `criteria` is an ordered
    mapping whose order is part of the schema: it decides which marker belongs to
    which option. Sorting it would give `{"a": …, "b": …}` and `{"b": …, "a": …}` the
    same key, so a shuffled schema -- which training produces deliberately (§7.2) --
    would silently reuse an encoding whose markers are in the other order, and every
    label would be off. The surrounding envelope is written in a fixed order instead,
    which gives determinism without touching the question's own ordering.
    """
    body = json.dumps(question.model_dump(mode="json"), ensure_ascii=False, sort_keys=False)
    envelope = [
        ("type", question.type),
        ("question", body),
        ("tokenizer", getattr(tokenizer, "name_or_path", type(tokenizer).__name__)),
        ("mask_token_id", tokenizer.mask_token_id),
    ]
    blob = json.dumps(envelope, ensure_ascii=False, sort_keys=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def encode_question(
    question: Question,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int = MAX_QUESTION_TOKENS,
) -> EncodedQuestion:
    """Encode a question and report where its markers landed.

    Built by concatenating token id runs rather than by tokenizing one big string and
    searching for mask tokens afterwards. Searching would be wrong the moment an
    option's own text contains the mask string, and it would be fragile against
    tokenizers that merge across the boundary.
    """
    mask_id = require_mask_token_id(tokenizer)
    layout = SpecialTokenLayout.from_tokenizer(tokenizer)

    texts = marker_texts(question)
    if not texts:
        raise ValueError(f"question type {question.type!r} produced no marker slots")

    def ids_of(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    # Two segments -- instructions, then the option block -- wrapped exactly the way
    # this tokenizer wraps a sentence pair, so the markers sit in a context the
    # backbone saw during pretraining (see `special_tokens` for why this is probed).
    input_ids: list[int] = [*layout.prefix, *ids_of(question.instructions), *layout.separator]
    marker_positions: list[int] = []
    for text in texts:
        input_ids.extend(ids_of(text))
        marker_positions.append(len(input_ids))
        input_ids.append(mask_id)
    input_ids.extend(layout.suffix)

    if len(input_ids) > max_tokens:
        raise QuestionTooLongError(
            f"question encodes to {len(input_ids)} tokens, over the {max_tokens} limit. "
            f"Shorten the option descriptions, or split a wide `choice` into a coarse "
            f"choice followed by a fine one (§5.1). Markers are never truncated away."
        )

    return EncodedQuestion(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        marker_positions=marker_positions,
        n_markers=len(texts),
        ordered=is_ordered(question),
        truncated=False,
        cache_key=question_cache_key(question, tokenizer),
    )


class QuestionEncoderCache:
    """Memoises encoded questions by `question_cache_key` (§5.2).

    Production sends the same schema over and over; re-encoding it every request is
    pure waste and shows up directly in latency. Deliberately a plain dict with an
    explicit bound rather than an LRU: schemas are few and long-lived, and an
    eviction policy that silently drops the hot schema would be worse than a clear
    error about the bound.
    """

    def __init__(self, max_entries: int = 4096) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: dict[str, EncodedQuestion] = {}
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    def get(
        self,
        question: Question,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_tokens: int = MAX_QUESTION_TOKENS,
    ) -> EncodedQuestion:
        key = question_cache_key(question, tokenizer)
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            return cached

        self.misses += 1
        encoded = encode_question(question, tokenizer, max_tokens=max_tokens)
        if len(self._entries) < self._max_entries:
            self._entries[key] = encoded
        return encoded

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0


# ---------------------------------------------------------------------------
# Joint encoding (s2c). The Laya arrangement, kept in this module on purpose.
# ---------------------------------------------------------------------------

MAX_JOINT_TOKENS = 1024


@dataclass(frozen=True)
class EncodedJoint:
    """One `(question, state)` pair as a single sequence.

    `separate` -- the mode above -- encodes a state once and reaches it from the
    question through cross-attention, so N questions cost one state encoding. `joint`
    puts them in one sequence, which means the state is re-encoded per question and
    the backbone's own attention does the mixing.

    §6.1 rejected joint encoding on a stated mechanism: `modernbert-ja-310m` has
    `local_attention: 128` with global attention every third layer, so a marker and a
    distant state token cannot see each other in two layers out of three, and the
    behaviour changes with where in the sequence things land. s2a then failed for an
    unrelated reason, so the mechanism is worth measuring rather than assuming. This
    is that measurement, not a change of position.

    The question comes first so the markers sit at low indices, within a local window
    of the start of the state. The state is truncated from the end if the pair does
    not fit; **markers are never truncated**, which is the same rule the separate
    encoder enforces.
    """

    input_ids: list[int]
    attention_mask: list[int]
    marker_positions: list[int]
    n_markers: int
    ordered: bool
    truncated: bool
    n_question_tokens: int
    n_state_tokens: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("input_ids and attention_mask must be the same length")
        if len(self.marker_positions) != self.n_markers:
            raise ValueError(
                f"expected {self.n_markers} marker positions, "
                f"got {len(self.marker_positions)}"
            )
        for position in self.marker_positions:
            if not 0 <= position < len(self.input_ids):
                raise ValueError(f"marker position {position} is outside the sequence")


def encode_joint(
    question: Question,
    state: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int = MAX_JOINT_TOKENS,
) -> EncodedJoint:
    """`[CLS] {instructions} [SEP] {options+markers} [SEP] {state} [SEP]`.

    Built from the same `marker_texts` and the same `SpecialTokenLayout` as
    `encode_question`, so the two modes cannot disagree about what a marker is or
    where the separators go. That is the §1-3 rule: one place a question becomes
    token ids, even when there are two arrangements of it.
    """
    mask_id = require_mask_token_id(tokenizer)
    layout = SpecialTokenLayout.from_tokenizer(tokenizer)

    texts = marker_texts(question)
    if not texts:
        raise ValueError(f"question type {question.type!r} produced no marker slots")

    def ids_of(text: str) -> list[int]:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])

    head: list[int] = [*layout.prefix, *ids_of(question.instructions), *layout.separator]
    marker_positions: list[int] = []
    for text in texts:
        head.extend(ids_of(text))
        marker_positions.append(len(head))
        head.append(mask_id)
    head.extend(layout.separator)

    room = max_tokens - len(head) - len(layout.suffix)
    if room < 1:
        raise QuestionTooLongError(
            f"the question alone encodes to {len(head)} tokens, leaving no room for a "
            f"state inside the {max_tokens} limit. Markers are never truncated away."
        )

    state_ids = ids_of(state)
    truncated = len(state_ids) > room
    input_ids = [*head, *state_ids[:room], *layout.suffix]

    return EncodedJoint(
        input_ids=input_ids,
        attention_mask=[1] * len(input_ids),
        marker_positions=marker_positions,
        n_markers=len(texts),
        ordered=is_ordered(question),
        truncated=truncated,
        n_question_tokens=len(head),
        n_state_tokens=min(len(state_ids), room),
    )
