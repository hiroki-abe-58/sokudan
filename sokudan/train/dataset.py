"""Dataset and collation for Stage 1 (SOKUDAN_SPEC.md §8).

Everything here goes through `sokudan.encoding`. That is the §1-3 rule and it is not
a stylistic preference: the moment training tokenises a question differently from
inference, marker positions stop matching and every reported number becomes fiction.
There is no tokenisation logic in this file, only batching of what the encoder
returned.

Batches are bucketed by state length before padding. Gate A could not build
flash-attn on this machine, so the unpadding path is unavailable and padding is real
work -- a batch mixing a 40-token state with a 900-token one pays for 900 on every
row (`docs/gate_a.md`).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from sokudan.encoding.question import MAX_JOINT_TOKENS, QuestionEncoderCache, encode_joint
from sokudan.encoding.state import encode_state
from sokudan.schema.question import is_ordered, parse_question


@dataclass
class Example:
    """One training example: a state, a schema, and the gold option index."""

    state: str
    question: Any          # a `Question`
    label: int
    ordered: bool
    doc_id: str
    domain: str
    attribute: str
    kind: str

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Example:
        question = parse_question(row["question"])
        return cls(
            state=row["state"],
            question=question,
            label=int(row["label"]),
            ordered=is_ordered(question),
            doc_id=row.get("doc_id", ""),
            domain=row.get("domain", ""),
            attribute=row.get("attribute", ""),
            kind=row.get("kind", question.type),
        )


def load_examples(path: str | Path) -> list[Example]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return [Example.from_row(row) for row in rows]


@dataclass
class Batch:
    state_input_ids: Tensor
    state_attention_mask: Tensor
    question_input_ids: Tensor
    question_attention_mask: Tensor
    marker_positions: Tensor
    marker_mask: Tensor
    labels: Tensor
    ordered: Tensor

    def to(self, device: torch.device | str) -> Batch:
        return Batch(**{
            name: value.to(device) for name, value in self.__dict__.items()
        })

    def __len__(self) -> int:
        return self.labels.shape[0]


class Collator:
    """Encodes a list of examples into padded tensors.

    The question-side encoder cache is shared across batches. Training reuses the
    same few hundred schema variants constantly, so this removes most of the
    tokenisation cost -- the same cache the serving path uses (§5.2).
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_state_tokens: int = 1024,
        cache: QuestionEncoderCache | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_state_tokens = max_state_tokens
        self.cache = cache if cache is not None else QuestionEncoderCache()
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer has no pad token; padding would be ambiguous")

    def __call__(self, examples: list[Example]) -> Batch:
        states = [
            encode_state(e.state, self.tokenizer, max_tokens=self.max_state_tokens)
            for e in examples
        ]
        questions = [self.cache.get(e.question, self.tokenizer) for e in examples]

        state_len = max(len(s.input_ids) for s in states)
        question_len = max(len(q.input_ids) for q in questions)
        n_markers = max(q.n_markers for q in questions)
        batch_size = len(examples)

        state_ids = torch.full((batch_size, state_len), self.pad_id, dtype=torch.long)
        state_mask = torch.zeros((batch_size, state_len), dtype=torch.long)
        question_ids = torch.full((batch_size, question_len), self.pad_id, dtype=torch.long)
        question_mask = torch.zeros((batch_size, question_len), dtype=torch.long)
        marker_positions = torch.zeros((batch_size, n_markers), dtype=torch.long)
        marker_mask = torch.zeros((batch_size, n_markers), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        ordered = torch.zeros(batch_size, dtype=torch.bool)

        for row, (example, state, question) in enumerate(zip(examples, states, questions,
                                                             strict=True)):
            state_ids[row, : len(state.input_ids)] = torch.tensor(state.input_ids)
            state_mask[row, : len(state.input_ids)] = 1
            question_ids[row, : len(question.input_ids)] = torch.tensor(question.input_ids)
            question_mask[row, : len(question.input_ids)] = 1

            positions = question.marker_positions
            marker_positions[row, : len(positions)] = torch.tensor(positions)
            marker_mask[row, : len(positions)] = 1
            # Padded marker slots must still index inside the sequence; the mask is
            # what excludes them. Pointing them at the first real marker is safe.
            marker_positions[row, len(positions):] = positions[0]

            if not 0 <= example.label < len(positions):
                raise ValueError(
                    f"label {example.label} is outside the {len(positions)} options of "
                    f"{example.domain}/{example.attribute}"
                )
            labels[row] = example.label
            ordered[row] = example.ordered

        return Batch(
            state_input_ids=state_ids,
            state_attention_mask=state_mask,
            question_input_ids=question_ids,
            question_attention_mask=question_mask,
            marker_positions=marker_positions,
            marker_mask=marker_mask,
            labels=labels,
            ordered=ordered,
        )


def length_bucketed_batches(
    examples: list[Example],
    batch_size: int,
    tokenizer: PreTrainedTokenizerBase,
    *,
    rng: random.Random,
    bucket_multiplier: int = 32,
    shuffle: bool = True,
) -> list[list[Example]]:
    """Group examples of similar state length, then shuffle the groups.

    Sorting globally would make every epoch see the same batches in the same order,
    which correlates batch composition with training step. Sorting inside a large
    window and shuffling the resulting batches keeps padding low while leaving the
    order stochastic.
    """
    order = list(range(len(examples)))
    if shuffle:
        rng.shuffle(order)

    # Character length is a good enough proxy for token length and far cheaper.
    window = batch_size * bucket_multiplier
    batches: list[list[Example]] = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start:start + window], key=lambda i: len(examples[i].state))
        for offset in range(0, len(chunk), batch_size):
            batches.append([examples[i] for i in chunk[offset:offset + batch_size]])

    if shuffle:
        rng.shuffle(batches)
    return batches


# ---------------------------------------------------------------------------
# Joint mode (s2c). Same `Example`, same batching, a different encoding.
# ---------------------------------------------------------------------------


@dataclass
class JointBatch:
    """One sequence per (question, state) pair, so no separate state tensors."""

    input_ids: Tensor
    attention_mask: Tensor
    marker_positions: Tensor
    marker_mask: Tensor
    labels: Tensor
    ordered: Tensor

    def to(self, device: torch.device | str) -> JointBatch:
        return JointBatch(**{
            name: value.to(device) for name, value in self.__dict__.items()
        })

    def __len__(self) -> int:
        return self.labels.shape[0]


class JointCollator:
    """Encodes examples with `encode_joint` (§6.1 measurement, s2c).

    **No question cache.** The separate arm can memoise `H_q` because the question
    encoding does not depend on the state; here it does, so every pair is unique and
    a cache would only waste memory. That difference is the point of the comparison
    and not an oversight.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_joint_tokens: int = MAX_JOINT_TOKENS,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_joint_tokens = max_joint_tokens
        self.pad_id = tokenizer.pad_token_id
        if self.pad_id is None:
            raise ValueError("tokenizer has no pad token; padding would be ambiguous")
        self.truncated = 0

    def __call__(self, examples: list[Example]) -> JointBatch:
        encoded = [
            encode_joint(e.question, e.state, self.tokenizer,
                         max_tokens=self.max_joint_tokens)
            for e in examples
        ]
        self.truncated += sum(1 for e in encoded if e.truncated)

        length = max(len(e.input_ids) for e in encoded)
        n_markers = max(e.n_markers for e in encoded)
        batch_size = len(examples)

        input_ids = torch.full((batch_size, length), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, length), dtype=torch.long)
        marker_positions = torch.zeros((batch_size, n_markers), dtype=torch.long)
        marker_mask = torch.zeros((batch_size, n_markers), dtype=torch.long)
        labels = torch.zeros(batch_size, dtype=torch.long)
        ordered = torch.zeros(batch_size, dtype=torch.bool)

        for row, (example, item) in enumerate(zip(examples, encoded, strict=True)):
            input_ids[row, : len(item.input_ids)] = torch.tensor(item.input_ids)
            attention_mask[row, : len(item.input_ids)] = 1

            positions = item.marker_positions
            marker_positions[row, : len(positions)] = torch.tensor(positions)
            marker_mask[row, : len(positions)] = 1
            marker_positions[row, len(positions):] = positions[0]

            if not 0 <= example.label < len(positions):
                raise ValueError(
                    f"label {example.label} is outside the {len(positions)} options of "
                    f"{example.domain}/{example.attribute}"
                )
            labels[row] = example.label
            ordered[row] = example.ordered

        return JointBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_positions=marker_positions,
            marker_mask=marker_mask,
            labels=labels,
            ordered=ordered,
        )
