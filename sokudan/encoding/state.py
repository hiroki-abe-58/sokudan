"""State encoding. **The only place a state becomes token ids** (SOKUDAN_SPEC.md §1-3).

    [CLS] {state} [SEP]

`train`, `eval` and `serve` all import this. If a second way of turning a state into
tensors appears anywhere, delete it and call this instead -- the moment the two drift,
every number the project reports becomes meaningless.

Nothing here hardcodes a special-token id. They come from the tokenizer, because the
backbone is a parameter and its vocabulary is not ours to assume.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

# ModernBERT-Ja's trained context. Longer states are truncated rather than silently
# wrapped or dropped, and the encoded result records that it happened.
MAX_STATE_TOKENS = 8192


@dataclass(frozen=True)
class EncodedState:
    input_ids: list[int]
    attention_mask: list[int]
    truncated: bool
    n_tokens: int

    def __post_init__(self) -> None:
        if len(self.input_ids) != len(self.attention_mask):
            raise ValueError("input_ids and attention_mask must be the same length")


def encode_state(
    state: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int = MAX_STATE_TOKENS,
) -> EncodedState:
    """Encode one state as `[CLS] {state} [SEP]`.

    An empty state is legal and encodes to just the two special tokens. A caller that
    considers an empty state an error should say so itself; this layer does not
    invent policy.
    """
    if max_tokens < 2:
        raise ValueError("max_tokens must leave room for [CLS] and [SEP]")

    encoded = tokenizer(
        state,
        add_special_tokens=True,
        truncation=True,
        max_length=max_tokens,
        return_attention_mask=True,
    )
    input_ids = list(encoded["input_ids"])

    # `truncation=True` is silent, so ask separately whether anything was lost.
    untruncated = len(tokenizer(state, add_special_tokens=True)["input_ids"])

    return EncodedState(
        input_ids=input_ids,
        attention_mask=list(encoded["attention_mask"]),
        truncated=untruncated > len(input_ids),
        n_tokens=len(input_ids),
    )


def encode_states(
    states: list[str],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int = MAX_STATE_TOKENS,
) -> list[EncodedState]:
    return [encode_state(state, tokenizer, max_tokens=max_tokens) for state in states]
