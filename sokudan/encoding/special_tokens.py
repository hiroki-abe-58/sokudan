"""Where the boundary tokens come from (SOKUDAN_SPEC.md §5.2).

§5.2 writes the layouts with BERT's `[CLS] … [SEP]` spelling. That is a description
of *roles*, not of literal token strings, and taking it literally is a trap on this
backbone:

    sbintuitions/modernbert-ja-310m
      tokenizer.cls_token_id -> 6   (<cls>)
      tokenizer.sep_token_id -> 4   (<sep>)
      tokenizer("あ")        -> [1, 1516, 2]        ['<s>', 'あ', '</s>']
      tokenizer("あ", "い")  -> [1, …, 2, 1, …, 2]  pairs repeat <s> … </s>

`<cls>` and `<sep>` are in the vocabulary but the post-processor never emits them, so
the model has effectively never seen a sequence shaped that way. Wrapping our markers
in them would put the `<mask>` positions we are about to read into a context that does
not occur in pretraining -- the same mistake §5.2 warns about when it forbids adding a
fresh special token for the marker.

So the layout is *probed* rather than assumed: encode sentinels, see what the
tokenizer actually wraps them with, and reuse exactly that. A different backbone with
a genuine `[CLS]`/`[SEP]` template gets its own convention for free.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

# Two short ASCII sentinels that every subword vocabulary tokenizes to at least one
# ordinary token, so the surrounding special tokens are unambiguous.
_SENTINEL_A = "a"
_SENTINEL_B = "b"


@dataclass(frozen=True)
class SpecialTokenLayout:
    """The special-token runs a tokenizer puts around one and two text segments."""

    prefix: list[int]
    """Emitted before the first segment (BERT: `[CLS]`; ModernBERT-Ja: `<s>`)."""
    separator: list[int]
    """Emitted between two segments (ModernBERT-Ja: `</s><s>`)."""
    suffix: list[int]
    """Emitted after the last segment (ModernBERT-Ja: `</s>`)."""

    @property
    def overhead_single(self) -> int:
        return len(self.prefix) + len(self.suffix)

    @property
    def overhead_pair(self) -> int:
        return len(self.prefix) + len(self.separator) + len(self.suffix)

    @classmethod
    def from_tokenizer(cls, tokenizer: PreTrainedTokenizerBase) -> SpecialTokenLayout:
        def ids(*texts: str, special: bool) -> list[int]:
            encoded = tokenizer(*texts, add_special_tokens=special)
            return list(encoded["input_ids"])

        bare_a = ids(_SENTINEL_A, special=False)
        bare_b = ids(_SENTINEL_B, special=False)
        single = ids(_SENTINEL_A, special=True)

        if not bare_a or not bare_b:
            raise ValueError("tokenizer produced no tokens for an ASCII sentinel")

        start = _find_sublist(single, bare_a)
        if start < 0:
            raise ValueError(
                "could not locate the sentinel inside the tokenizer's single-segment "
                "output; this tokenizer rewrites its input and cannot be probed"
            )
        prefix = single[:start]
        suffix = single[start + len(bare_a):]

        # Pair encoding is optional: some tokenizers refuse two segments. Falling back
        # to `suffix + prefix` reproduces the common `</s><s>` / `[SEP]` behaviour.
        try:
            pair = ids(_SENTINEL_A, _SENTINEL_B, special=True)
        except (TypeError, ValueError, NotImplementedError):
            return cls(prefix=prefix, separator=suffix + prefix, suffix=suffix)

        a_at = _find_sublist(pair, bare_a)
        b_at = _find_sublist(pair, bare_b, start=a_at + len(bare_a) if a_at >= 0 else 0)
        if a_at < 0 or b_at < 0:
            return cls(prefix=prefix, separator=suffix + prefix, suffix=suffix)

        return cls(
            prefix=pair[:a_at],
            separator=pair[a_at + len(bare_a):b_at],
            suffix=pair[b_at + len(bare_b):],
        )


def _find_sublist(haystack: list[int], needle: list[int], start: int = 0) -> int:
    """Index of the first occurrence of `needle` in `haystack`, or -1."""
    if not needle or len(needle) > len(haystack):
        return -1
    for i in range(start, len(haystack) - len(needle) + 1):
        if haystack[i:i + len(needle)] == needle:
            return i
    return -1


def require_mask_token_id(tokenizer: PreTrainedTokenizerBase) -> int:
    """The marker id, read from the tokenizer and never hardcoded (§5.2)."""
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError(
            f"tokenizer {getattr(tokenizer, 'name_or_path', '?')} has no mask token. "
            "§5.2 requires reusing the pretrained mask token as the option marker "
            "rather than adding a new special token and resizing the embedding."
        )
    return int(mask_id)
