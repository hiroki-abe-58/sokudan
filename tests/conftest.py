"""Shared fixtures."""

from __future__ import annotations

import pytest

from sokudan.config import BACKBONE_MODEL_ID


@pytest.fixture(scope="session")
def tokenizer():
    """The real backbone tokenizer, from the local HF cache.

    Encoding is tested against the actual tokenizer rather than a stub. A stub would
    happily agree with a wrong assumption about special tokens -- which is exactly
    the bug this project would not survive (§1-3).
    """
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"backbone tokenizer unavailable: {type(exc).__name__}: {exc}")
