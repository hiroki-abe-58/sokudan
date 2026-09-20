"""Backbone smoke test (SOKUDAN_SPEC.md §4.1): Japanese fill-mask must work.

Marked `network` + `slow`: it downloads ~600MB on first run.
    uv run pytest tests/test_backbone_smoke.py -m "network" -v
"""

from __future__ import annotations

import pytest

from sokudan.config import BACKBONE_MODEL_ID


@pytest.mark.network
@pytest.mark.slow
def test_japanese_fill_mask_produces_japanese() -> None:
    from transformers import pipeline

    fill = pipeline("fill-mask", model=BACKBONE_MODEL_ID)
    mask = fill.tokenizer.mask_token
    results = fill(f"日本の首都は{mask}です。", top_k=5)

    tokens = [r["token_str"].strip() for r in results]
    assert results, "fill-mask returned nothing"
    assert any(any("぀" <= c <= "ヿ" or "一" <= c <= "鿿" for c in t)
               for t in tokens), f"no Japanese in top-5: {tokens}"


@pytest.mark.network
@pytest.mark.slow
def test_mask_token_id_is_not_hardcoded() -> None:
    """§5.2: markers reuse the pretrained <mask> token; read it from the tokenizer."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    assert tok.mask_token_id is not None
    assert tok.mask_token is not None
