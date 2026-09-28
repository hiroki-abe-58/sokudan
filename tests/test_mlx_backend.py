"""The MLX backend against the torch model (tests/fixtures/mlx_parity.json).

Skipped where `mlx` is not installed. The torch side is recorded by
tests/fixtures/make_mlx_parity.py, so these run without torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from sokudan.backends.mlx.model import (  # noqa: E402
    OrdinalHead,
    convert_weights,
    masked_softmax,
)

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "mlx_parity.json")
                     .read_text(encoding="utf-8"))


def _sokudan_checkpoint_names(n_layers: int = 25) -> list[str]:
    """The 157 tensor names of a modernbert-ja-310m joint checkpoint."""
    names = ["backbone.model.embeddings.tok_embeddings.weight",
             "backbone.model.embeddings.norm.weight", "backbone.model.final_norm.weight",
             "scorer.weight", "ordinal.cut.weight", "ordinal.cut.bias",
             "ordinal.location.weight", "ordinal.location.bias"]
    for i in range(n_layers):
        names += [f"backbone.model.layers.{i}.{part}.weight"
                  for part in ("attn.Wqkv", "attn.Wo", "mlp_norm", "mlp.Wi", "mlp.Wo")]
        if i > 0:  # layer 0's attn_norm is an Identity
            names.append(f"backbone.model.layers.{i}.attn_norm.weight")
    return names


def test_the_weight_table_strips_the_backbone_prefix_and_keeps_the_heads():
    names = _sokudan_checkpoint_names()
    assert len(names) == 157
    converted = convert_weights(dict.fromkeys(names))
    assert len(converted) == 157
    assert sum(k.startswith("backbone.") for k in converted) == 152
    assert {k for k in converted if not k.startswith("backbone.")} == {
        "scorer.weight", "ordinal.cut.weight", "ordinal.cut.bias",
        "ordinal.location.weight", "ordinal.location.bias"}
    assert converted.keys() >= {"backbone.embeddings.tok_embeddings.weight",
                                "backbone.layers.24.mlp.Wo.weight",
                                "backbone.final_norm.weight"}


def test_the_table_matches_the_mlx_model_parameters_exactly():
    from mlx.utils import tree_flatten
    from transformers import AutoConfig

    from sokudan.backends.mlx.model import JointModel
    from sokudan.backends.mlx.modernbert import EncoderConfig
    from sokudan.config import BACKBONE_MODEL_ID

    try:
        encoder = AutoConfig.from_pretrained(BACKBONE_MODEL_ID).to_dict()
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"backbone config unavailable: {type(exc).__name__}: {exc}")
    model = JointModel(EncoderConfig.from_dict(encoder))
    parameters = {name for name, _ in tree_flatten(model.parameters())}
    assert set(convert_weights(dict.fromkeys(_sokudan_checkpoint_names()))) == parameters


def test_the_published_checkpoint_has_exactly_those_tensors():
    from safetensors import safe_open

    from sokudan.predict import locate_checkpoint

    try:
        path = locate_checkpoint(FIXTURE["predict"]["model"])
    except Exception as exc:  # pragma: no cover - depends on the local cache / network
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")
    with safe_open(str(path / "model.safetensors"), framework="numpy") as f:
        assert sorted(f.keys()) == sorted(_sokudan_checkpoint_names())


def test_masked_softmax_matches_torch():
    heads = FIXTURE["heads"]
    got = masked_softmax(mx.array(heads["logits"]), mx.array(heads["mask"]))
    np.testing.assert_allclose(np.array(got), heads["masked_softmax"], atol=1e-6)


def test_the_ordinal_head_matches_torch():
    heads = FIXTURE["heads"]
    head = OrdinalHead(8)
    head.load_weights([(f"{name}.{part}", mx.array(heads[name][part]))
                       for name in ("cut", "location") for part in ("weight", "bias")])
    got = head(mx.array(heads["states"]), mx.array(heads["mask"]))
    np.testing.assert_allclose(np.array(got), heads["ordinal"], atol=1e-6)


def _same_shape(a, b, path="") -> None:
    """Same keys, types and strings; numbers within 2e-3 (the outputs are rounded)."""
    assert type(a) is type(b), path
    if isinstance(a, dict):
        assert list(a) == list(b), path
        for key in a:
            _same_shape(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, list):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _same_shape(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert abs(a - b) <= 2e-3, path
    else:
        assert a == b, path


def _load(**options):
    import sokudan

    try:
        return sokudan.load(FIXTURE["predict"]["model"], backend="mlx", **options)
    except Exception as exc:  # pragma: no cover - depends on the local cache / network
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")


def _raw(agent, questions, state):
    from sokudan.predict import _Prepared
    from sokudan.schema.question import parse_questions

    prepared = {qid: _Prepared(q, q.type, list(q.labels))
                for qid, q in parse_questions(questions).items()}
    return agent._forward(state, prepared)[0]


@pytest.fixture(scope="module")
def mlx_agent():
    """float32: the precision the correctness gate compares with torch."""
    return _load(dtype="float32")


@pytest.mark.slow
def test_predict_answers_in_the_torch_shape(mlx_agent):
    recorded = FIXTURE["predict"]
    assert mlx_agent.backend.name == "mlx"
    for state, expected in zip(recorded["states"], recorded["outputs"], strict=True):
        _same_shape(mlx_agent.predict(state, recorded["questions"]), expected)


@pytest.mark.slow
def test_raw_probabilities_match_torch_cpu(mlx_agent):
    recorded = FIXTURE["predict"]
    for state, expected in zip(recorded["states"], recorded["raw"], strict=True):
        rows = _raw(mlx_agent, recorded["questions"], state)
        for qid, probs in expected.items():
            assert int(np.argmax(rows[qid])) == int(np.argmax(probs))
            np.testing.assert_allclose(rows[qid], probs, atol=1e-3)


@pytest.mark.slow
def test_the_default_is_float16_within_the_dtype_gate():
    agent = _load()
    assert (agent.backend.name, agent.dtype) == ("mlx", "float16")
    recorded = FIXTURE["predict"]
    for state, expected in zip(recorded["states"], recorded["raw"], strict=True):
        rows = _raw(agent, recorded["questions"], state)
        for qid, probs in expected.items():
            assert int(np.argmax(rows[qid])) == int(np.argmax(probs))
            np.testing.assert_allclose(rows[qid], probs, atol=5e-2)


def test_quantized_and_unknown_dtypes_are_refused(tmp_path):
    from sokudan.backends.mlx import MLXBackend

    (tmp_path / "model.safetensors").write_bytes(b"")
    for dtype in ("8bit", "4bit", "bfloat16"):
        with pytest.raises(ValueError, match="dtype"):
            MLXBackend.load(tmp_path, dtype=dtype)
