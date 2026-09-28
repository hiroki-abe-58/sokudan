"""`device="auto"`: cuda, then mps, then cpu; an explicit device is used as given."""

from __future__ import annotations

import pytest
import torch

from sokudan.predict import resolve_device


def _available(monkeypatch, *, cuda: bool, mps: bool) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(("cuda", "mps", "expected"), [
    (True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"), (False, False, "cpu"),
])
def test_auto_prefers_cuda_then_mps_then_cpu(monkeypatch, cuda, mps, expected):
    _available(monkeypatch, cuda=cuda, mps=mps)
    assert resolve_device("auto") == expected
    assert resolve_device(None) == expected


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps", "cuda:1"])
def test_an_explicit_device_is_used_as_given(monkeypatch, device):
    _available(monkeypatch, cuda=False, mps=False)
    assert resolve_device(device) == device


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_auto_picks_mps_on_apple_silicon():
    # conftest hides CUDA, so on a Mac the first available device is mps.
    assert resolve_device() == "mps"


@pytest.mark.slow
@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs an MPS device")
def test_load_places_the_model_on_the_resolved_device():
    import sokudan

    try:
        auto = sokudan.load("GeneLab/sokudan-ja-310m")
    except Exception as exc:  # pragma: no cover - depends on the local cache / network
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")
    assert auto.device == "mps"
    assert next(auto.model.parameters()).device.type == "mps"
    cpu = sokudan.load("GeneLab/sokudan-ja-310m", device="cpu")
    assert cpu.device == "cpu"
    assert next(cpu.model.parameters()).device.type == "cpu"
