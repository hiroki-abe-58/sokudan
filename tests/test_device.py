"""`device="auto"`: cuda, then mps, then cpu; an explicit device is used as given."""

from __future__ import annotations

import pytest
import torch

from sokudan.backends.torch_backend import resolve_device


def _available(monkeypatch, *, cuda: bool, mps: bool, cuda_count: int | None = None) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    count = (1 if cuda else 0) if cuda_count is None else cuda_count
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)


@pytest.mark.parametrize(("mps", "expected"), [(True, "mps"), (False, "cpu")])
def test_cuda_available_with_no_device_is_not_chosen(monkeypatch, mps, expected):
    # Measured on Windows with CUDA_VISIBLE_DEVICES="": is_available() True, device_count() 0
    _available(monkeypatch, cuda=True, mps=mps, cuda_count=0)
    assert resolve_device("auto") == expected


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

    # backend="torch": with mlx installed, backend="auto" chooses MLX before any torch device
    try:
        auto = sokudan.load("GeneLab/sokudan-ja-310m", backend="torch")
    except Exception as exc:  # pragma: no cover - depends on the local cache / network
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")
    assert auto.backend.name == "torch" and auto.device == "mps"
    assert next(auto.model.parameters()).device.type == "mps"
    cpu = sokudan.load("GeneLab/sokudan-ja-310m", device="cpu")
    assert cpu.device == "cpu"
    assert next(cpu.model.parameters()).device.type == "cpu"
