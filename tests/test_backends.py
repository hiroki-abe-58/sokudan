"""The backend boundary: the padded numpy batch, and `import sokudan` without torch."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from sokudan.backends import Backend, Batch


@dataclass
class _Encoded:
    input_ids: list[int]
    marker_positions: list[int]
    marker_positions_back: list[int] | None = None

    @property
    def n_markers(self) -> int:
        return len(self.marker_positions)


def test_rows_are_padded_and_padded_markers_repeat_the_first_position():
    batch = Batch.build([_Encoded([1, 7, 5, 5, 2], [2, 3]),
                         _Encoded([1, 5, 5, 5], [1, 2, 3])], [False, True], pad_id=3)
    assert batch.input_ids.tolist() == [[1, 7, 5, 5, 2], [1, 5, 5, 5, 3]]
    assert batch.attention_mask.tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 0]]
    assert batch.marker_positions.tolist() == [[2, 3, 2], [1, 2, 3]]
    assert batch.marker_mask.tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch.ordered.tolist() == [False, True]
    assert batch.input_ids.dtype == np.int64 and batch.ordered.dtype == bool
    assert batch.marker_positions_back is None and batch.state_ids is None


def test_sandwich_back_positions_pad_the_same_way():
    batch = Batch.build([_Encoded([1, 5, 2, 5, 2], [1], [3]),
                         _Encoded([1, 5, 5, 2, 5, 5, 2], [1, 2], [4, 5])],
                        [False, False], pad_id=3, sandwich=True)
    assert batch.marker_positions_back.tolist() == [[3, 3], [4, 5]]


class _FakeBackend(Backend):
    """Uniform probabilities, or NaN when `broken`."""

    encoding = "joint"
    input_order = "question_first"
    backbone_id = "sbintuitions/modernbert-ja-310m"
    dtype = "float32"

    def __init__(self, name: str, device: str, *, broken: bool = False) -> None:
        self.name, self.device, self.broken, self.model = name, device, broken, None

    def probs(self, batch: Batch) -> np.ndarray:
        mask = batch.marker_mask.astype(np.float32)
        out = mask / mask.sum(axis=1, keepdims=True)
        return out * np.nan if self.broken else out


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


@pytest.fixture
def fake_loads(monkeypatch, tokenizer):
    """`_load_backend` builds fakes; `failing` names the backends (`"torch"`) or backend
    and device (`"torch/mps"`) that raise or return NaN."""
    import sokudan.predict as predict

    calls: list[tuple[str, str]] = []
    failing: dict[str, str] = {}

    def load_backend(name, path, device, dtype):
        calls.append((name, device))
        how = failing.get(f"{name}/{device}", failing.get(name))
        if how == "raise":
            raise RuntimeError(f"{name} is broken")
        return _FakeBackend(name, device, broken=how == "nan")

    monkeypatch.setattr(predict, "_load_backend", load_backend)
    monkeypatch.setattr(predict, "is_apple_silicon", lambda: True)
    monkeypatch.setattr(predict, "_mlx_importable", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return calls, failing


def test_auto_takes_mlx_first_on_apple_silicon(fake_loads, checkpoint):
    import sokudan

    calls, _ = fake_loads
    agent = sokudan.load(checkpoint)
    assert agent.backend.name == "mlx" and calls == [("mlx", "gpu")]


@pytest.mark.parametrize("how", ["raise", "nan"])
def test_auto_warns_and_falls_through_when_a_backend_fails(fake_loads, checkpoint, how):
    import sokudan

    calls, failing = fake_loads
    failing["mlx"] = how
    with pytest.warns(RuntimeWarning, match="mlx"):
        agent = sokudan.load(checkpoint)
    assert (agent.backend.name, agent.device) == ("torch", "mps")
    assert calls == [("mlx", "gpu"), ("torch", "mps")]


def test_auto_order_without_mlx_is_mps_then_cpu(fake_loads, checkpoint, monkeypatch):
    import sokudan
    import sokudan.predict as predict

    calls, failing = fake_loads
    monkeypatch.setattr(predict, "_mlx_importable", lambda: False)
    failing["torch"] = "raise"
    with pytest.warns(RuntimeWarning), pytest.raises(RuntimeError, match="no backend"):
        sokudan.load(checkpoint)
    assert calls == [("torch", "mps"), ("torch", "cpu")]


def test_a_failed_mlx_self_check_and_a_failed_mps_end_on_the_cpu(fake_loads, checkpoint):
    import sokudan

    calls, failing = fake_loads
    failing["mlx"] = "nan"
    failing["torch/mps"] = "raise"
    with pytest.warns(RuntimeWarning) as warned:
        agent = sokudan.load(checkpoint)
    assert (agent.backend.name, agent.device) == ("torch", "cpu")
    assert calls == [("mlx", "gpu"), ("torch", "mps"), ("torch", "cpu")]
    assert [("mlx" in str(w.message), "mps" in str(w.message)) for w in warned] == [
        (True, False), (False, True)]


def test_without_mps_a_failed_mlx_goes_straight_to_the_cpu(fake_loads, checkpoint,
                                                           monkeypatch):
    import sokudan

    calls, failing = fake_loads
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    failing["mlx"] = "nan"
    with pytest.warns(RuntimeWarning, match="mlx"):
        agent = sokudan.load(checkpoint)
    assert (agent.backend.name, agent.device) == ("torch", "cpu")
    assert calls == [("mlx", "gpu"), ("torch", "cpu")]


def test_an_explicit_backend_raises_instead_of_falling_back(fake_loads, checkpoint):
    import sokudan

    _, failing = fake_loads
    failing["mlx"] = "raise"
    with pytest.raises(RuntimeError, match="mlx is broken"):
        sokudan.load(checkpoint, backend="mlx")


def test_an_explicit_device_means_torch_on_that_device(fake_loads, checkpoint):
    import sokudan

    calls, _ = fake_loads
    agent = sokudan.load(checkpoint, device="cpu")
    assert (agent.backend.name, agent.device) == ("torch", "cpu")
    assert calls == [("torch", "cpu")]


def test_import_sokudan_does_not_import_torch():
    code = "import sys, sokudan; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True)
    assert out.stdout.strip() == "False"
