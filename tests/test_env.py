"""Environment smoke tests (SOKUDAN_SPEC.md §4.1).

These assert the machine can actually run the planned training, not merely that
imports succeed. The distinction matters: a cu126 build on this box imports fine,
reports `is_available() == True`, and then fails on the first real kernel launch
with "no kernel image is available for execution on the device".
"""

from __future__ import annotations

import sys

import pytest

BLACKWELL_CAPABILITY = (12, 0)  # RTX 5090, sm_120


def test_python_is_311() -> None:
    assert sys.version_info[:2] == (3, 11), f"expected Python 3.11, got {sys.version}"


def test_transformers_version() -> None:
    import transformers
    from packaging.version import Version

    assert Version(transformers.__version__) >= Version("4.48.0"), transformers.__version__


def test_torch_cuda_build_is_128_or_newer() -> None:
    import torch
    from packaging.version import Version

    assert torch.version.cuda is not None, "CPU-only torch build"
    assert Version(torch.version.cuda) >= Version("12.8"), (
        f"torch built against CUDA {torch.version.cuda}; sm_120 needs 12.8+"
    )


@pytest.mark.gpu
def test_cuda_is_available() -> None:
    import torch

    assert torch.cuda.is_available(), "no CUDA device visible"


@pytest.mark.gpu
def test_device_capability_is_sm120() -> None:
    import torch

    assert torch.cuda.get_device_capability(0) == BLACKWELL_CAPABILITY


@pytest.mark.gpu
def test_sm120_kernels_are_compiled_in() -> None:
    """`arch_list` must contain sm_120; otherwise every kernel launch fails."""
    import torch

    arch_list = torch.cuda.get_arch_list()
    assert "sm_120" in arch_list, f"sm_120 missing from arch_list: {arch_list}"


@pytest.mark.gpu
def test_bf16_matmul_actually_runs_on_device() -> None:
    """The test that the broken cu126 install fails. Do not remove."""
    import torch

    a = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16)
    out = (a @ a).float()
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()


@pytest.mark.gpu
def test_vram_is_at_least_30gb() -> None:
    import torch

    total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    assert total_gb >= 30.0, f"only {total_gb:.1f} GiB of VRAM"
