"""Which array library `pip install sokudan` brings, per platform, and what a missing one says.

The base dependencies carry environment markers (pyproject.toml): Apple silicon on Darwin
23+ (macOS 14+) gets MLX and no torch, everything else gets torch. The table below is
evaluated from the pyproject itself, so a marker edit that moves a platform fails here.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from packaging.requirements import Requirement

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"

# (sys_platform, platform_machine, platform_release) -> (torch installed, mlx installed)
PLATFORMS = {
    ("linux", "x86_64", "6.8.0-45-generic"): (True, False),
    ("linux", "aarch64", "6.8.0-45-generic"): (True, False),
    ("win32", "AMD64", "10"): (True, False),
    ("darwin", "x86_64", "23.6.0"): (True, False),
    ("darwin", "arm64", "22.6.0"): (True, False),
    ("darwin", "arm64", "23.6.0"): (False, True),
    ("darwin", "arm64", "24.6.0"): (False, True),
    ("darwin", "arm64", "25.0.0"): (False, True),
}


def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _installed(requirements: list[str], name: str, environment: dict[str, str]) -> bool:
    found = [Requirement(r) for r in requirements if Requirement(r).name == name]
    return any(r.marker is None or r.marker.evaluate(environment) for r in found)


@pytest.mark.parametrize(("platform", "expected"), PLATFORMS.items(),
                         ids=["-".join(p) for p in PLATFORMS])
def test_base_install_brings_torch_or_mlx_by_platform(platform, expected):
    sys_platform, machine, release = platform
    environment = {"sys_platform": sys_platform, "platform_machine": machine,
                   "platform_release": release}
    base = _project()["dependencies"]
    assert (_installed(base, "torch", environment), _installed(base, "mlx", environment)) \
        == expected


@pytest.mark.parametrize("platform", PLATFORMS, ids=["-".join(p) for p in PLATFORMS])
def test_the_torch_extra_brings_torch_everywhere(platform):
    environment = dict(zip(("sys_platform", "platform_machine", "platform_release"),
                           platform, strict=True))
    assert _installed(_project()["optional-dependencies"]["torch"], "torch", environment)


def test_the_mlx_extra_uses_the_base_marker():
    project = _project()
    base = {str(Requirement(r).marker) for r in project["dependencies"]
            if Requirement(r).name == "mlx"}
    extra = {str(Requirement(r).marker) for r in project["optional-dependencies"]["mlx"]}
    assert base == extra and len(base) == 1


@pytest.fixture
def checkpoint(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"")
    return tmp_path


@pytest.mark.parametrize(("backend", "modules", "hint"), [
    ("torch", ["torch", "sokudan.backends.torch_backend"], r"sokudan\[torch\]"),
    ("mlx", ["mlx", "mlx.core", "mlx.nn", "sokudan.backends.mlx",
             "sokudan.backends.mlx.model", "sokudan.backends.mlx.modernbert"],
     r"sokudan\[mlx\]"),
])
def test_an_explicit_backend_that_is_not_installed_says_how_to_install_it(
        monkeypatch, checkpoint, backend, modules, hint):
    import sokudan

    for module in modules[1:]:
        monkeypatch.delitem(sys.modules, module, raising=False)
    # None in sys.modules makes `import <module>` raise ModuleNotFoundError.
    monkeypatch.setitem(sys.modules, modules[0], None)
    with pytest.raises(ImportError, match=hint):
        sokudan.load(checkpoint, backend=backend)
