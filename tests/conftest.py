"""Shared fixtures, and the GPU rule (docs/research_protocol.md §8).

CUDA is hidden from every test by default: a test run must never take GPU memory from
a training run in another process (it did once, VRAM total 30.5 of 32.6 GB). Tests that
need the device carry `@pytest.mark.gpu` and run only with `SOKUDAN_TEST_GPU=1`, when
nothing else is on the GPU. The variable is set here, before any test module imports
torch, so it also reaches subprocesses the tests start.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

GPU_ALLOWED = os.environ.get("SOKUDAN_TEST_GPU") == "1"
if not GPU_ALLOWED:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

from sokudan.config import BACKBONE_MODEL_ID  # noqa: E402

DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A test that reads generated data this checkout does not have is skipped, not failed.

    The training and validation corpora under `data/` (`train_v2b.jsonl`, `val_v2.jsonl`,
    the held-out sets, the distillation cache ...) are generated locally and not published,
    so a fresh clone cannot run the tests that check behaviour on every row of them. Only a
    `FileNotFoundError` for a path inside `data/` is turned into a skip, with the path as the
    reason; any other failure, and any missing file elsewhere, still fails.
    """
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is None or not call.excinfo.errisinstance(FileNotFoundError):
        return
    missing = getattr(call.excinfo.value, "filename", None)
    if not missing:
        return
    path = Path(missing).resolve()
    if DATA_DIR in path.parents:
        report.outcome = "skipped"
        where = path.relative_to(DATA_DIR.parent)
        report.longrepr = (str(item.path), item.location[1] or 0,
                           f"Skipped: generated data not in this checkout: {where}")


def pytest_collection_modifyitems(config, items):
    if GPU_ALLOWED:
        return
    skip = pytest.mark.skip(reason="gpu test: run with SOKUDAN_TEST_GPU=1 when the GPU is free")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


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
