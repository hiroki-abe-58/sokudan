"""`sokudan.load` no longer prints transformers' load report (v0.3.0).

The report listed the pretrained checkpoint's masked-LM head (`head.*`, `decoder.bias`)
as UNEXPECTED on every `sokudan.load`. `TorchBackend.load` builds the model inside
`quiet_transformers()`, which silences it for that call only; `Backbone.load` on its own
(training) keeps it. The weights must not change.
"""

from __future__ import annotations

import pytest
import torch
from transformers.utils import logging as hf_logging

from sokudan.backends.torch_backend import quiet_transformers
from sokudan.model.backbone import Backbone


def _reports(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "LOAD REPORT" in r.getMessage()]


def test_default_backbone_load_still_reports(caplog):
    Backbone.load()
    reports = _reports(caplog)
    assert reports and "UNEXPECTED" in reports[0]


def test_quiet_backbone_load_logs_no_report_and_restores_the_level(caplog):
    before = hf_logging.get_verbosity()
    with quiet_transformers():
        Backbone.load()
    assert _reports(caplog) == []
    assert hf_logging.get_verbosity() == before


def test_quiet_changes_no_weight():
    loud = Backbone.load().state_dict()
    with quiet_transformers():
        quiet = Backbone.load().state_dict()
    assert loud.keys() == quiet.keys()
    assert all(torch.equal(loud[k], quiet[k]) for k in loud)


@pytest.mark.slow
def test_sokudan_load_on_torch_logs_no_report(caplog):
    import sokudan

    try:
        agent = sokudan.load("GeneLab/sokudan-ja-310m", backend="torch", device="cpu",
                             temperatures=None)
    except Exception as exc:  # pragma: no cover - depends on the local cache / network
        pytest.skip(f"model unavailable: {type(exc).__name__}: {exc}")
    assert agent.backend.name == "torch"
    assert _reports(caplog) == []
