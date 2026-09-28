"""Temperatures at load time (docs/calibration.md §3, §10).

v0.2.1: `load` applies the bool temperatures of the `calibration.json` beside the
checkpoint by default; `temperatures=None` turns calibration off. An `Agent` built without
temperatures (the golden fixture) is uncalibrated.

- `read_temperatures` reads the shipped `calibration.json` format.
- An agent with no temperatures answers exactly as before (the recorded golden outputs);
  with them, each answer is its raw distribution rescaled by its bucket's temperature, and
  the argmax (and a bool's side of 0.5) does not move.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sokudan.calibration.temperature import apply_temperature
from sokudan.predict import read_temperatures

GOLDEN = Path(__file__).parent / "fixtures" / "predict_golden.json"


def core(out: dict) -> dict:
    """The response without v0.2.1's calibration keys (the golden outputs predate them);
    an agent built without temperatures reports `calibrated: False` and no answers."""
    assert out["calibrated"] is False and out["calibrated_answers"] == []
    return {k: v for k, v in out.items() if k not in ("calibrated", "calibrated_answers")}


REPO = Path(__file__).resolve().parents[1]
# the file shipped beside the weights: `assets/calibration.json` in the public repository
# (the copy uploaded to the Hub), `runs/release_candidate/calibration.json` on the research
# branch
SHIPPED = next((p for p in (REPO / "assets" / "calibration.json",
                            REPO / "runs" / "release_candidate" / "calibration.json")
                if p.exists()), REPO / "assets" / "calibration.json")
SAVED_BENCH = REPO / "runs" / "calibration"


def test_read_temperatures_reads_the_calibration_format(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"checkpoint": "x", "procedure": "docs/calibration.md",
                                "temperatures": {"bool/2": 2.07, "choice/4": 2.3,
                                                 "score/3": 4.5}}), encoding="utf-8")
    assert read_temperatures(path) == {("bool", 2): 2.07, ("choice", 4): 2.3,
                                       ("score", 3): 4.5}


def test_the_shipped_file_parses_when_present():
    if not SHIPPED.exists():
        pytest.skip("runs/release_candidate/calibration.json not shipped")
    parsed = read_temperatures(SHIPPED)
    assert ("bool", 2) in parsed and all(t > 0 for t in parsed.values())


@pytest.mark.slow
def test_off_by_default_and_applied_when_given():
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES, build_agent

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    agent = build_agent()
    assert agent.temperatures == {}
    for state, expected in zip(STATES, golden, strict=True):
        assert core(agent.predict(state, QUESTIONS)) == expected
    temps = {("bool", 2): 2.0, ("choice", 4): 2.0, ("score", 3): 2.0}
    agent.temperatures = temps
    for state, expected in zip(STATES, golden, strict=True):
        out = agent.predict(state, QUESTIONS)["answers"]
        raw = expected["answers"]
        p_true = raw["churn"]["noul"]
        want = apply_temperature(np.array([[1 - p_true, p_true]]), 2.0)[0, 1]
        assert abs(out["churn"]["noul"] - want) < 2e-4  # the golden noul is rounded
        assert (out["churn"]["noul"] > 0.5) == (p_true > 0.5)
        assert out["department"]["choice"] == raw["department"]["choice"]
        assert out["department"]["confidence"] <= raw["department"]["confidence"] + 1e-4


# v0.2.1: bool calibration on by default (docs/calibration.md §10) -------------------------

def write(path: Path, temps: dict[str, float]) -> Path:
    path.write_text(json.dumps({"temperatures": temps}), encoding="utf-8")
    return path


def test_default_takes_only_the_bool_temperatures_beside_the_checkpoint(tmp_path):
    from sokudan.predict import DEFAULT_CALIBRATION, resolve_temperatures

    write(tmp_path / "calibration.json", {"bool/2": 2.07, "choice/4": 2.3, "score/3": 4.5})
    assert resolve_temperatures(DEFAULT_CALIBRATION, tmp_path) == {("bool", 2): 2.07}


def test_default_without_a_file_is_uncalibrated_and_none_turns_it_off(tmp_path):
    from sokudan.predict import DEFAULT_CALIBRATION, resolve_temperatures

    assert resolve_temperatures(DEFAULT_CALIBRATION, tmp_path) == {}
    write(tmp_path / "calibration.json", {"bool/2": 2.07})
    assert resolve_temperatures(None, tmp_path) == {}


def test_an_explicit_file_or_dict_is_applied_whole(tmp_path):
    from sokudan.predict import resolve_temperatures

    full = write(tmp_path / "calibration_full.json", {"bool/2": 2.0, "score/3": 4.5})
    assert resolve_temperatures(full, None) == {("bool", 2): 2.0, ("score", 3): 4.5}
    beside = resolve_temperatures("calibration_full.json", tmp_path)
    assert beside == {("bool", 2): 2.0, ("score", 3): 4.5}
    assert resolve_temperatures({("choice", 4): 1.5}, tmp_path) == {("choice", 4): 1.5}
    with pytest.raises(FileNotFoundError):
        resolve_temperatures(tmp_path / "missing.json", None)


def test_the_shipped_default_is_bool_only():
    from sokudan.predict import DEFAULT_CALIBRATION, resolve_temperatures

    if not SHIPPED.exists():
        pytest.skip("runs/release_candidate/calibration.json not shipped")
    temps = resolve_temperatures(DEFAULT_CALIBRATION, SHIPPED.parent)
    assert set(temps) == {("bool", 2)}


@pytest.mark.parametrize("bench", ["bench_ja", "bench_en"])
def test_default_on_keeps_bool_argmax_and_auroc_and_lowers_ece(bench):
    """On the saved bench outputs (docs/calibration.md §7, no inference)."""
    from scripts.calibration_heldout import p_true_ece
    from sokudan.calibration.metrics import auroc
    from sokudan.predict import DEFAULT_CALIBRATION, resolve_temperatures

    saved = SAVED_BENCH / f"{bench}.npz"
    if not SHIPPED.exists() or not saved.exists():
        pytest.skip("shipped calibration or saved bench outputs not present")
    t = resolve_temperatures(DEFAULT_CALIBRATION, SHIPPED.parent)[("bool", 2)]
    d = np.load(saved)
    y = d["gold_bool"]
    off = d["bool_probs"][:, 1].astype(np.float64)
    on = apply_temperature(d["bool_probs"], t)[:, 1]
    assert np.array_equal(off > 0.5, on > 0.5)
    assert auroc(on, y) == pytest.approx(auroc(off, y), abs=1e-12)
    assert p_true_ece(on, y) < p_true_ece(off, y)


@pytest.mark.slow
def test_responses_say_which_answers_were_calibrated():
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES, build_agent

    agent = build_agent()
    agent.temperatures = {("bool", 2): 2.07}
    out = agent.predict(STATES[0], QUESTIONS)
    assert out["calibrated"] is True and out["calibrated_answers"] == ["churn"]
    only_choice = {"department": QUESTIONS["department"]}
    out = agent.predict(STATES[0], only_choice)
    assert out["calibrated"] is False and out["calibrated_answers"] == []
