"""`POST /v1/systemone` contract tests.

A fake agent stands in for the model. What is being tested is the wire contract --
field names, aliases, limits, status codes -- and a real checkpoint would make these
slow without testing any of it more thoroughly. `test_predict.py` covers the numbers.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from sokudan.serve import app as serve


class FakeAgent:
    """Returns the shape `Agent.predict` returns, without loading 338M parameters."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        answers: dict[str, Any] = {}
        for key, question in questions.items():
            kind = question["type"]
            if kind in ("bool", "noul"):
                answers[key] = {"type": "bool", "noul": 0.1234}
            elif kind == "score":
                levels = question["criteria"]
                answers[key] = {
                    "type": "score",
                    "score": 1.0,
                    "probabilities": {str(i): 1 / len(levels) for i in range(len(levels))},
                    "legend": {str(i): label for i, label in enumerate(levels)},
                    "confidence": 0.5,
                }
            else:
                options = list(question["criteria"])
                answers[key] = {
                    "type": "choice",
                    "choice": options[0],
                    "probabilities": {o: 1 / len(options) for o in options},
                    "confidence": 0.5,
                }
        return {
            "model": "sokudan-ja-310m",
            "answers": answers,
            "usage": {"state_tokens": 10, "state_truncated": False,
                      "question_tokens": 20, "output_tokens": 0},
        }


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch):
    agent = FakeAgent()
    monkeypatch.setattr(serve, "load_agent", lambda: None)
    with TestClient(serve.app) as test_client:
        serve.state.agent = agent
        serve.state.checkpoint = "runs/test/model.pt"
        serve.state.calibrated = False
        serve.state.load_error = None
        test_client.agent = agent  # type: ignore[attr-defined]
        yield test_client
    serve.state.agent = None


BENCH_QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "この問い合わせはどの部署が担当すべきか",
        "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                     "営業": "料金・新規契約", "その他": "上記以外"},
    },
    "urgency": {
        "type": "score",
        "instructions": "この依頼の緊急度は",
        "criteria": ["急がない", "早めに", "業務が止まっている"],
    },
    "churn": {"type": "noul", "instructions": "送信者は解約を示唆しているか"},
}

STATE = "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。"


def test_systemone_returns_the_published_wire_shape(client):
    response = client.post("/v1/systemone",
                           json={"state": STATE, "questions": BENCH_QUESTIONS})
    assert response.status_code == 200
    body = response.json()

    assert set(body) >= {"model", "answers", "usage"}
    assert set(body["answers"]) == {"department", "urgency", "churn"}
    assert body["answers"]["department"]["choice"] in BENCH_QUESTIONS["department"]["criteria"]
    assert "probabilities" in body["answers"]["department"]
    assert "score" in body["answers"]["urgency"]
    assert "noul" in body["answers"]["churn"]


def test_noul_and_bool_are_the_same_type_on_the_wire(client):
    for alias in ("noul", "bool"):
        response = client.post("/v1/systemone", json={
            "state": STATE,
            "questions": {"q": {"type": alias, "instructions": "解約の示唆があるか"}},
        })
        assert response.status_code == 200, alias
        assert "noul" in response.json()["answers"]["q"], alias


def test_model_field_is_accepted_and_ignored(client):
    response = client.post("/v1/systemone", json={
        "state": STATE, "questions": BENCH_QUESTIONS, "model": "jev-latest",
    })
    assert response.status_code == 200
    # The server reports what actually answered, not what was asked for.
    assert response.json()["model"] == "sokudan-ja-310m"


def test_state_accepts_a_string_a_mapping_and_a_transcript(client):
    for state in (STATE, {"body": STATE}, [{"role": "user", "content": STATE}]):
        response = client.post("/v1/systemone", json={
            "state": state, "questions": {"q": {"type": "noul", "instructions": "至急か"}},
        })
        assert response.status_code == 200


def test_uncalibrated_responses_say_so(client):
    body = client.post("/v1/systemone", json={
        "state": STATE, "questions": {"q": {"type": "noul", "instructions": "至急か"}},
    }).json()
    assert body["calibrated"] is False
    assert "較正" in body["calibration_note"]


def test_calibrated_responses_carry_no_warning(client):
    serve.state.calibrated = True
    body = client.post("/v1/systemone", json={
        "state": STATE, "questions": {"q": {"type": "noul", "instructions": "至急か"}},
    }).json()
    assert body["calibrated"] is True
    assert "calibration_note" not in body


def test_choice_without_criteria_is_rejected(client):
    response = client.post("/v1/systemone", json={
        "state": STATE, "questions": {"q": {"type": "choice", "instructions": "どれか"}},
    })
    assert response.status_code == 422


def test_score_criteria_must_be_an_ordered_list(client):
    response = client.post("/v1/systemone", json={
        "state": STATE,
        "questions": {"q": {"type": "score", "instructions": "程度は",
                            "criteria": {"低": "", "高": ""}}},
    })
    assert response.status_code == 422


def test_unknown_question_type_is_rejected(client):
    response = client.post("/v1/systemone", json={
        "state": STATE, "questions": {"q": {"type": "ranking", "instructions": "順位は"}},
    })
    assert response.status_code == 422


def test_empty_questions_is_rejected(client):
    response = client.post("/v1/systemone", json={"state": STATE, "questions": {}})
    assert response.status_code == 422


def test_too_many_questions_is_rejected(client):
    questions = {
        f"q{i}": {"type": "noul", "instructions": f"質問{i}"}
        for i in range(serve.MAX_QUESTIONS + 1)
    }
    response = client.post("/v1/systemone", json={"state": STATE, "questions": questions})
    assert response.status_code == 422


def test_oversized_state_is_rejected_with_413(client):
    response = client.post("/v1/systemone", json={
        "state": "あ" * (serve.MAX_STATE_CHARS + 1),
        "questions": {"q": {"type": "noul", "instructions": "至急か"}},
    })
    assert response.status_code == 413


def test_healthz_is_up_even_without_a_model(client):
    serve.state.agent = None
    assert client.get("/healthz").status_code == 200


def test_ready_is_503_without_a_model_and_names_the_reason(client):
    serve.state.agent = None
    serve.state.load_error = "SOKUDAN_CHECKPOINT is not set"
    response = client.get("/ready")
    assert response.status_code == 503
    assert "SOKUDAN_CHECKPOINT" in response.json()["reason"]


def test_ready_reports_calibration_state(client):
    body = client.get("/ready").json()
    assert body["status"] == "ready"
    assert body["calibrated"] is False
    assert body["limits"]["max_questions"] == serve.MAX_QUESTIONS


def test_requests_without_a_model_get_503(client):
    serve.state.agent = None
    serve.state.load_error = "no model loaded"
    response = client.post("/v1/systemone", json={
        "state": STATE, "questions": {"q": {"type": "noul", "instructions": "至急か"}},
    })
    assert response.status_code == 503


def test_queue_overflow_returns_429_with_retry_after(client):
    serve.state.inflight = serve.MAX_CONCURRENCY + serve.MAX_QUEUE
    try:
        response = client.post("/v1/systemone", json={
            "state": STATE, "questions": {"q": {"type": "noul", "instructions": "至急か"}},
        })
        assert response.status_code == 429
        assert response.headers.get("Retry-After") == "1"
    finally:
        serve.state.inflight = 0


def test_the_state_reaches_the_model_unchanged(client):
    client.post("/v1/systemone", json={"state": STATE, "questions": BENCH_QUESTIONS})
    seen_state, seen_questions = client.agent.calls[-1]
    assert seen_state == STATE
    assert seen_questions["department"]["criteria"] == BENCH_QUESTIONS["department"]["criteria"]
    # Option order decides which marker means which option (§7.2), so the server
    # must not sort, dedupe or otherwise tidy it on the way through.
    assert list(seen_questions["department"]["criteria"]) == \
        list(BENCH_QUESTIONS["department"]["criteria"])


# ---------------------------------------------------------------------------
# Checkpoint resolution. The README's Quickstart names a Hub repo id, so the
# package has to accept one -- a README describing behaviour the code lacks is the
# same class of error as an unmeasured number in the model card.
# ---------------------------------------------------------------------------


def test_a_directory_of_safetensors_loads_like_a_pt_file(tmp_path) -> None:
    import json as _json

    import torch
    from safetensors.torch import save_file

    from sokudan.predict import _resolve_checkpoint

    save_file({"scorer.weight": torch.zeros(1, 8)}, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(
        _json.dumps({"encoding": "joint", "backbone": "b"}), encoding="utf-8"
    )

    blob, directory = _resolve_checkpoint(tmp_path)
    assert "scorer.weight" in blob["state_dict"]
    assert blob["config"]["encoding"] == "joint"
    assert directory == tmp_path


def test_a_pt_file_still_loads_and_reports_its_directory(tmp_path) -> None:
    import torch

    from sokudan.predict import _resolve_checkpoint

    path = tmp_path / "model.pt"
    torch.save({"state_dict": {"scorer.weight": torch.zeros(1, 8)},
                "config": {"encoding": "separate"}}, path)

    blob, directory = _resolve_checkpoint(path)
    assert blob["config"]["encoding"] == "separate"
    assert directory == tmp_path


def test_a_name_that_is_neither_a_path_nor_a_repo_id_is_rejected() -> None:
    from sokudan.predict import _resolve_checkpoint

    with pytest.raises(FileNotFoundError, match="Hub repo id"):
        _resolve_checkpoint("not-a-path-and-not-a-repo")


def test_a_directory_without_weights_says_so(tmp_path) -> None:
    from sokudan.predict import _resolve_checkpoint

    with pytest.raises(FileNotFoundError, match="model.safetensors"):
        _resolve_checkpoint(tmp_path)
