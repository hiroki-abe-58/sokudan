"""`sokudan serve`: the `/v1/systemone`-compatible endpoint.

A fake agent stands in for the model, as in `test_serve.py`. It runs the real question
parser, so a request sokudan cannot read fails here the way it would with weights
loaded, and returns `Agent.predict`'s exact shape (`type: "bool"`, top-probability
confidence, sokudan's usage block) so the translation to the wire shape is what is
under test.

The published examples in `fixtures/systemone_wire_examples.json` are sent as they
are. Their response values are the publishers' and are never compared; field names,
types and key sets are.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sokudan.schema.question import parse_questions
from sokudan.serve import wire
from sokudan.serve.systemone import ServerSettings, create_app, health_body

EXAMPLES = json.loads(
    (Path(__file__).parent / "fixtures" / "systemone_wire_examples.json").read_text(
        encoding="utf-8")
)


class FakeAgent:
    """`Agent.predict`'s output shape, with probabilities that favour the first slot."""

    device = "cpu"

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def predict(self, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((state, questions))
        parsed = parse_questions(questions)
        answers: dict[str, Any] = {}
        for key, question in parsed.items():
            labels = list(question.labels)
            weights = [len(labels) - i for i in range(len(labels))]
            probs = [w / sum(weights) for w in weights]
            if question.type == "bool":
                answers[key] = {"type": "bool", "noul": round(probs[1], 4)}
            elif question.type == "score":
                answers[key] = {
                    "type": "score",
                    "score": sum(i * p for i, p in enumerate(probs)),
                    "probabilities": {str(i): round(p, 4) for i, p in enumerate(probs)},
                    "legend": {str(i): label for i, label in enumerate(labels)},
                    "confidence": round(max(probs), 4),
                }
            else:
                answers[key] = {
                    "type": "choice",
                    "choice": labels[0],
                    "probabilities": {label: round(p, 4)
                                      for label, p in zip(labels, probs, strict=True)},
                    "confidence": round(max(probs), 4),
                }
        n = len(parsed)
        return {
            "model": "sokudan-ja-310m",
            "answers": answers,
            "encoding": "joint",
            "usage": {"state_tokens": 12, "state_truncated": False,
                      "question_tokens": 20 * n, "backbone_passes": n,
                      "output_tokens": 0},
        }


@pytest.fixture
def agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture
def client(agent: FakeAgent):
    settings = ServerSettings(model_ref="test/fake", device="cpu")
    with TestClient(create_app(agent, settings)) as test_client:
        yield test_client


STATE = "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。"

QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "この問い合わせはどの部署が担当すべきか",
        "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                     "営業": "料金・新規契約", "その他": None},
    },
    "urgency": {
        "type": "score",
        "instructions": "この依頼の緊急度は",
        "criteria": ["急がない", "早めに", "業務が止まっている"],
    },
    "churn": {"type": "noul", "instructions": "送信者は解約を示唆しているか"},
}

ANSWER_KEYS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "probabilities", "confidence"},
    "score": {"type", "score", "legend", "probabilities", "confidence"},
}


def post(client: TestClient, body: dict[str, Any], **kwargs: Any):
    return client.post("/v1/systemone", json=body, **kwargs)


def assert_wire_answer(question: dict[str, Any], answer: dict[str, Any]) -> None:
    """The published answer shape for one question, with no sokudan fields mixed in."""
    kind = "noul" if question["type"] in ("noul", "bool") else question["type"]
    assert answer["type"] == kind
    assert set(answer) == ANSWER_KEYS[kind]
    if kind == "noul":
        assert isinstance(answer["noul"], float) and 0.0 <= answer["noul"] <= 1.0
        return
    assert isinstance(answer["confidence"], float) and 0.0 <= answer["confidence"] <= 1.0
    probabilities = answer["probabilities"]
    assert all(isinstance(p, float) for p in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    if kind == "choice":
        assert list(probabilities) == list(question["criteria"])
        assert answer["choice"] in question["criteria"]
    else:
        levels = question["criteria"]
        assert list(probabilities) == [str(i) for i in range(len(levels))]
        assert answer["legend"] == {str(i): level for i, level in enumerate(levels)}
        assert isinstance(answer["score"], float)
        assert 0.0 <= answer["score"] <= len(levels) - 1


def assert_wire_response(body: dict[str, Any], request: dict[str, Any]) -> None:
    assert isinstance(body["model"], str)
    assert set(body["answers"]) == set(request["questions"])
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}
    assert isinstance(body["usage"]["input_tokens"], int)
    assert body["usage"]["output_tokens"] == 0
    for key, question in request["questions"].items():
        assert_wire_answer(question, body["answers"][key])


# ---------------------------------------------------------------------------
# The three types, round trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(QUESTIONS))
def test_each_type_round_trips_to_the_wire_shape(client, key):
    request = {"state": STATE, "model": "jev-latest", "questions": {key: QUESTIONS[key]}}
    response = post(client, request)
    assert response.status_code == 200, response.text
    assert_wire_response(response.json(), request)


def test_noul_answer_says_noul_not_bool(client):
    body = post(client, {"state": STATE, "questions": {"q": QUESTIONS["churn"]}}).json()
    assert body["answers"]["q"] == {"type": "noul", "noul": body["answers"]["q"]["noul"]}


def test_bool_is_accepted_as_sokudans_name_for_noul(client):
    body = post(client, {"state": STATE, "questions": {
        "q": {"type": "bool", "instructions": "至急か"}}}).json()
    assert body["answers"]["q"]["type"] == "noul"


def test_choice_null_description_reaches_the_model_as_a_bare_label(client, agent):
    post(client, {"state": STATE, "questions": {"department": QUESTIONS["department"]}})
    _, questions = agent.calls[-1]
    assert questions["department"]["criteria"] == {
        "請求": "支払い・返金", "技術": "不具合・障害", "営業": "料金・新規契約", "その他": "",
    }


def test_option_and_level_order_are_kept(client, agent):
    criteria = {"z": None, "a": None, "m": None}
    body = post(client, {"state": STATE, "questions": {
        "c": {"type": "choice", "instructions": "どれか", "criteria": criteria},
        "s": {"type": "score", "instructions": "程度は", "criteria": ["高", "低", "中"]},
    }}).json()
    _, questions = agent.calls[-1]
    assert list(questions["c"]["criteria"]) == ["z", "a", "m"]
    assert questions["s"]["criteria"] == ["高", "低", "中"]
    assert list(body["answers"]["c"]["probabilities"]) == ["z", "a", "m"]


def test_structured_instructions_and_levels_are_rendered_and_the_legend_echoes_them(
        client, agent):
    level = {"label": "急ぎ", "examples": ["今日中", "至急"]}
    body = post(client, {"state": STATE, "questions": {
        "q": {"type": "score",
              "instructions": {"question": "緊急度は", "context": {"sla_hours": 24}},
              "criteria": ["急がない", level]},
    }}).json()
    _, questions = agent.calls[-1]
    assert questions["q"]["instructions"] == 'question: 緊急度は\ncontext: {"sla_hours": 24}'
    assert questions["q"]["criteria"][1] == 'label: 急ぎ\nexamples: ["今日中", "至急"]'
    assert body["answers"]["q"]["legend"] == {"0": "急がない", "1": level}


def test_noul_criteria_are_rendered_into_the_yes_and_no_markers(client, agent):
    post(client, {"state": STATE, "questions": {
        "q": {"type": "noul", "instructions": "急ぎか",
              "criteria": {"true": "期限が明記されている", "false": "期限がない"}},
    }})
    _, questions = agent.calls[-1]
    assert questions["q"]["true_label"] == "はい: 期限が明記されている"
    assert questions["q"]["false_label"] == "いいえ: 期限がない"


def test_noul_without_criteria_uses_the_trained_labels(client, agent):
    post(client, {"state": STATE, "questions": {"q": QUESTIONS["churn"]}})
    _, questions = agent.calls[-1]
    assert "true_label" not in questions["q"] and "false_label" not in questions["q"]


# ---------------------------------------------------------------------------
# Published examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "example", EXAMPLES["requests"], ids=[e["source"] for e in EXAMPLES["requests"]]
)
def test_published_requests_are_answered_in_the_wire_shape(client, example):
    response = post(client, example["body"])
    assert response.status_code == 200, response.text
    assert_wire_response(response.json(), example["body"])


@pytest.mark.parametrize(
    "published", EXAMPLES["responses"], ids=[e["source"] for e in EXAMPLES["responses"]]
)
def test_response_shape_matches_the_published_response(client, published):
    request = EXAMPLES["requests"][published["request_index"]]["body"]
    ours = post(client, request).json()
    theirs = published["body"]

    assert set(theirs) - {"latency_ms"} <= set(ours)
    assert set(ours["usage"]) == set(theirs["usage"])
    assert set(ours["answers"]) == set(theirs["answers"])
    for key, answer in theirs["answers"].items():
        mine = ours["answers"][key]
        assert set(mine) == set(answer), key
        for field, value in answer.items():
            assert type(mine[field]) is type(value) or (
                isinstance(value, (int, float)) and isinstance(mine[field], float)
            ), (key, field)
        if "probabilities" in answer:
            assert list(mine["probabilities"]) == list(answer["probabilities"])
        if "legend" in answer:
            assert mine["legend"] == answer["legend"]


# ---------------------------------------------------------------------------
# confidence: the wire definition, checked against the published numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("published", EXAMPLES["responses"],
                         ids=[e["source"] for e in EXAMPLES["responses"]])
def test_confidence_formula_reproduces_published_confidences(published):
    """Published probabilities are rounded to two places, so agreement is to 0.01."""
    for answer in published["body"]["answers"].values():
        if answer["type"] == "choice":
            ours = wire.choice_confidence(list(answer["probabilities"].values()))
        elif answer["type"] == "score":
            ours = wire.score_confidence(list(answer["probabilities"].values()))
        else:
            continue
        assert ours == pytest.approx(answer["confidence"], abs=0.011)


def test_confidence_limits():
    assert wire.choice_confidence([1.0, 0.0, 0.0]) == 1.0
    assert wire.choice_confidence([1 / 3] * 3) == 0.0
    assert wire.choice_confidence([1.0]) == 1.0
    assert wire.score_confidence([0.0, 1.0, 0.0]) == 1.0
    assert wire.score_confidence([1 / 3] * 3) == 0.0
    assert wire.score_confidence([0.5, 0.0, 0.5]) == 0.0


def test_confidence_is_the_wire_definition_not_the_top_probability(client):
    body = post(client, {"state": STATE, "questions": {"q": QUESTIONS["department"]}}).json()
    probabilities = list(body["answers"]["q"]["probabilities"].values())
    assert body["answers"]["q"]["confidence"] == wire.choice_confidence(probabilities)
    assert body["answers"]["q"]["confidence"] != max(probabilities)


# ---------------------------------------------------------------------------
# One call per request, usage, state handling
# ---------------------------------------------------------------------------


def test_all_questions_go_to_the_model_in_one_call(client, agent):
    body = post(client, {"state": STATE, "questions": QUESTIONS}).json()
    assert len(agent.calls) == 1
    assert set(agent.calls[0][1]) == set(QUESTIONS)
    assert set(body["answers"]) == set(QUESTIONS)
    assert body["sokudan"]["backbone_passes"] == 3


def test_input_tokens_count_the_state_once_per_joint_pass(client):
    body = post(client, {"state": STATE, "questions": QUESTIONS}).json()
    # FakeAgent: 12 state tokens, 20 question tokens per question, 3 questions.
    assert body["usage"] == {"input_tokens": 3 * 20 + 3 * 12, "output_tokens": 0}


def test_input_tokens_count_the_state_once_with_separate_encoding():
    usage = {"state_tokens": 12, "question_tokens": 60, "backbone_passes": 2}
    assert wire.input_tokens(usage, "separate") == 72


@pytest.mark.parametrize("state, fmt", [
    (STATE, "text"),
    ({"body": STATE}, "object_as_key_value_lines"),
    ([{"role": "user", "content": STATE}], "array_as_lines"),
])
def test_state_reaches_predict_unchanged_and_the_rendering_is_reported(
        client, agent, state, fmt):
    body = post(client, {"state": state, "questions": {"q": QUESTIONS["churn"]}}).json()
    assert agent.calls[-1][0] == state
    assert body["sokudan"]["state_format"] == fmt


def test_model_field_is_optional_and_the_response_names_what_answered(client):
    for extra in ({}, {"model": "jev-latest"}, {"model": "kev-latest"}):
        body = post(client, {"state": STATE, "questions": {"q": QUESTIONS["churn"]},
                             **extra}).json()
        assert body["model"] == "sokudan-ja-310m"


def test_authorization_header_is_accepted_and_not_checked(client):
    for header in ("Bearer local", "Bearer anything", "garbage"):
        response = post(client, {"state": STATE, "questions": {"q": QUESTIONS["churn"]}},
                        headers={"Authorization": header})
        assert response.status_code == 200


def test_the_response_says_whether_it_was_calibrated(client):
    # the fake agent carries no temperatures and reports none
    body = post(client, {"state": STATE, "questions": {"q": QUESTIONS["churn"]}}).json()
    assert body["sokudan"]["calibrated"] is False
    assert body["sokudan"]["calibrated_answers"] == []
    assert "order_marginalize" not in body["sokudan"]


def test_predicts_calibration_fields_reach_the_response(agent):
    real = agent.predict

    def calibrated_predict(state, questions):
        out = real(state, questions)
        return {**out, "calibrated": True, "calibrated_answers": ["q"]}

    agent.predict = calibrated_predict
    with TestClient(create_app(agent, ServerSettings(model_ref="test/fake"))) as client:
        body = post(client, {"state": STATE, "questions": {"q": QUESTIONS["churn"]}}).json()
    assert body["sokudan"]["calibrated"] is True
    assert body["sokudan"]["calibrated_answers"] == ["q"]


# ---------------------------------------------------------------------------
# 422 and the other refusals
# ---------------------------------------------------------------------------


def detail_text(response) -> str:
    return json.dumps(response.json()["detail"], ensure_ascii=False)


@pytest.mark.parametrize("question, reason", [
    ({"type": "ranking", "instructions": "順位は"}, "ranking"),
    ({"type": "choice", "instructions": "どれか"}, "criteria"),
    ({"type": "choice", "instructions": "どれか", "criteria": {}}, "at least one option"),
    ({"type": "choice", "instructions": "どれか",
      "criteria": {f"o{i}": None for i in range(256)}}, "at most 255"),
    ({"type": "choice", "instructions": "どれか", "criteria": {" ": None}}, "blank"),
    ({"type": "choice", "instructions": "どれか", "criteria": ["a", "b"]}, "criteria"),
    ({"type": "score", "instructions": "程度は", "criteria": ["低"]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "程度は",
      "criteria": [str(i) for i in range(11)]}, "2 to 10 levels"),
    ({"type": "score", "instructions": "程度は", "criteria": ["低", None]}, "level 1"),
    ({"type": "score", "instructions": "程度は", "criteria": {"低": "", "高": ""}}, "criteria"),
    ({"type": "noul", "instructions": ""}, "instructions"),
    ({"type": "noul"}, "instructions"),
    ({"type": "noul", "instructions": "至急か", "criteria": {"maybe": "x"}}, "maybe"),
])
def test_malformed_questions_get_422_with_the_reason(client, question, reason):
    response = post(client, {"state": STATE, "questions": {"q": question}})
    assert response.status_code == 422
    assert reason in detail_text(response)


def test_levels_that_render_to_the_same_text_get_422_from_sokudans_parser(client, agent):
    response = post(client, {"state": STATE, "questions": {
        "q": {"type": "score", "instructions": "程度は", "criteria": ["同じ", "同じ"]}}})
    assert response.status_code == 422
    assert "distinct" in detail_text(response)
    assert agent.calls == []


@pytest.mark.parametrize("body, reason", [
    ({"questions": {"q": {"type": "noul", "instructions": "至急か"}}}, "state"),
    ({"state": None, "questions": {"q": {"type": "noul", "instructions": "至急か"}}},
     "state is null"),
    ({"state": STATE, "questions": {}}, "questions is empty"),
    ({"state": STATE}, "questions"),
    ({"state": STATE, "questions": {" ": {"type": "noul", "instructions": "至急か"}}},
     "blank"),
])
def test_malformed_requests_get_422_with_the_reason(client, body, reason):
    response = post(client, body)
    assert response.status_code == 422
    assert reason in detail_text(response)


def test_too_many_questions_get_422(client):
    questions = {f"q{i}": {"type": "noul", "instructions": f"質問{i}"}
                 for i in range(wire.MAX_QUESTIONS + 1)}
    response = post(client, {"state": STATE, "questions": questions})
    assert response.status_code == 422
    assert f"at most {wire.MAX_QUESTIONS}" in detail_text(response)


def test_oversized_state_gets_413(agent):
    settings = ServerSettings(max_state_chars=10)
    with TestClient(create_app(agent, settings)) as small:
        response = post(small, {"state": "あ" * 11, "questions": {"q": QUESTIONS["churn"]}})
    assert response.status_code == 413


def test_no_model_gets_503():
    with TestClient(create_app(None)) as empty:
        response = post(empty, {"state": STATE, "questions": {"q": QUESTIONS["churn"]}})
        assert response.status_code == 503
        assert empty.get("/health").status_code == 503


# ---------------------------------------------------------------------------
# /health, order_marginalize, CLI
# ---------------------------------------------------------------------------


def test_health_reports_what_a_client_needs_to_know(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "sokudan-ja-310m"
    assert body["model_ref"] == "test/fake"
    assert body["calibrated"] is False
    assert body["calibration"]["temperatures"] == {}
    assert "order_marginalize" not in body
    assert "key: value" in body["state_rendering"]["object_as_key_value_lines"]
    assert "not the training input" in body["state_rendering"]["object_as_key_value_lines"]
    assert body["limits"]["max_choice_options"] == 255
    assert body["limits"]["score_levels"] == [2, 10]


def test_the_order_marginalize_setting_is_gone():
    assert not hasattr(ServerSettings(), "order_marginalize")
    assert "order_marginalize" not in health_body(ServerSettings(), loaded=True)


def test_cli_lists_serve_and_no_longer_takes_order_marginalize(capsys):
    from sokudan.cli import main

    assert main(["--help"]) == 0
    assert "serve" in capsys.readouterr().out
    with pytest.raises(SystemExit) as exit_info:
        main(["serve", "--order-marginalize"])
    assert exit_info.value.code == 2  # argparse: unrecognised argument


@pytest.mark.parametrize(("flag", "env", "expected"), [
    (None, None, {}),                                  # load()'s default: shipped bool
    ("none", None, {"temperatures": None}),
    ("OFF", None, {"temperatures": None}),
    ("t.json", None, {"temperatures": "t.json"}),
    (None, "none", {"temperatures": None}),
    (None, "x.json", {"temperatures": "x.json"}),
    ("t.json", "none", {"temperatures": "t.json"}),    # the flag wins
])
def test_temperatures_setting(monkeypatch, flag, env, expected):
    from sokudan.serve.systemone import temperatures_setting

    if env is None:
        monkeypatch.delenv("SOKUDAN_TEMPERATURES", raising=False)
    else:
        monkeypatch.setenv("SOKUDAN_TEMPERATURES", env)
    assert temperatures_setting(flag) == expected


def test_load_agent_reports_what_was_actually_loaded(monkeypatch):
    import sokudan
    from sokudan.serve.systemone import load_agent

    seen = {}

    def fake_load(ref, **kwargs):
        seen.update(kwargs)
        return _Loaded()

    monkeypatch.delenv("SOKUDAN_TEMPERATURES", raising=False)
    monkeypatch.setattr(sokudan, "load", fake_load)
    settings = ServerSettings(model_ref="test/fake")
    load_agent(settings)
    assert "temperatures" not in seen          # load()'s own default
    assert (seen["backend"], seen["dtype"]) == ("auto", None)
    assert (settings.backend, settings.device, settings.dtype) == ("mlx", "gpu", "float16")
    assert settings.calibrated is True
    assert health_body(settings, loaded=True)["calibration"]["temperatures"] == {"bool/2": 2.07}


class _Loaded:
    """What `sokudan.load` returns, as far as the server reads it."""

    class backend:  # noqa: N801 - stands in for the Backend object's `.name`
        name = "mlx"

    device = "gpu"
    dtype = "float16"
    temperatures = {("bool", 2): 2.07}


@pytest.mark.parametrize(("argv", "expected"), [
    ([], ("auto", "auto", None)),
    (["--backend", "torch", "--device", "mps"], ("torch", "mps", None)),
    (["--backend", "mlx", "--dtype", "float32"], ("mlx", "auto", "float32")),
])
def test_serve_passes_backend_and_dtype_and_logs_the_choice(monkeypatch, capsys, argv,
                                                            expected):
    import sys
    import types

    import sokudan
    from sokudan.serve.systemone import main

    seen = {}

    def fake_load(ref, **kwargs):
        seen.update(kwargs)
        return _Loaded()

    monkeypatch.delenv("SOKUDAN_TEMPERATURES", raising=False)
    monkeypatch.setattr(sokudan, "load", fake_load)
    monkeypatch.setitem(sys.modules, "uvicorn",
                        types.SimpleNamespace(run=lambda app, host, port: None))
    assert main(argv) == 0
    assert (seen["backend"], seen["device"], seen["dtype"]) == expected
    assert "backend=mlx device=gpu dtype=float16" in capsys.readouterr().out


def test_serve_rejects_an_unknown_backend():
    from sokudan.serve.systemone import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--backend", "jax"])
    assert exit_info.value.code == 2


def test_cli_serve_help_names_the_default_model(capsys):
    from sokudan.cli import main

    with pytest.raises(SystemExit) as exit_info:
        main(["serve", "--help"])
    assert exit_info.value.code == 0
    assert "GeneLab/sokudan-ja-310m" in capsys.readouterr().out
