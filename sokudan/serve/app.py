"""`POST /v1/systemone` (SOKUDAN_SPEC.md §10).

    uv run uvicorn sokudan.serve.app:app --port 8000
    SOKUDAN_CHECKPOINT=runs/v01_seed0/model.pt SOKUDAN_TEMPERATURES=runs/v01_seed0/temperatures.json

The schema is sokudan's own, and it is the one the Quickstart already used:

    request   model, state, questions{ key: {type, instructions, criteria} }
              type is one of choice | score | bool (alias noul)
    response  model, answers{ key: {choice | score | noul, probabilities,
                                    confidence} }, usage

Everything the endpoint returns comes from `Agent.predict`. The server adds
transport, not semantics.

**`noul` is accepted and returned, `bool` is accepted too.** `sokudan.schema.question`
has had the alias since Phase 1; it resolves in one place so nothing downstream knows
about it.

**Where the probabilities come from.** They are the decision head's softmax over
marker logits, optionally divided by a fitted temperature. They are not a number the
model wrote out as text and they are not self-reported. `docs/architecture.md` has
the mechanism.

**Calibration is opt-in and its absence is reported.** Without
`SOKUDAN_TEMPERATURES` the probabilities are raw head outputs; `calibrated: false`
appears in every response and `/ready` says so too. A client cannot accidentally
treat uncalibrated numbers as calibrated ones without the response saying otherwise.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

MAX_QUESTIONS = int(os.environ.get("SOKUDAN_MAX_QUESTIONS", "64"))
MAX_STATE_CHARS = int(os.environ.get("SOKUDAN_MAX_STATE_CHARS", "100000"))
MAX_CONCURRENCY = int(os.environ.get("SOKUDAN_MAX_CONCURRENCY", "4"))
MAX_QUEUE = int(os.environ.get("SOKUDAN_MAX_QUEUE", "32"))

QuestionType = Literal["choice", "score", "bool", "noul"]


class Question(BaseModel):
    type: QuestionType
    instructions: str = Field(min_length=1)
    criteria: dict[str, str] | list[str] | None = None

    @model_validator(mode="after")
    def _criteria_shape(self) -> Question:
        """`criteria` is required for choice and score, and its shape differs.

        Checked on the model rather than on the field: a field validator does not
        run when the field is absent, so a choice question with no criteria passed
        validation and only failed later, inside the model call.
        """
        if self.type == "choice":
            if not isinstance(self.criteria, dict) or not self.criteria:
                raise ValueError(
                    "a choice question needs criteria as a non-empty object of "
                    "option -> description"
                )
        elif self.type == "score":
            if not isinstance(self.criteria, list) or len(self.criteria) < 2:
                raise ValueError(
                    "a score question needs criteria as an ordered list of at least "
                    "two levels"
                )
        return self


class SystemOneRequest(BaseModel):
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    model: str | None = None

    @field_validator("questions")
    @classmethod
    def _question_budget(cls, value: dict[str, Question]) -> dict[str, Question]:
        if len(value) > MAX_QUESTIONS:
            raise ValueError(f"at most {MAX_QUESTIONS} questions per request, got {len(value)}")
        return value


class _State:
    """Process-wide model handle. Loaded once at startup, never per request."""

    agent: Any | None = None
    checkpoint: str | None = None
    calibrated: bool = False
    load_error: str | None = None
    semaphore: asyncio.Semaphore | None = None
    inflight: int = 0


state = _State()


def load_agent() -> None:
    checkpoint = os.environ.get("SOKUDAN_CHECKPOINT")
    if not checkpoint:
        state.load_error = "SOKUDAN_CHECKPOINT is not set"
        return
    try:
        import sokudan

        temperatures = os.environ.get("SOKUDAN_TEMPERATURES") or None
        state.agent = sokudan.load(checkpoint, temperatures=temperatures)
        state.checkpoint = checkpoint
        state.calibrated = bool(temperatures)
        state.load_error = None
    except Exception as exc:  # noqa: BLE001 -- surfaced through /ready, not swallowed
        state.load_error = f"{type(exc).__name__}: {exc}"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    load_agent()
    yield
    state.agent = None


app = FastAPI(
    title="sokudan",
    version="0.1.0",
    summary="Japanese System One decision model. Typed answers with probabilities, no generation.",
    lifespan=lifespan,
)


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    """Liveness. Says nothing about whether a model is loaded -- that is /ready."""
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> JSONResponse:
    if state.agent is None:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": state.load_error or "no model loaded"},
        )
    return JSONResponse(content={
        "status": "ready",
        "model": "sokudan-ja-310m",
        "checkpoint": state.checkpoint,
        "calibrated": state.calibrated,
        "calibration_note": (
            None if state.calibrated else
            "確率は較正されていません。SOKUDAN_TEMPERATURES を設定してください。"
        ),
        "limits": {
            "max_questions": MAX_QUESTIONS,
            "max_state_chars": MAX_STATE_CHARS,
            "max_concurrency": MAX_CONCURRENCY,
            "max_queue": MAX_QUEUE,
        },
    })


@app.post("/v1/systemone")
async def systemone(request: SystemOneRequest, http: Request) -> dict[str, Any]:
    if state.agent is None:
        raise HTTPException(status_code=503, detail=state.load_error or "no model loaded")

    text_length = len(request.state if isinstance(request.state, str) else str(request.state))
    if text_length > MAX_STATE_CHARS:
        raise HTTPException(
            status_code=413,
            detail=f"state is {text_length} characters; the limit is {MAX_STATE_CHARS}",
        )

    semaphore = state.semaphore
    assert semaphore is not None
    if state.inflight >= MAX_CONCURRENCY + MAX_QUEUE:
        # 429 rather than 529: the queue is full now, and the client may retry.
        raise HTTPException(
            status_code=429,
            detail=f"queue full ({state.inflight} in flight); retry after a moment",
            headers={"Retry-After": "1"},
        )

    questions = {
        key: question.model_dump(exclude_none=True)
        for key, question in request.questions.items()
    }

    state.inflight += 1
    started = time.perf_counter()
    try:
        async with semaphore:
            if await http.is_disconnected():
                raise HTTPException(status_code=499, detail="client disconnected")
            try:
                # The model call is synchronous and GPU-bound; a thread keeps the
                # event loop free to serve /healthz while it runs.
                result = await asyncio.to_thread(
                    state.agent.predict, request.state, questions
                )
            except (ValueError, TypeError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        state.inflight -= 1

    result["calibrated"] = state.calibrated
    if not state.calibrated:
        result["calibration_note"] = (
            "確率は較正されていません。README の Limits を参照してください。"
        )
    result["timing"] = {"total_ms": round((time.perf_counter() - started) * 1000, 2)}
    return result
