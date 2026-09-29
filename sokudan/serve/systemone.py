"""A `/v1/systemone`-compatible server for sokudan.

    sokudan serve --model GeneLab/sokudan-ja-310m --port 8000

Any client that speaks TypeSafe's `/v1/systemone` wire format can point its base URL
here. `docs/serving.md` is the user guide; `docs/systemone_wire_format.md` is the
field-by-field table of the format and what sokudan does with each field.

Relation to `sokudan.serve.app`: that module is the earlier server, configured by
environment variables and speaking sokudan's own response shape (`type: "bool"`,
top-probability `confidence`, sokudan's usage block). This one speaks the wire shape
and is what `sokudan serve` runs. Both call the same `Agent.predict`.

**One call per request.** Every question in a request goes to `Agent.predict` in a
single call. What that costs is `predict`'s business: with joint encoding the
backbone still runs once per question, and the response reports how many passes it
took (`sokudan.backbone_passes`).

**Authorization.** A bearer header is accepted and not checked. This is a local
server; put it behind something that authenticates before exposing it.

**Calibration** (v0.2.1). The server loads the model the way `sokudan.load` does by
default: the bool temperatures of the `calibration.json` shipped beside the weights
(score and choice stay raw; docs/calibration.md). `--temperatures none` (or `off`, or
the environment variable `SOKUDAN_TEMPERATURES=none`) serves the raw probabilities; a
path applies that file. `/health` says whether the loaded model has any temperature and
which; each response says whether a temperature was applied to any of its answers
(`sokudan.calibrated`) and to which (`sokudan.calibrated_answers`).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from sokudan.schema.question import parse_questions
from sokudan.serve import wire

MODEL_NAME = "sokudan-ja-310m"
DEFAULT_MODEL_REF = "GeneLab/sokudan-ja-310m"

STATE_RENDERING = {
    "text": "a string state reaches the model unchanged (the input the model was "
            "trained and evaluated on)",
    "object_as_key_value_lines": "an object state is rendered as one `key: value` line "
                                 "per top-level key before encoding; this is not the "
                                 "training input and changes the output (README, "
                                 "Limits). Send the text as a string where you can.",
    "array_as_lines": "an array state is rendered one item per line; "
                      "{role, content} items as `role: content`",
}


@dataclass
class ServerSettings:
    model_ref: str = DEFAULT_MODEL_REF
    device: str | None = None
    # what was asked for before loading (dtype None: the backend's default), what was
    # chosen after
    backend: str = "auto"
    dtype: str | None = None
    calibrated: bool = False
    max_state_chars: int = 100_000
    # 1 by default: the question-encoding cache inside `Agent` is a plain dict.
    max_concurrency: int = 1
    max_queue: int = 32
    # the loaded model's temperatures, as "type/K" -> T (empty: uncalibrated)
    temperatures: dict[str, float] | None = None


def health_body(settings: ServerSettings, loaded: bool) -> dict[str, Any]:
    import sokudan

    return {
        "status": "ok" if loaded else "not_ready",
        "model": MODEL_NAME,
        "model_ref": settings.model_ref,
        "device": settings.device,
        "sokudan_version": sokudan.__version__,
        "calibrated": settings.calibrated,
        "calibration": {
            "temperatures": dict(settings.temperatures or {}),
            "note": "v0.2.1 default: bool temperatures only (the calibration.json beside "
                    "the weights); score and choice are raw. --temperatures none serves "
                    "raw probabilities.",
        },
        "state_rendering": STATE_RENDERING,
        "noul_criteria": "rendered into the yes/no marker text as `はい: <true>` / "
                         "`いいえ: <false>`; effect on accuracy not measured",
        "confidence": {
            "choice": "(p_max - 1/K) / (1 - 1/K), clipped to [0, 1]",
            "score": "max(0, 1 - E|level - mode| / D), D = mean distance of a uniform "
                     "distribution over the levels from its middle",
        },
        "auth": "none; an Authorization header is accepted and not checked",
        "limits": {
            "max_questions": wire.MAX_QUESTIONS,
            "max_choice_options": wire.MAX_CHOICE_OPTIONS,
            "score_levels": [wire.MIN_SCORE_LEVELS, wire.MAX_SCORE_LEVELS],
            "max_state_chars": settings.max_state_chars,
            "max_concurrency": settings.max_concurrency,
            "max_queue": settings.max_queue,
        },
    }


def create_app(agent: Any, settings: ServerSettings | None = None) -> FastAPI:
    """Build the app around an already-loaded agent (anything with `.predict`)."""
    settings = settings or ServerSettings()

    import sokudan

    app = FastAPI(
        title="sokudan",
        version=sokudan.__version__,
        summary="/v1/systemone-compatible server for sokudan-ja-310m. "
                "Typed answers with probabilities, no generation.",
    )
    semaphore = asyncio.Semaphore(settings.max_concurrency)
    inflight = {"n": 0}

    @app.get("/health")
    async def health() -> JSONResponse:
        loaded = agent is not None
        return JSONResponse(status_code=200 if loaded else 503,
                            content=health_body(settings, loaded))

    @app.post("/v1/systemone")
    async def systemone(request: wire.SystemOneRequest) -> dict[str, Any]:
        if agent is None:
            raise HTTPException(status_code=503, detail="no model loaded")

        size = len(request.state if isinstance(request.state, str)
                   else json.dumps(request.state, ensure_ascii=False))
        if size > settings.max_state_chars:
            raise HTTPException(
                status_code=413,
                detail=f"state is {size} characters; the limit is {settings.max_state_chars}",
            )

        questions = wire.to_sokudan_questions(request)
        # Validate with the same parser `predict` uses, so a question sokudan cannot
        # read (two score levels that render to the same text, say) is a 422 with
        # the reason, not a 500 from inside the model call.
        try:
            parse_questions(questions)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if inflight["n"] >= settings.max_concurrency + settings.max_queue:
            raise HTTPException(
                status_code=429,
                detail=f"queue full ({inflight['n']} in flight); retry after a moment",
                headers={"Retry-After": "1"},
            )

        inflight["n"] += 1
        started = time.perf_counter()
        try:
            async with semaphore:
                # One call for every question in the request.
                result = await asyncio.to_thread(agent.predict, request.state, questions)
        finally:
            inflight["n"] -= 1

        # per response, from `Agent.predict` (v0.2.1); an agent without those fields
        # falls back to whether the loaded model has any temperature
        extension = {"calibrated": bool(result.get("calibrated", settings.calibrated)),
                     "calibrated_answers": list(result.get("calibrated_answers", []))}
        body = wire.to_wire_response(request, result, model_name=MODEL_NAME,
                                     extension=extension)
        body["sokudan"]["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return body

    return app


def temperatures_setting(value: str | None) -> dict[str, Any]:
    """`load` keyword arguments for `--temperatures` / `SOKUDAN_TEMPERATURES`: unset is
    `load`'s default (the shipped bool calibration), `none` / `off` is no calibration,
    anything else is a temperatures file."""
    if value is None:
        value = os.environ.get("SOKUDAN_TEMPERATURES") or None
    if value is None:
        return {}
    if value.strip().lower() in ("none", "off"):
        return {"temperatures": None}
    return {"temperatures": value}


def load_agent(settings: ServerSettings, temperatures: str | None = None) -> Any:
    import sokudan

    agent = sokudan.load(settings.model_ref, backend=settings.backend,
                         device=settings.device, dtype=settings.dtype,
                         **temperatures_setting(temperatures))
    settings.backend = agent.backend.name
    settings.device = agent.device
    settings.dtype = agent.dtype
    temps = getattr(agent, "temperatures", None) or {}
    settings.temperatures = {f"{kind}/{k}": float(t) for (kind, k), t in temps.items()}
    settings.calibrated = bool(temps)
    return agent


def main(argv: list[str] | None = None) -> int:
    """`sokudan serve`: load the model once, then serve `/v1/systemone`."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="sokudan serve",
        description="Serve sokudan on a /v1/systemone-compatible HTTP endpoint.",
    )
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address (default 127.0.0.1; 0.0.0.0 accepts other machines)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=DEFAULT_MODEL_REF,
                        help="Hub repo id (optionally @revision), a directory with "
                             "model.safetensors, or a model.pt "
                             f"(default {DEFAULT_MODEL_REF}, i.e. the Hub main: v0.2 "
                             "weights plus the v0.2.1 calibration.json)")
    parser.add_argument("--backend", default="auto", choices=["auto", "mlx", "torch"],
                        help="auto (the default: MLX on Apple silicon when mlx is "
                             "installed, else torch on mps, cuda or cpu), mlx or torch")
    parser.add_argument("--device", default="auto",
                        help="auto (the default), or a torch device: cpu, cuda or mps")
    parser.add_argument("--dtype", default=None,
                        help="the backend's default when omitted (MLX float16, torch "
                             "float32); MLX also takes float32")
    parser.add_argument("--temperatures", default=None,
                        help="default: the bool calibration shipped beside the weights "
                             "(score and choice stay raw); 'none' or 'off' for raw "
                             "probabilities; or a temperatures file. Also read from "
                             "SOKUDAN_TEMPERATURES")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--max-queue", type=int, default=32)
    args = parser.parse_args(argv)

    import uvicorn

    settings = ServerSettings(model_ref=args.model, device=args.device,
                              backend=args.backend, dtype=args.dtype,
                              max_concurrency=args.max_concurrency,
                              max_queue=args.max_queue)
    print(f"sokudan serve: loading {args.model} ...", flush=True)
    agent = load_agent(settings, temperatures=args.temperatures)
    print(f"sokudan serve: loaded; backend={settings.backend} device={settings.device} "
          f"dtype={settings.dtype} calibrated={settings.calibrated}", flush=True)
    uvicorn.run(create_app(agent, settings), host=args.host, port=args.port)
    return 0
