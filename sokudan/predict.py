"""The three-line entry point (SOKUDAN_SPEC.md §10, §11).

    import sokudan
    agent = sokudan.load("runs/s0/model.pt")
    result = agent.predict({"body": "先月の請求が二重になっています"}, questions)

The response shape follows `laya`'s, which was read off an actual call rather than
guessed -- `{"answers": {qid: {...}}, "usage": {...}}`, with `choice`/`score`/`noul`
naming each primitive's answer. Matching a shape we verified by running the library
beats inventing one from a specification we cannot check.

What this does **not** share with that API is where the probabilities come from. They
are the head's logits read directly, not a model's own report of its confidence.

The forward pass itself runs in a backend (`sokudan.backends`); this module does not
import torch.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sokudan.backends import Backend, Batch
from sokudan.encoding.question import QuestionEncoderCache, encode_joint
from sokudan.schema.question import Question, is_ordered, parse_questions


@dataclass
class _Prepared:
    question: Question
    kind: str
    labels: list[str]


class Agent:
    """A loaded checkpoint that answers typed questions about a state.

    `model` is a `Backend`, or a torch model, which runs on `device` through the torch
    backend. `agent.backend` is the backend in use.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        device: str = "cuda",
        temperatures: dict[tuple[str, int], float] | None = None,
        max_state_tokens: int = 1024,
        encoding: str = "separate",
    ) -> None:
        if not isinstance(model, Backend):
            from sokudan.backends.torch_backend import TorchBackend

            model = TorchBackend(model, device, encoding=encoding)
        self.backend = model
        self.tokenizer = tokenizer
        self.temperatures = temperatures or {}
        self.max_state_tokens = max_state_tokens
        self.cache = QuestionEncoderCache()
        self.encoding = encoding
        if encoding not in ("separate", "joint"):
            raise ValueError(f"unknown encoding {encoding!r}")

    @property
    def model(self) -> object:
        """The backend's model (a torch module or an MLX module)."""
        return self.backend.model

    @property
    def device(self) -> str:
        return self.backend.device

    @property
    def dtype(self) -> str:
        return self.backend.dtype

    @staticmethod
    def _state_text(state: str | dict[str, Any] | list[Any]) -> str:
        """Accept a string, a JSON-ish dict, or a conversation, like `laya` does."""
        if isinstance(state, str):
            return state
        if isinstance(state, dict):
            return "\n".join(f"{key}: {value}" for key, value in state.items())
        if isinstance(state, list):
            parts = []
            for turn in state:
                if isinstance(turn, dict):
                    parts.append(
                        f"{turn.get('role', '')}: {turn.get('content', turn)}".strip(": ")
                    )
                else:
                    parts.append(str(turn))
            return "\n".join(parts)
        raise TypeError(f"unsupported state type {type(state).__name__}")

    def predict(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: dict[str, dict[str, Any] | Question],
        *,
        order_marginalize: bool | Mapping[str, bool] = False,
    ) -> dict[str, Any]:
        """Answer every question about one state in a single pass per question batch.

        `order_marginalize=True` (joint encoding only; off by default, which keeps the
        single pass) asks each question again with its options reordered -- the K cyclic
        shifts of a `choice`, the original and reversed levels of a `score`, the two slot
        orders of a `bool` -- maps every answer back to the original order and averages
        the probabilities before any temperature (`sokudan.order_marginalize`,
        docs/order_marginalization.md). All the reorderings go through the backbone in one
        batch, one row each. A mapping picks the types, e.g.
        `{"choice": False, "score": True, "bool": False}` (types left out are off); the
        other types are asked once, in the same batch. All off is the single pass.
        """
        from sokudan.calibration.temperature import apply_temperature

        if not questions:
            raise ValueError("no questions given")

        parsed = parse_questions(questions)
        prepared = {
            qid: _Prepared(question, question.type, list(question.labels))
            for qid, question in parsed.items()
        }

        from sokudan.order_marginalize import marginalized_types

        text = self._state_text(state)
        order = list(prepared)
        types = marginalized_types(order_marginalize)
        if types:
            if self.encoding != "joint":
                raise NotImplementedError("order_marginalize needs the joint encoding")
            from sokudan.order_marginalize import combine, variants

            expanded: dict[str, _Prepared] = {}
            perms: dict[str, list[tuple[str, list[int]]]] = {}
            for qid in order:
                item = prepared[qid]
                if item.kind not in types:
                    expanded[qid] = item
                    continue
                for i, (question, perm) in enumerate(variants(item.question)):
                    key = f"{qid}\x1f{i}"
                    expanded[key] = _Prepared(question, item.kind, list(item.labels))
                    perms.setdefault(qid, []).append((key, perm))
            rows, n, state_tokens, truncated, question_tokens = self._forward(text, expanded)
            raw = {qid: (combine([rows[key] for key, _ in perms[qid]],
                                 [perm for _, perm in perms[qid]])
                         if qid in perms else rows[qid])
                   for qid in order}
        else:
            raw, n, state_tokens, truncated, question_tokens = self._forward(text, prepared)

        answers: dict[str, Any] = {}
        calibrated: list[str] = []
        for qid in order:
            item = prepared[qid]
            k = len(item.labels)
            row_probs = raw[qid]
            temperature = self.temperatures.get((item.kind, k), 1.0)
            if temperature != 1.0:
                row_probs = apply_temperature(row_probs.reshape(1, -1), temperature)[0]
                calibrated.append(qid)

            entry: dict[str, Any] = {"type": item.kind}
            if item.kind == "score":
                entry["score"] = float((row_probs * range(k)).sum())
                entry["probabilities"] = {str(i): round(float(p), 4)
                                          for i, p in enumerate(row_probs)}
                entry["legend"] = {str(i): label for i, label in enumerate(item.labels)}
            elif item.kind == "bool":
                entry["noul"] = round(float(row_probs[1]), 4)
            else:
                best = int(row_probs.argmax())
                entry["choice"] = item.labels[best]
                entry["probabilities"] = {label: round(float(p), 4)
                                          for label, p in zip(item.labels, row_probs,
                                                              strict=True)}
            if item.kind != "bool":
                entry["confidence"] = round(float(row_probs.max()), 4)
            answers[qid] = entry

        usage = {
            "state_tokens": state_tokens,
            "state_truncated": truncated,
            "question_tokens": max(question_tokens, 0),
            "backbone_passes": n if self.encoding == "joint" else 2,
            "output_tokens": 0,
        }
        if types:
            usage["order_marginalized"] = (True if types == {"choice", "score", "bool"}
                                           else sorted(types))
        return {
            "model": "sokudan-ja-310m",
            "answers": answers,
            "encoding": self.encoding,
            # v0.2.1 (docs/calibration.md §10): whether a temperature was applied to any
            # answer, and to which (by default only bool answers are calibrated).
            "calibrated": bool(calibrated),
            "calibrated_answers": calibrated,
            "usage": usage,
        }



    def _forward(self, text: str, prepared: dict[str, _Prepared]):
        """One batch through the model: `({qid: probabilities over its K options}, rows,
        state tokens, truncated, question tokens)`."""
        from sokudan.encoding.state import encode_state

        order = list(prepared)
        n = len(order)
        input_order = self.backend.input_order

        encoded_state = None
        if self.encoding == "joint":
            # v0.1. The state is re-encoded per question, so there is nothing to
            # cache and latency grows with the number of questions
            # (`docs/architecture.md` §1.2).
            encoded = [
                encode_joint(prepared[qid].question, text, self.tokenizer,
                             max_tokens=self.max_state_tokens,
                             input_order=input_order)
                for qid in order
            ]
            state_tokens = max((e.n_state_tokens for e in encoded), default=0)
            truncated = any(e.truncated for e in encoded)
        else:
            encoded_state = encode_state(
                text, self.tokenizer, max_tokens=self.max_state_tokens
            )
            encoded = [self.cache.get(prepared[qid].question, self.tokenizer)
                       for qid in order]
            state_tokens = encoded_state.n_tokens
            truncated = encoded_state.truncated

        batch = Batch.build(
            encoded, [is_ordered(prepared[qid].question) for qid in order],
            self.tokenizer.pad_token_id, sandwich=input_order == "sandwich",
            state=encoded_state,
        )
        probs = self.backend.probs(batch)
        question_tokens = int(batch.attention_mask.sum())
        if self.encoding == "joint":
            question_tokens -= state_tokens * n
        rows = {qid: probs[row, :len(prepared[qid].labels)]
                for row, qid in enumerate(order)}
        return rows, n, state_tokens, truncated, question_tokens



def locate_checkpoint(checkpoint: str | Path) -> Path:
    """Accept a `.pt` file, a directory of safetensors, or a Hub repo id.

    The published weights are safetensors in a repository, and the training script
    writes a `.pt`. Both have to load through one entry point, or the Quickstart in
    the README describes something the package cannot do.

    Returns the `.pt` file or the directory holding `model.safetensors` (downloaded
    from the Hub if it was a repo id). Reading the weights is the backend's job.
    """
    path = Path(str(checkpoint))

    if path.is_file():
        return path

    if path.is_dir():
        if not (path / "model.safetensors").exists():
            raise FileNotFoundError(f"{path} has no model.safetensors")
        return path

    # Not on disk: treat it as `repo_id` or `repo_id@revision`.
    repo_id, _, revision = str(checkpoint).partition("@")
    if "/" not in repo_id:
        raise FileNotFoundError(
            f"{checkpoint!r} is neither a file, a directory, nor a Hub repo id "
            f"(expected something like 'GeneLab/sokudan-ja-310m')"
        )
    from huggingface_hub import snapshot_download

    local = Path(snapshot_download(
        repo_id, revision=revision or None,
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"],
    ))
    return locate_checkpoint(local)


def read_temperatures(path: str | Path) -> dict[tuple[str, int], float]:
    """`{(type, option count): T}` from a temperatures file: `temperatures.json` from
    `scripts/calibrate.py` or `calibration.json` from `scripts/calibration_heldout.py`
    (docs/calibration.md) -- both hold `{"temperatures": {"bool/2": T, ...}}`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    parsed: dict[tuple[str, int], float] = {}
    for key, value in data["temperatures"].items():
        kind, _, count = key.partition("/")
        parsed[(kind, int(count))] = float(value)
    return parsed


DEFAULT_CALIBRATION = "calibration.json"
"""v0.2.1: the calibration file `load` looks for beside the checkpoint by default. Only its
bool temperatures are applied by default (docs/calibration.md §8-§10)."""


def resolve_temperatures(
    temperatures: str | Path | dict[tuple[str, int], float] | None,
    resolved_dir: Path | None,
) -> dict[tuple[str, int], float]:
    """What `load` applies.

    - `DEFAULT_CALIBRATION` (the default): the `calibration.json` beside the checkpoint
      (the `.pt` file's directory, the safetensors directory, or the Hub download), bool
      temperatures only; no file there means no calibration.
    - `None`: no calibration (the raw head probabilities).
    - a bare file name: that file beside the checkpoint, else the path as given; any other
      path: that file (it must exist); a dict: as given. All of these are applied whole.
    """
    if temperatures is None:
        return {}
    if isinstance(temperatures, dict):
        return dict(temperatures)
    default = str(temperatures) == DEFAULT_CALIBRATION
    path = Path(temperatures)
    if (resolved_dir is not None and not path.exists()
            and path.name == str(temperatures)):
        # A bare name beside a Hub / directory / .pt checkpoint: the file that came with
        # the weights rather than the working directory's.
        candidate = resolved_dir / str(temperatures)
        if candidate.exists():
            path = candidate
    if default:
        if not path.exists():
            return {}
        return {k: t for k, t in read_temperatures(path).items() if k[0] == "bool"}
    return read_temperatures(path)


BACKENDS = ("auto", "mlx", "torch")

SELF_CHECK_STATE = "先月の請求が二重になっています。"
SELF_CHECK_QUESTIONS = {
    "choice": {"type": "choice", "instructions": "担当部署は",
               "criteria": {"請求": "支払い", "技術": "不具合", "その他": "上記以外"}},
    "score": {"type": "score", "instructions": "緊急度は", "criteria": ["低", "中", "高"]},
    "bool": {"type": "bool", "instructions": "返金を求めているか"},
}
"""The short request `load` answers once before returning an agent (every head runs)."""


def is_apple_silicon() -> bool:
    import platform

    return platform.system() == "Darwin" and platform.machine() == "arm64"


def _mlx_importable() -> bool:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return False
    return True


def _candidates(backend: str, device: str | None) -> Iterator[tuple[str, str]]:
    """`(backend, device)` pairs to try, in order. `auto` with `device="auto"`: mlx (if it
    imports, on Apple silicon), then torch on mps, cuda, cpu. An explicit device is a
    torch device."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    explicit_device = device not in (None, "auto")
    if backend == "mlx":
        if explicit_device and device != "gpu":
            raise ValueError(f"the MLX backend runs on the default MLX device, not {device!r}")
        yield "mlx", "gpu"
        return
    if backend == "torch" or explicit_device:
        yield "torch", device or "auto"
        return
    if is_apple_silicon() and _mlx_importable():
        yield "mlx", "gpu"
    try:
        import torch
    except ImportError:
        return
    if torch.backends.mps.is_available():
        yield "torch", "mps"
    if torch.cuda.is_available():
        yield "torch", "cuda"
    yield "torch", "cpu"


def _load_backend(name: str, path: Path, device: str, dtype: str | None) -> Backend:
    if name == "mlx":
        from sokudan.backends.mlx import DTYPES, MLXBackend

        if dtype is not None and dtype not in DTYPES:
            raise ValueError(f"the MLX backend's dtype is one of {DTYPES}, got {dtype!r}")
        return MLXBackend.load(path, dtype=dtype)
    from sokudan.backends.torch_backend import TorchBackend

    return TorchBackend.load(path, device=device, dtype=dtype)


def self_check(agent: Agent) -> None:
    """Answer `SELF_CHECK_QUESTIONS` once; raise if that fails or gives a non-finite
    probability."""
    import math

    answers = agent.predict(SELF_CHECK_STATE, SELF_CHECK_QUESTIONS)["answers"]
    values = [answers["bool"]["noul"], answers["score"]["score"],
              *answers["choice"]["probabilities"].values(),
              *answers["score"]["probabilities"].values()]
    if not all(math.isfinite(v) for v in values):
        raise FloatingPointError(f"self-check gave non-finite probabilities: {answers}")


def load(
    checkpoint: str | Path,
    *,
    backend: str = "auto",
    device: str | None = "auto",
    dtype: str | None = None,
    temperatures: str | Path | dict[tuple[str, int], float] | None = DEFAULT_CALIBRATION,
) -> Agent:
    """Load a checkpoint, from disk or from the Hub.

    Args:
        checkpoint: a local `model.pt`, a local directory holding
            `model.safetensors` + `config.json`, or a Hub repo id such as
            `GeneLab/sokudan-ja-310m`. A repo id may carry a revision after `@`
            (`GeneLab/sokudan-ja-310m@seed1`).
        backend: `"auto"` (the default) tries MLX (when `mlx` imports on Apple
            silicon), then torch on mps, cuda and cpu, and uses the first that loads
            and answers a short self-check request; a failure is a warning and the next
            one is tried. `"mlx"` or `"torch"` uses that backend and raises on failure.
            `agent.backend` says which one was chosen.
        device: `"auto"` (the default). With the torch backend it picks cuda, then
            mps, then cpu. `"cpu"`, `"cuda"`, `"mps"` (or any torch device string) are
            torch devices and are used as given, with `backend="auto"` too.
        dtype: `None` is the backend's default. torch: `"float32"` only. MLX:
            `sokudan.backends.mlx.DTYPES` (the backbone's precision; the heads run in
            float32).
        temperatures: by default (v0.2.1) the `calibration.json` shipped beside the
            checkpoint, **bool temperatures only** -- score and choice stay raw
            (docs/calibration.md §10). `None` turns calibration off (the raw head
            probabilities, as in v0.2). A path to a `temperatures.json` /
            `calibration.json`, or a dict, applies that file or dict whole. Each
            response says whether a temperature was applied (`calibrated`).
    """
    import warnings

    from transformers import AutoTokenizer

    path = locate_checkpoint(checkpoint)
    parsed = resolve_temperatures(temperatures, path.parent if path.is_file() else path)
    fallback = backend == "auto" and device in (None, "auto")
    failures: list[str] = []
    for name, where in _candidates(backend, device):
        try:
            runner = _load_backend(name, path, where, dtype)
            agent = Agent(runner, AutoTokenizer.from_pretrained(runner.backbone_id),
                          temperatures=parsed, encoding=runner.encoding)
            self_check(agent)
            return agent
        except Exception as exc:
            if not fallback:
                raise
            failures.append(f"{name} ({where}): {type(exc).__name__}: {exc}")
            warnings.warn(f"sokudan.load: {failures[-1]}; trying the next backend",
                          RuntimeWarning, stacklevel=2)
    raise RuntimeError("no backend could load the checkpoint: "
                       + ("; ".join(failures) or "none available"))
