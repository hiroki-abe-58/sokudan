"""Environment-backed configuration.

Rule: no API keys or endpoints in code. Everything comes from environment variables
(see `.env.example`). `.env` is loaded if present but real environment variables win.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent

# Real env vars take precedence over .env (override=False).
load_dotenv(REPO_ROOT / ".env", override=False)

# A key left blank in .env becomes an empty string in the environment, and some
# libraries treat "set but empty" as "use this value". huggingface_hub, for one,
# builds the header `Bearer ` from an empty HF_TOKEN and the request dies with
# `LocalProtocolError: Illegal header value`. Unset rather than empty is what these
# libraries actually mean by "no credential".
for _name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HOME"):
    if _name in os.environ and not os.environ[_name].strip():
        del os.environ[_name]


class MissingConfig(RuntimeError):
    """A required environment variable is unset."""


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingConfig(
            f"{name} is not set. Copy .env.example to .env and fill it in, "
            f"or export {name} in your shell."
        )
    return value


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class LocalLLMConfig:
    """An OpenAI-compatible local endpoint (vLLM / Ollama / LM Studio / llama.cpp)."""

    base_url: str
    model: str
    api_key: str

    @classmethod
    def from_env(cls) -> LocalLLMConfig:
        return cls(
            base_url=_require("SOKUDAN_LOCAL_LLM_BASE_URL").rstrip("/"),
            model=_require("SOKUDAN_LOCAL_LLM_MODEL"),
            api_key=_get("SOKUDAN_LOCAL_LLM_API_KEY", "local") or "local",
        )


BACKBONE_MODEL_ID = "sbintuitions/modernbert-ja-310m"
