"""One client for the local, OpenAI-compatible LLM endpoint.

Used by every stage that needs a local LLM: `bench_ja` generation (§4.2), the
LLM-as-classifier baseline (§9), and the synthetic training builders (§7.4b).
Keeping it in one place is the same DRY rule that applies to encoding (§1-3, §2).

Endpoint and model come from the environment (`sokudan.config`). No keys in code.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import httpx

from sokudan.config import LocalLLMConfig


class LocalLLMError(RuntimeError):
    """The endpoint refused, timed out, or returned something unusable."""


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LocalLLM:
    """Minimal async chat client. Retries on transport errors and 5xx / 429."""

    def __init__(
        self,
        config: LocalLLMConfig | None = None,
        *,
        concurrency: int = 4,
        timeout_s: float = 300.0,
        max_retries: int = 4,
    ) -> None:
        self.config = config or LocalLLMConfig.from_env()
        self._sem = asyncio.Semaphore(concurrency)
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> LocalLLM:
        self._client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self._timeout_s,
            headers={"Authorization": f"Bearer {self.config.api_key}"},
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int = 1024,
        seed: int | None = None,
    ) -> ChatResult:
        if self._client is None:
            raise LocalLLMError("LocalLLM used outside its async context manager")

        payload: dict[str, object] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed

        last_error: Exception | None = None
        async with self._sem:
            for attempt in range(self._max_retries):
                try:
                    loop = asyncio.get_running_loop()
                    started = loop.time()
                    response = await self._client.post("/chat/completions", json=payload)
                    if response.status_code in (429, 500, 502, 503, 504):
                        raise LocalLLMError(f"HTTP {response.status_code}")
                    response.raise_for_status()
                    body = response.json()
                    usage = body.get("usage") or {}
                    return ChatResult(
                        text=body["choices"][0]["message"]["content"] or "",
                        prompt_tokens=int(usage.get("prompt_tokens", 0)),
                        completion_tokens=int(usage.get("completion_tokens", 0)),
                        latency_s=loop.time() - started,
                    )
                except (httpx.HTTPError, LocalLLMError, KeyError, ValueError) as exc:
                    last_error = exc
                    if attempt == self._max_retries - 1:
                        break
                    await asyncio.sleep(2.0**attempt)

        raise LocalLLMError(
            f"chat failed after {self._max_retries} attempts: {type(last_error).__name__}: "
            f"{last_error}"
        )
