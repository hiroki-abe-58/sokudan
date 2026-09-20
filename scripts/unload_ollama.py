"""Free the local LLM's VRAM before training (the generation -> unload -> train order).

    uv run python scripts/unload_ollama.py

The generator holds ~18GB resident. Training a 310M model with room for activations
wants that back, and the two competing is how a training run turns into an OOM or a
silent slowdown. Ollama unloads a model when it is asked for with `keep_alive: 0`.

Verifies by reading `/api/ps` afterwards rather than assuming the request worked.
"""

from __future__ import annotations

import os
import sys
import time

import httpx


def main() -> int:
    base = os.environ.get("SOKUDAN_LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    root = base.rsplit("/v1", 1)[0]

    with httpx.Client(timeout=120.0) as client:
        try:
            loaded = client.get(f"{root}/api/ps").json().get("models", [])
        except httpx.HTTPError as exc:
            print(f"ollama is not reachable at {root}: {type(exc).__name__}: {exc}")
            print("Nothing to unload.")
            return 0

        if not loaded:
            print("No model is resident. Nothing to unload.")
            return 0

        for entry in loaded:
            name = entry.get("name") or entry.get("model")
            size_gb = entry.get("size_vram", entry.get("size", 0)) / 1024**3
            print(f"unloading {name} ({size_gb:.1f} GiB VRAM)...")
            client.post(f"{root}/api/generate", json={"model": name, "keep_alive": 0})

        for _ in range(30):
            time.sleep(1.0)
            still = client.get(f"{root}/api/ps").json().get("models", [])
            if not still:
                print("all models unloaded")
                return 0
        print("WARNING: a model is still resident:", still, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
