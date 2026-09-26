# Clean-install check, not a serving image.
#
# The 0.1.1 release existed because `pip install sokudan` followed by `predict()` raised
# ModuleNotFoundError on scipy: the development venv had scipy pulled in by an optional
# extra, so nothing on this machine could see the break. The same gap reappeared in the
# calibration path and was only caught when a `uv sync` happened to prune the extra.
#
# So the check is the point: install from source with *core dependencies only*, into an
# image that has nothing else, and run the paths the README and the model card tell
# people to run. Anything that needs an extra must fail here, loudly.
#
#   docker build -t sokudan-smoke .
#   docker run --rm sokudan-smoke
#
# CPU torch, because this proves the dependency closure rather than the kernels.

FROM python:3.11-slim

# git: hatchling reads the working tree, not a wheel.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Torch first and from the CPU index, so the default PyPI CUDA wheel is never pulled.
# Pinned to the same major the project develops against.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu "torch>=2.8"

COPY pyproject.toml README.md ./
COPY sokudan ./sokudan

# No extras. `[dev]`, `[serve]` and `[bench]` are deliberately absent: a smoke test that
# installs them cannot see the failure it exists to catch.
RUN pip install --no-cache-dir .

COPY scripts/smoke_install.py ./

# HF_HUB_OFFLINE is not set: the tokenizer check needs the hub once. Everything after it
# runs on what the install brought.
ENTRYPOINT ["python", "smoke_install.py"]
