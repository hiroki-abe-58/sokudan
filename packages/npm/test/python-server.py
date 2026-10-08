"""Real HTTP/wire implementation with the existing Python tests' fake model."""

import runpy
import sys
from pathlib import Path

import uvicorn

from sokudan.serve.systemone import ServerSettings, create_app

root = Path(__file__).resolve().parents[3]
fake_agent = runpy.run_path(str(root / "tests/test_systemone_server.py"))["FakeAgent"]
app = create_app(fake_agent(), ServerSettings(model_ref="test/fake", device="cpu"))
uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[1]))
