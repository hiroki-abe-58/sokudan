"""Namespace availability check (SOKUDAN_SPEC.md §0).

Reports raw HTTP status codes. It does not guess: a 404 on an unauthenticated
request means "not publicly visible", which is not the same as "reserved by nobody".

Usage:
    uv run python scripts/check_namespaces.py
"""

from __future__ import annotations

import json
import sys
import time

import httpx

PACKAGE = "sokudan"
GITHUB_REPO = "hiroki-abe-58/sokudan"
HF_MODEL = "hiroki-abe-58/sokudan-ja-310m"

CHECKS = [
    ("PyPI JSON API", f"https://pypi.org/pypi/{PACKAGE}/json"),
    ("PyPI simple index", f"https://pypi.org/simple/{PACKAGE}/"),
    ("GitHub repo", f"https://api.github.com/repos/{GITHUB_REPO}"),
    ("HF model", f"https://huggingface.co/api/models/{HF_MODEL}"),
]


def main() -> int:
    print(f"namespace check  {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
    with httpx.Client(timeout=20.0, follow_redirects=True) as client:
        for label, url in CHECKS:
            try:
                response = client.get(url)
                status = response.status_code
                verdict = {
                    404: "free (not publicly visible)",
                    200: "TAKEN",
                    401: "unauthenticated request rejected -- inconclusive",
                }.get(status, "inconclusive")
                print(f"{label:22s} HTTP {status:3d}  {verdict}\n{'':22s} {url}")
            except httpx.HTTPError as exc:
                print(f"{label:22s} ERROR {type(exc).__name__}: {exc}")

        print("\n-- HF public search for 'sokudan' --")
        try:
            hits = client.get(
                "https://huggingface.co/api/models",
                params={"search": PACKAGE, "limit": 20},
            ).json()
            print(json.dumps([h.get("id") for h in hits], ensure_ascii=False)
                  if hits else "no public models match 'sokudan'")
        except httpx.HTTPError as exc:
            print(f"ERROR {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
