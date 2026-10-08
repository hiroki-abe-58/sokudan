"""Verify a real CPU serving container, including its first model download."""

import json
import time
import urllib.error
import urllib.request


def main() -> None:
    deadline = time.monotonic() + 600
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=5) as response:
                health = json.load(response)
            assert health["status"] == "ok", health
            break
        except (urllib.error.URLError, TimeoutError) as error:
            last_error = error
            time.sleep(5)
    else:
        raise RuntimeError(f"server did not become ready: {last_error}")

    payload = {
        "state": "先月の請求で同じ金額が二回引き落とされています。",
        "questions": {
            "department": {
                "type": "choice",
                "instructions": "この問い合わせはどの部署が担当すべきか",
                "criteria": {"請求": "支払い・返金", "技術": "不具合・障害"},
            },
        },
    }
    request = urllib.request.Request(
        "http://127.0.0.1:8000/v1/systemone",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        result = json.load(response)
    answer = result["answers"]["department"]
    assert answer["choice"] in payload["questions"]["department"]["criteria"], result
    assert abs(sum(answer["probabilities"].values()) - 1) < 0.001, result
    assert result["usage"]["output_tokens"] == 0, result
    print(json.dumps({"health": health["status"], "answer": answer}, ensure_ascii=False))


if __name__ == "__main__":
    main()
