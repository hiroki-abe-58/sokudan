"""Record torch outputs the MLX backend is tested against (tests/test_mlx_backend.py).

    uv run python tests/fixtures/make_mlx_parity.py

The MLX tests run where torch may not be installed, so the torch side is recorded here:

- `heads`: `masked_softmax` and `OrdinalHead` on random marker states, masks mixing K,
  and random head weights (hidden size 8) -- the port checked without the backbone.
- `predict`: `Agent.predict` (default calibration) and the raw probabilities of the
  published weights on the CPU, for the golden states and questions.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

MODEL = "GeneLab/sokudan-ja-310m@5f91a0d962b45df1794cfadff008a4da88a67a53"
OUT = Path(__file__).with_name("mlx_parity.json")


def heads() -> dict:
    from sokudan.model.head import masked_softmax
    from sokudan.model.ordinal import OrdinalHead

    generator = torch.Generator().manual_seed(0)
    states = torch.randn(5, 6, 8, generator=generator) * 3
    mask = torch.zeros(5, 6)
    for row, k in enumerate([2, 3, 6, 4, 5]):
        mask[row, :k] = 1
    logits = torch.randn(5, 6, generator=generator) * 4
    head = OrdinalHead(8)
    with torch.no_grad():
        for linear in (head.cut, head.location):
            linear.weight.copy_(torch.randn(1, 8, generator=generator))
            linear.bias.copy_(torch.randn(1, generator=generator))
        probs, _ = head(states, mask)
    return {
        "states": states.tolist(), "mask": mask.tolist(), "logits": logits.tolist(),
        "cut": {"weight": head.cut.weight.tolist(), "bias": head.cut.bias.tolist()},
        "location": {"weight": head.location.weight.tolist(),
                     "bias": head.location.bias.tolist()},
        "masked_softmax": masked_softmax(logits, mask).tolist(),
        "ordinal": probs.tolist(),
    }


def predict() -> dict:
    import sokudan
    from sokudan.predict import _Prepared
    from sokudan.schema.question import parse_questions
    from tests.fixtures.make_predict_golden import QUESTIONS, STATES

    agent = sokudan.load(MODEL, backend="torch", device="cpu")
    prepared = {qid: _Prepared(q, q.type, list(q.labels))
                for qid, q in parse_questions(QUESTIONS).items()}
    with torch.no_grad():
        raw = [{qid: p.tolist() for qid, p in agent._forward(s, prepared)[0].items()}
               for s in STATES]
    return {"model": MODEL, "states": STATES, "questions": QUESTIONS,
            "outputs": [agent.predict(s, QUESTIONS) for s in STATES], "raw": raw}


def main() -> None:
    OUT.write_text(json.dumps({"heads": heads(), "predict": predict()},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
