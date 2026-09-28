"""Record `Agent.predict` outputs of the code as it is, for tests/test_order_marginalize.py.

    CUDA_VISIBLE_DEVICES= uv run python tests/fixtures/make_predict_golden.py

Written before `order_marginalize` was added (docs/order_marginalization.md), so the
test can show that `order_marginalize=False` leaves every output exactly as it was.
The model is the joint architecture built from the backbone with torch seeded at 0 (the
scorer is the only randomly initialised parameter), on the CPU.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

STATES = [
    "先月の請求で同じ金額が二回引き落とされています。至急ご確認ください。",
    "来月から利用人数を増やしたいので、見積もりをお願いできますか。急ぎではありません。",
]
QUESTIONS = {
    "department": {"type": "choice", "instructions": "この問い合わせはどの部署が担当すべきか",
                   "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                                "営業": "料金・新規契約", "その他": "上記以外"}},
    "urgency": {"type": "score", "instructions": "この依頼の緊急度は",
                "criteria": ["急がない", "早めに", "業務が止まっている"]},
    "churn": {"type": "noul", "instructions": "解約を示唆しているか"},
}
OUT = Path(__file__).with_name("predict_golden.json")


def build_agent():
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.model.joint import SokudanJointModel
    from sokudan.predict import Agent

    torch.manual_seed(0)
    model = SokudanJointModel.from_pretrained_backbone(BACKBONE_MODEL_ID)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    return Agent(model, tokenizer, device="cpu", encoding="joint")


def main() -> None:
    agent = build_agent()
    outputs = [agent.predict(s, QUESTIONS) for s in STATES]
    OUT.write_text(json.dumps(outputs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
