"""The input set every MLX phase compares backends on (no labels; agreement only).

    python scripts/mlx/parity_set.py --out parity_set.json

The repository has no held-out or dev data (`data/` holds only bench_ja / bench_en, which
this comparison does not use), so the set is synthesised from the states in the README,
the Space demo and the tests. `random.Random(SEED)` makes it the same set on every run.

- choice K 2-8, score K 3-7, bool (some with non-default slot labels)
- 1-3 questions per state, so batches mix K, types and padded rows
- state length from one sentence to past the joint limit of 1024 tokens (the longest
  are truncated by `encode_joint`, which is part of what is compared)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20260928
N_ITEMS = 320

# README / README_ja Quickstart, tests/fixtures/make_predict_golden.py, spaces/demo/app.py
# EXAMPLES, tests/fixtures/systemone_wire_examples.json -- split into sentences.
SENTENCES = [
    "先月の請求で同じ金額が二回引き落とされています。",
    "至急ご確認ください。",
    "来月から利用人数を増やしたいので、見積もりをお願いできますか。",
    "急ぎではありません。",
    "至急ご確認のうえ、返金の手続きをお願いできますでしょうか。",
    "明日までに回答がない場合は、こちらも対応を考えます。",
    "先日はご対応ありがとうございました。",
    "おかげさまで問題なく稼働しています。",
    "来期の増席について、見積もりをいただけると助かります。",
    "ログインが数日前から断続的に切れます。",
    "業務が止まっており、今日中に何らかの回避策をいただけないと困ります。",
    "改善しないようなら他社への乗り換えも検討します。",
    "新しいプランの料金表を送っていただけますか。",
]
ENGLISH = [
    "Help! My payouts have been failing for 3 days.",
    "Could you send me the pricing for the enterprise plan?",
]

CHOICE_INSTRUCTIONS = [
    "この問い合わせはどの部署が担当すべきか",
    "問い合わせの種類はどれか",
    "次に取るべき対応はどれか",
    "送信者の主な要望はどれか",
    "Which team should handle this?",
]
CHOICE_OPTIONS = [
    ("請求", "支払い・返金"), ("技術", "不具合・障害"), ("営業", "料金・新規契約"),
    ("その他", "上記以外"), ("解約", "契約の終了・乗り換え"), ("お礼", "感謝・報告のみ"),
    ("見積もり", "価格・数量の相談"), ("アカウント", "ログイン・権限"),
    ("配送", "発送・到着の遅れ"), ("苦情", "対応への不満"), ("返金", "代金の払い戻し"),
    ("billing", "Payments, invoicing, refunds"), ("technical", "Bugs, outages, integrations"),
    ("sales", "Pricing, upgrades, new accounts"), ("法務", "契約書・規約"),
    ("人事", "採用・労務"),
]
SCORE_INSTRUCTIONS = [
    "この依頼の緊急度は",
    "送信者の不満の強さは",
    "対応の難しさは",
    "顧客の満足度は",
]
SCORE_SCALES = [
    ["なし", "ごく低い", "低い", "やや低い", "中程度", "やや高い", "高い"],
    ["急がない", "今月中", "今週中", "早めに", "明日まで", "今日中", "業務が止まっている"],
    ["とても不満", "不満", "やや不満", "普通", "やや満足", "満足", "とても満足"],
]
BOOL_INSTRUCTIONS = [
    "解約を示唆しているか",
    "顧客は返金を求めているか",
    "期限が明記されているか",
    "送信者は解約・契約終了を示唆しているか",
    "Does this convey urgency?",
]
BOOL_LABELS = [None, None, ("期限がない", "期限が明記されている"), ("no", "yes")]

# Sentence counts, from one sentence to past the 1024-token joint limit.
N_SENTENCES = [1, 1, 2, 3, 5, 8, 12, 18, 25, 35, 45, 55, 65, 75, 90]


def make_state(rng: random.Random) -> str:
    n = rng.choice(N_SENTENCES)
    if rng.random() < 0.08:
        return " ".join(rng.choice(ENGLISH) for _ in range(n))
    return "".join(rng.choice(SENTENCES) for _ in range(n))


def make_question(rng: random.Random, kind: str) -> dict:
    if kind == "choice":
        k = rng.randint(2, 8)
        options = rng.sample(CHOICE_OPTIONS, k)
        return {"type": "choice", "instructions": rng.choice(CHOICE_INSTRUCTIONS),
                "criteria": dict(options)}
    if kind == "score":
        k = rng.randint(3, 7)
        scale = rng.choice(SCORE_SCALES)
        levels = [scale[i] for i in sorted(rng.sample(range(len(scale)), k))]
        return {"type": "score", "instructions": rng.choice(SCORE_INSTRUCTIONS),
                "criteria": levels}
    question = {"type": rng.choice(["bool", "noul"]),
                "instructions": rng.choice(BOOL_INSTRUCTIONS)}
    labels = rng.choice(BOOL_LABELS)
    if labels is not None:
        question["false_label"], question["true_label"] = labels
    return question


def build() -> list[dict]:
    rng = random.Random(SEED)
    items = []
    for i in range(N_ITEMS):
        n_questions = rng.randint(1, 3)
        questions = {f"q{j}": make_question(rng, rng.choice(["choice", "score", "bool"]))
                     for j in range(n_questions)}
        items.append({"id": i, "state": make_state(rng), "questions": questions})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    text = json.dumps(build(), ensure_ascii=False, indent=1)
    # LF on every platform: the printed sha256 is of `text`, and the file must match it
    # (Path.write_text would write CRLF on Windows).
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"{args.out} sha256={hashlib.sha256(text.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
