"""Gradio demo for `sokudan-ja-310m` v0.2, for a free CPU Space.

Paste a Japanese business message, pick any of three preset questions (department
routing, urgency, churn suggestion) and optionally write one of your own. Every
selected question goes to `Agent.predict` in one call, and each comes back as a typed
answer with the probabilities read off the decision head. Nothing is generated.

Two things the demo shows rather than hides:

* **The state is passed as a string.** A dict would be rendered as `key: value` lines,
  which is not the input the model was trained on (README, Limits).
* **Probabilities are uncalibrated.** The shipped temperatures improve choice and bool
  ECE but make score RPS worse, so the demo leaves them off and says so.

This replaces `space/app.py` (written for v0.1, which passed the state as a dict).
"""

from __future__ import annotations

import time
from typing import Any

import gradio as gr

MODEL_ID = "GeneLab/sokudan-ja-310m"

PRESETS: dict[str, tuple[str, dict[str, Any]]] = {
    "department": ("部署ルーティング（choice）", {
        "type": "choice",
        "instructions": "この問い合わせはどの部署が担当すべきか",
        "criteria": {"請求": "支払い・返金", "技術": "不具合・障害",
                     "営業": "料金・新規契約", "その他": "上記以外"},
    }),
    "urgency": ("緊急度（score）", {
        "type": "score",
        "instructions": "この依頼の緊急度は",
        "criteria": ["急がない", "早めに", "業務が止まっている"],
    }),
    "churn": ("解約示唆（noul）", {
        "type": "noul",
        "instructions": "送信者は解約・契約終了を示唆しているか",
    }),
}

EXAMPLES = [
    "先月の請求で同じ金額が二回引き落とされています。至急ご確認のうえ、返金の手続きを"
    "お願いできますでしょうか。明日までに回答がない場合は、こちらも対応を考えます。",
    "先日はご対応ありがとうございました。おかげさまで問題なく稼働しています。"
    "来期の増席について、見積もりをいただけると助かります。急ぎではありません。",
    "ログインが数日前から断続的に切れます。業務が止まっており、今日中に何らかの"
    "回避策をいただけないと困ります。改善しないようなら他社への乗り換えも検討します。",
]

CUSTOM_KEY = "custom"
CUSTOM_TYPES = ["choice", "score", "noul"]

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        import sokudan

        _agent = sokudan.load(MODEL_ID, device="cpu")
    return _agent


def custom_question(kind: str, instructions: str, options: str) -> dict[str, Any] | None:
    """Build the free-form question. Raises gr.Error with the reason if it is unusable.

    choice: one option per line, `名前: 説明` or just `名前`.
    score:  one level per line, lowest first.
    noul:   no options.
    """
    instructions = (instructions or "").strip()
    if not instructions:
        return None
    lines = [line.strip() for line in (options or "").splitlines() if line.strip()]
    if kind == "noul":
        return {"type": "noul", "instructions": instructions}
    if kind == "choice":
        criteria: dict[str, str] = {}
        for line in lines:
            name, _, description = line.partition(":")
            if not description and "：" in line:
                name, _, description = line.partition("：")
            criteria[name.strip()] = description.strip()
        if len(criteria) < 2:
            raise gr.Error("choice には選択肢を 2 行以上書いてください（1 行に「名前: 説明」）。")
        return {"type": "choice", "instructions": instructions, "criteria": criteria}
    if len(lines) < 2:
        raise gr.Error("score には水準を 2 行以上、低い方から書いてください。")
    if len(set(lines)) != len(lines):
        raise gr.Error("score の水準は、それぞれ違う文にしてください。")
    return {"type": "score", "instructions": instructions, "criteria": lines}


def as_label(answer: dict[str, Any]) -> dict[str, float]:
    """The distribution in the shape `gr.Label` draws."""
    if answer["type"] == "bool":
        return {"はい": answer["noul"], "いいえ": round(1 - answer["noul"], 4)}
    if answer["type"] == "score":
        legend = answer["legend"]
        return {legend[k]: p for k, p in answer["probabilities"].items()}
    return dict(answer["probabilities"])


def headline(title: str, answer: dict[str, Any]) -> str:
    if answer["type"] == "bool":
        return f"- **{title}**: P(はい) = **{answer['noul']:.4f}**"
    if answer["type"] == "score":
        k = len(answer["legend"]) - 1
        nearest = answer["legend"][str(round(answer["score"]))]
        return (f"- **{title}**: score **{answer['score']:.2f}**（0〜{k}、"
                f"最も近い水準は「{nearest}」）")
    return f"- **{title}**: **{answer['choice']}**"


def analyse(text: str, presets: list[str], kind: str, instructions: str, options: str):
    empty = gr.update(value=None, visible=False)
    if not text or not text.strip():
        raise gr.Error("日本語の文を入力してください。")

    questions = {key: PRESETS[key][1] for key in PRESETS if PRESETS[key][0] in presets}
    custom = custom_question(kind, instructions, options)
    if custom is not None:
        questions[CUSTOM_KEY] = custom
    if not questions:
        raise gr.Error("質問を 1 つ以上選ぶか、自由入力の質問を書いてください。")

    started = time.perf_counter()
    # The state goes in as a string: the input the model was trained on.
    result = get_agent().predict(text.strip(), questions)
    elapsed = (time.perf_counter() - started) * 1000
    answers = result["answers"]

    lines = []
    labels = []
    for key in [*PRESETS, CUSTOM_KEY]:
        if key not in answers:
            labels.append(empty)
            continue
        title = PRESETS[key][0] if key in PRESETS else f"自由入力（{kind}）"
        lines.append(headline(title, answers[key]))
        labels.append(gr.update(value=as_label(answers[key]), label=title, visible=True))

    usage = result["usage"]
    lines += [
        "",
        f"推論 {elapsed:.0f} ms（この Space の CPU）。"
        f"backbone のパス数 {usage['backbone_passes']}（質問ごとに state を読み直すので、"
        f"時間は質問数に比例します）。生成したトークンは {usage['output_tokens']}。",
    ]
    if usage["state_truncated"]:
        lines.append("**state が長すぎて切り詰められました。**")
    return "\n".join(lines), *labels


with gr.Blocks(title="sokudan-ja-310m") as demo:
    gr.Markdown(
        """# 即断 / sokudan-ja-310m v0.2

日本語の業務文を貼ると、選んだ質問に**テキストを生成せずに**答えます。
確率は decision head のロジットから読んだ値で、モデルの自己申告ではありません。

- モデル: [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m)（Apache-2.0）
- コード・全指標・限界: [github.com/hiroki-abe-58/sokudan](https://github.com/hiroki-abe-58/sokudan)

> 初回だけ、重みの取得と読み込みに時間がかかります。
"""
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(label="日本語の業務文（state）", lines=7,
                              placeholder="例: 先月の請求で同じ金額が二回引き落とされています…")
            presets = gr.CheckboxGroup(
                choices=[title for title, _ in PRESETS.values()],
                value=[title for title, _ in PRESETS.values()],
                label="プリセットの質問",
            )
            with gr.Accordion("自由入力の質問（任意）", open=False):
                kind = gr.Radio(CUSTOM_TYPES, value="noul", label="型")
                instructions = gr.Textbox(label="質問文", placeholder="例: 返金を求めているか")
                options = gr.Textbox(
                    label="選択肢（choice: 1 行に「名前: 説明」/ score: 1 行に 1 水準、低い方から"
                          " / noul: 不要）",
                    lines=4,
                )
            button = gr.Button("判定する", variant="primary")
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text])
        with gr.Column():
            summary = gr.Markdown()
            outputs = [gr.Label(visible=False, num_top_classes=10) for _ in range(4)]

    inputs = [text, presets, kind, instructions, options]
    button.click(analyse, inputs=inputs, outputs=[summary, *outputs])

    gr.Markdown(
        """---
**読む前に知っておいてほしい限界**（詳しくは README とモデルカード）:

- **確率は較正していません。** 同梱の温度は choice と bool の ECE を改善しますが、
  score の RPS を悪化させます。
- **noul（はい/いいえ）は「はい」を過少に出します。** `bench_ja` で平均 P(はい) 0.133、
  正解の陽性率 0.297。順位付けは機能します（AUROC 0.844）が、閾値はご自身で決めてください。
- **4 段階以上の score では、第 1 水準がほとんど選ばれません。** 3 段階から外挿しないでください。
- **学習データも評価データも合成データです**（業務の問い合わせ文）。
  ほかの種類の文では測っていません。
- **日本語専用です。** 英語の文では、noul が多数決とほぼ同じ精度になります。
"""
    )

if __name__ == "__main__":
    demo.launch()
