"""Gradio demo for `sokudan-ja-310m`, running on a free CPU Space.

The point of the demo is the thing that is hard to convey in a README: the model does
not write anything. You paste Japanese text, and three typed answers come back with
the probabilities read off the head's logits. There is no text to parse.

Two numbers are shown that a demo would normally hide.

**`backbone_passes`.** v0.1 uses joint encoding, so the state is re-encoded once per
question and latency scales with how many questions you ask. Three questions means
three passes. Showing it keeps the README's latency caveat visible at the point where
someone is forming an impression of the speed.

**Uncalibrated by default.** The shipped temperatures improve `choice` and `bool`
calibration but make `score` RPS worse (0.090 -> 0.149), so the demo leaves them off
and says so rather than quietly picking one.
"""

from __future__ import annotations

import time

import gradio as gr

MODEL_ID = "GeneLab/sokudan-ja-310m"

EXAMPLES = [
    "先月の請求で同じ金額が二回引き落とされています。至急ご確認のうえ、返金の手続きを"
    "お願いできますでしょうか。明日までに回答がない場合は、こちらも対応を考えます。",
    "先日はご対応ありがとうございました。おかげさまで問題なく稼働しています。"
    "来期の増席について、見積もりをいただけると助かります。急ぎではありません。",
    "ログインが数日前から断続的に切れます。業務が止まっており、今日中に何らかの"
    "回避策をいただけないと困ります。よろしくお願いいたします。",
]

_agent = None


def get_agent():
    global _agent
    if _agent is None:
        import sokudan

        _agent = sokudan.load(MODEL_ID, device="cpu")
    return _agent


QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "この問い合わせはどの部署が担当すべきか",
        "criteria": {
            "請求": "支払い・返金・請求書など金銭に関するもの",
            "技術": "不具合・障害・ログイン不能など製品が動かないこと",
            "営業": "料金プラン・新規契約・見積もりなど購入判断",
            "その他": "上のどれにも当てはまらないもの",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "この依頼の緊急度は",
        "criteria": ["急がない", "早めに", "業務が止まっている"],
    },
    "churn": {
        "type": "noul",
        "instructions": "送信者は解約・契約終了を示唆しているか",
    },
}


def analyse(text: str):
    if not text or not text.strip():
        return "テキストを入力してください。", None, None, ""

    started = time.perf_counter()
    result = get_agent().predict({"body": text}, QUESTIONS)
    elapsed = (time.perf_counter() - started) * 1000
    answers = result["answers"]

    department = answers["department"]
    choice_rows = [[k, round(v, 4)] for k, v in department["probabilities"].items()]

    urgency = answers["urgency"]
    legend = urgency["legend"]
    score_rows = [
        [legend[k], round(v, 4)] for k, v in urgency["probabilities"].items()
    ]

    noul = answers["churn"]["noul"]
    summary = (
        f"### 部署: **{department['choice']}**  \n"
        f"### 緊急度: **{legend[str(round(urgency['score']))]}**"
        f"（期待値 {urgency['score']:.2f} / 0-2）  \n"
        f"### 解約の示唆: **P(true) = {noul:.4f}**"
    )

    usage = result["usage"]
    detail = (
        f"- 推論 **{elapsed:.0f} ms**（無料の CPU Space。GPU では 1 問あたり 11.9 ms）\n"
        f"- **backbone パス数: {usage['backbone_passes']}** "
        f"— joint encoding なので質問ごとに state を再エンコードします。"
        f"レイテンシは質問数に比例します\n"
        f"- state トークン数: {usage['state_tokens']}"
        f"{'（**切り詰められました**）' if usage['state_truncated'] else ''}\n"
        f"- 生成トークン数: {usage['output_tokens']} — テキストは一切生成していません\n"
        f"- **確率は未較正です。** 同梱の温度は `choice` と `bool` の ECE を改善しますが、"
        f"`score` の RPS は悪化させます\n"
        f"- **解約の示唆は true を過少予測します**（`bench_ja` で平均 P(true) 0.125 対 "
        f"gold 0.297）。AUROC 0.789 で順位付けは機能しますが、閾値はご自身の事前確率に"
        f"合わせてください"
    )
    return summary, choice_rows, score_rows, detail


with gr.Blocks(title="sokudan-ja-310m") as demo:
    gr.Markdown(
        """# 即断 / sokudan-ja-310m

日本語のテキストを貼ると、**テキストを一切生成せずに**単一フォワードパスで
型付きの回答と確率を返します。確率は head のロジットを直接読んだもので、
モデルの自己申告ではありません。

- モデル: [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m)（314.6M、Apache-2.0）
- コードと全指標: [github.com/hiroki-abe-58/sokudan](https://github.com/hiroki-abe-58/sokudan)

> **初回は読み込みに 1〜2 分かかります。** 無料の CPU Space で 314M の
> モデルを起動時に取得するためです。2 回目以降は数秒で返ります。
"""
    )
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="日本語の問い合わせ文",
                placeholder="例: 先月の請求で同じ金額が二回引き落とされています…",
                lines=8,
            )
            button = gr.Button("判定する", variant="primary")
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[text])
        with gr.Column():
            summary = gr.Markdown()
            choice_table = gr.Dataframe(
                headers=["部署", "確率"], label="choice: どの部署が担当すべきか",
                interactive=False,
            )
            score_table = gr.Dataframe(
                headers=["緊急度", "確率"], label="score: 緊急度（動的 K の cumulative link）",
                interactive=False,
            )
            detail = gr.Markdown()

    button.click(analyse, inputs=[text], outputs=[summary, choice_table, score_table, detail])
    text.submit(analyse, inputs=[text], outputs=[summary, choice_table, score_table, detail])

    gr.Markdown(
        """---
**このデモが示していない限界**（すべて
[`docs/benchmarks.md`](https://github.com/hiroki-abe-58/sokudan/blob/main/docs/benchmarks.md) に実測付きで）:

- 学習データは合成データのみ（21 ドメイン、単一の生成モデル）
- 評価は `bench_ja` の 3 スキーマ・300 件のみ
- 4 段階以上の `score` は精度が落ちます（4 段階で acc 0.427）
- 300 トークンを超える state での性能は**測定していません**
- スキーマはリクエストごとに自由に変えられます（再学習不要）。このデモは 3 つを固定しています
"""
    )

if __name__ == "__main__":
    demo.launch()
