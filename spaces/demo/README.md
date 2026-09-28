---
title: sokudan-ja-310m
emoji: ⚡
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.28.0
python_version: "3.11"
app_file: app.py
pinned: false
license: apache-2.0
models:
  - GeneLab/sokudan-ja-310m
short_description: Japanese typed decisions with probabilities, no generation
---

# 即断 / sokudan-ja-310m v0.2.1

日本語の業務文（state）と型付きの質問を受け取り、**テキストを生成せずに**型付きの答えと確率を返すモデルのデモです。

- 質問はプリセットが 3 つ（部署ルーティング = choice、緊急度 = score、解約示唆 = noul）と、自由入力が 1 つ。選んだ質問は 1 回の呼び出しでまとめて答えます。
- 確率は decision head のロジットから読んだ値です。**較正しているのは noul（はい/いいえ）だけ**で、choice と score は生の値です。
- 無料の CPU で動きます（314.6M パラメータ）。初回だけ、重みの取得と読み込みに時間がかかります。

モデル: [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m) ・ コード、全指標、限界: [github.com/hiroki-abe-58/sokudan](https://github.com/hiroki-abe-58/sokudan)
