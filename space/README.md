---
title: sokudan-ja-310m
emoji: 即
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: apache-2.0
models:
  - GeneLab/sokudan-ja-310m
---

# 即断 / sokudan-ja-310m

日本語テキストと型付き質問を受け取り、**テキストを一切生成せずに**単一フォワードパスで
型付き回答と確率を返すエンコーダモデルのデモです。

確率は head のロジットを直接読んだもので、モデルの自己申告ではありません。

- モデル: [`GeneLab/sokudan-ja-310m`](https://huggingface.co/GeneLab/sokudan-ja-310m)
- コード・全指標・限界: [github.com/hiroki-abe-58/sokudan](https://github.com/hiroki-abe-58/sokudan)

**初回の読み込みに 1〜2 分かかります。** 無料の CPU Space で 314.6M のモデルを
起動時に取得するためです。
