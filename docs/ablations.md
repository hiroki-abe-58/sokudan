# アブレーション（v0.1、joint、各 1 シード）

> **すべて held-out val（未知スキーマ × 未知文書）で評価しています。**
> `bench_ja` には一切触れていません——アブレーションの掃引で繰り返し当てれば、
> それは held-out テストセットではなく検証セットになってしまうためです。
> ベースラインは公開している構成（`runs/v01_seed0`、seed 0、epoch 2）です。

| アブレーション | ビュー | ep | 秒 | held-out 全体 | I 段 | E 段 | S 段 | val choice | val RPS↓ | val bool | val bool AUROC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (epoch 2) | 67,394 | 2 | 659 | 0.888 | 0.841 | 0.995 | 0.722 | 0.705 | 0.187 | 0.930 | 0.984 |
| (a) epoch 1 | 67,394 | 1 | 329 | 0.887 | 0.837 | 0.994 | 0.702 | 0.723 | 0.161 | 0.936 | 0.986 |
| (a) epoch 3 | 67,394 | 3 | 984 | 0.897 | 0.798 | 0.995 | 0.786 | 0.705 | 0.189 | 0.931 | 0.982 |
| (b) no schema randomisation | 67,394 | 2 | 645 | 0.865 | 0.832 | 0.984 | 0.668 | 0.695 | 0.211 | 0.929 | 0.978 |
| (c) derived booleans 1:2 | 67,393 | 2 | 655 | 0.881 | 0.814 | 0.992 | 0.690 | 0.683 | 0.174 | 0.930 | 0.981 |
| (d) no ordinal head | 67,394 | 2 | 638 | 0.858 | 0.830 | 0.987 | 0.634 | 0.694 | 0.190 | 0.927 | 0.984 |

各行の意味:

- **baseline (epoch 2)** — published configuration
- **(a) epoch 1** — half the training
- **(a) epoch 3** — half again more
- **(b) no schema randomisation** — option order, surface forms and paraphrases fixed
- **(c) derived booleans 1:2** — derived restored, intent cut to match
- **(d) no ordinal head** — score through the plain softmax

生データ: `runs/abl_reports/*.json`、`runs/abl_*/metrics.json`
