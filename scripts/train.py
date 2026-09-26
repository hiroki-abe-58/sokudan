"""Stage 1 training entrypoint (SOKUDAN_SPEC.md §8, §14.2 block 7:00-8:00).

    uv run python scripts/train.py --seed 0 --run-id s0

Writes `runs/<id>/report.md` with every metric before and after, the seed, and the
wall clock. §14.4 cut the three-seed requirement down to one for the sprint, so the
report says "single seed" in as many words -- a number without its variance should
not be readable as though it had one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

import sokudan.config  # noqa: F401  -- loads .env before huggingface_hub reads it
from sokudan.config import BACKBONE_MODEL_ID
from sokudan.model.sokudan import SokudanModel
from sokudan.train.dataset import load_examples
from sokudan.train.loop import TrainConfig, train


def format_metrics(metrics: dict[str, Any]) -> str:
    lines = []
    for kind in ("choice", "score", "bool"):
        entry = metrics.get(kind)
        if not entry:
            continue
        keys = [k for k in entry if k != "n"]
        lines.append(f"**{kind}** (n={int(entry['n'])})")
        lines.append("")
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("|" + "---|" * len(keys))
        lines.append("| " + " | ".join(f"{entry[k]:.4f}" for k in keys) + " |")
        lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path, result: dict[str, Any], args: argparse.Namespace) -> Path:
    config = result["config"]
    epochs = result["epochs"]
    cache_hits = result["question_cache"]["hits"]
    cache_misses = result["question_cache"]["misses"]
    total_seconds = sum(e["seconds"] for e in epochs)

    report = f"""# 学習レポート — run `{args.run_id}`

> **単一シード (seed={config['seed']}) の結果です。**
> SOKUDAN_SPEC.md §9 は 3 シード以上の平均±標準偏差を要求していますが、
> §14.4 の切り捨て順 1 に従い当日は 1 シードに落としています。
> **この数字に分散は付いていません。** 複数シードでの再測定は後日。

実行日時: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}
バックボーン: `{BACKBONE_MODEL_ID}`
学習データ: `{args.train}` ({result['n_train']} 例)
検証データ: `{args.val}` ({result['n_val']} 例)

## 設定

| 項目 | 値 |
|---|---|
| seed | {config['seed']} |
| epochs | {config['epochs']} |
| batch_size | {config['batch_size']} |
| grad_accum | {config.get('grad_accum', 1)} |
| learning_rate (backbone) | {config['learning_rate']} |
| learning_rate (head) | {config['head_learning_rate']} |
| weight_decay | {config['weight_decay']} |
| warmup_fraction | {config['warmup_fraction']} |
| max_grad_norm | {config['max_grad_norm']} |
| max_state_tokens | {config['max_state_tokens']} |
| ordinal_weight | {config['ordinal_weight']} |
| autocast dtype | {config['amp_dtype']} |
| head layers | {args.head_layers} |
| attn implementation | sdpa（Gate A でフォールバック確定、`docs/gate_a.md`） |
| torch.compile | 無効（§6.2: 可変長・可変選択肢数で再コンパイルが起きるため） |

## 所要時間

| 項目 | 値 |
|---|---|
| 学習 | {total_seconds:.0f} 秒 |
| 例/秒 | {epochs[-1]['examples_per_second'] if epochs else 0} |
| 質問エンコードのキャッシュ | hit {cache_hits} / miss {cache_misses} |

## 学習前（ランダム初期化の head）

{format_metrics(result['before'])}

## 学習後

{format_metrics(result['after'])}

## 注意

- ここでの検証セットは**合成データの held-out 文書**であり、スキーマは学習時と同系統です。
  **未知スキーマでの汎化はこの表では測れません。** それは `bench_ja` の役割で、
  `docs/benchmarks.md` に別途記載します。
- 検証分割は**文書単位**です。同じ文書から作られたスキーマ変種が
  train と val に分かれることはありません（`data/manifest.json` の
  `document_overlap_between_splits` で確認）。
- 較正（温度）はこの段階では適用していません。§8 Stage 2 で別途。

生データ: `metrics.json`
"""
    path = run_dir / "report.md"
    path.write_text(report, encoding="utf-8")
    return path


def save_checkpoint(model: torch.nn.Module, path: Path, *, head_layers: int,
                    encoding: str) -> Path:
    """Write weights plus what it takes to rebuild the model around them.

    `local_attention` is read from the built backbone, not from the command line, so
    the checkpoint records the window the weights were actually trained under. The
    state dict does not carry it -- the window is config, not a parameter -- and a
    loader that fell back to the pretrained config would silently evaluate at 128.
    """
    torch.save(
        {"state_dict": model.state_dict(),
         "config": {"n_head_layers": head_layers,
                    "backbone": BACKBONE_MODEL_ID,
                    "encoding": encoding,
                    "local_attention": model.backbone.spec.local_attention,
                    "input_order": getattr(model, "input_order", "question_first")}},
        path,
    )
    return path


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    """Build the model, unseeded -- `train` seeds before anything else.

    The scorer's random initialisation (`nn.init.normal_`, std 0.002, the only parameter
    drawn at construction; docs/regression_check.md §8) therefore comes from an unseeded
    generator and differs from run to run whatever `--seed` says. That is deliberate:
    from 2026-09-26 (f75e3f2) this function seeded first, and the pre-registered 16 vs 16
    check found the seeded code lower on held-out M1m (p = 0.047,
    docs/regression_check.md §10), so the order was put back to the one every baseline
    run (v0.1 seeds 0-15) was trained with. The data order and everything after
    construction follow `--seed` either way.
    """
    if args.encoding == "joint":
        from sokudan.model.joint import SokudanJointModel

        return SokudanJointModel.from_pretrained_backbone(
            use_ordinal=not args.no_ordinal, local_attention=args.local_attention,
            input_order=args.input_order,
        )
    return SokudanModel.from_pretrained_backbone(
        n_head_layers=args.head_layers, local_attention=args.local_attention,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--val", default="data/val.jsonl")
    parser.add_argument("--run-id", default="s0")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--head-layers", type=int, default=2)
    parser.add_argument("--max-state-tokens", type=int, default=1024)
    parser.add_argument("--ordinal-weight", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None, help="truncate the train split")
    parser.add_argument("--no-ordinal", action="store_true",
                        help="ablation: score rows use the plain softmax")
    parser.add_argument("--encoding", choices=("separate", "joint"),
                        default="separate")
    parser.add_argument(
        "--local-attention", type=int, default=None,
        help="override the backbone's sliding-window size (total width; 1024 = +-512). "
             "Default keeps the pretrained value (128) and v0.1's exact code path.",
    )
    parser.add_argument(
        "--input-order", choices=("question_first", "state_first", "sandwich"),
        default="question_first",
        help="joint arm: block order. Default is v0.1's (question, then state).",
    )
    parser.add_argument(
        "--distill-targets", default=None,
        help="teacher targets (.npz) for distillation (docs/distill.md); default: off",
    )
    parser.add_argument("--distill-alpha", type=float, default=0.5,
                        help="label-loss weight when distilling (the KL gets 1 - alpha)")
    parser.add_argument(
        "--grad-accum", type=int, default=1,
        help="micro-batches per optimizer step (memory fallback; 1 = the plain loop)",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", action="store_true", help="write the checkpoint")
    parser.add_argument(
        "--max-vram-fraction", type=float, default=None,
        help="cap this process's share of device memory (e.g. 0.9). On Windows, running "
             "past dedicated VRAM does not raise OOM -- the driver pages to system memory "
             "and training slows several-fold -- so a cap turns that into allocator "
             "reclaim instead. Memory policy only; it does not change the arithmetic.",
    )
    args = parser.parse_args()

    if args.max_vram_fraction and args.device.startswith("cuda"):
        # v0.4 crept from 27.6 to 32.1 GB of a 32.6 GB card mid-epoch and dropped from
        # 8.1 to 1.1 steps/s: the caching allocator kept blocks from long-sequence
        # batches until the driver started paging. Under a cap the allocator frees its
        # cache and retries before failing, which is what it should have done anyway.
        torch.cuda.set_per_process_memory_fraction(args.max_vram_fraction)

    from transformers import AutoTokenizer

    train_examples = load_examples(args.train)
    val_examples = load_examples(args.val)
    if args.limit:
        train_examples = train_examples[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    model = build_model(args)
    print(f"input_order = {getattr(model, 'input_order', 'question_first')}")
    print(f"backbone local_attention = {model.backbone.spec.local_attention} "
          f"({'override' if args.local_attention is not None else 'pretrained'})")

    config = TrainConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        head_learning_rate=args.head_lr,
        max_state_tokens=args.max_state_tokens,
        ordinal_weight=args.ordinal_weight,
        device=args.device,
        encoding=args.encoding,
        input_order=args.input_order,
        grad_accum=args.grad_accum,
        distill_targets=args.distill_targets,
        distill_alpha=args.distill_alpha,
    )

    run_dir = Path(args.runs_dir) / args.run_id
    teacher = None
    if args.distill_targets:
        from sokudan.train.distill import load_targets

        # Checked against the training file line by line; --limit keeps a prefix.
        teacher = load_targets(args.distill_targets, args.train)
        print(f"distilling from {args.distill_targets} (alpha {args.distill_alpha})")
    result = train(model, tokenizer, train_examples, val_examples, config, run_dir=run_dir,
                   teacher=teacher)

    report = write_report(run_dir, result, args)
    print(f"\nreport -> {report}")

    if args.save:
        checkpoint = save_checkpoint(model, run_dir / "model.pt",
                                     head_layers=args.head_layers, encoding=args.encoding)
        print(f"checkpoint -> {checkpoint}")

    print(json.dumps(result["after"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
