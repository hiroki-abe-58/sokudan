"""Assemble the Hugging Face repository for `sokudan-ja-310m` and push it.

    uv run python scripts/push_to_hub.py --seeds runs/s2a runs/s2b runs/s2c   # dry run
    uv run python scripts/push_to_hub.py --seeds runs/s2a runs/s2b runs/s2c --push

Builds the upload directory locally first and prints it. Nothing leaves the machine
without `--push`, because the model card is the one artefact where an unmeasured
number is worst: SOKUDAN_SPEC.md §11 says overstating it here is how the project
dies, and a card is read by people who will never open `docs/benchmarks.md`.

The `bench_ja` table is filled from the per-seed result files as **mean ± standard
deviation across seeds**, never from a single run. If fewer than three seeds are
given, the card says so in the table caption instead of quietly reporting one seed
as though it were the result -- which is the thing `docs/benchmarks.md` §8 had to
disclose after the fact.

Any `TBD` left in the card after filling is reported and, unless `--allow-tbd`, stops
the push. A card that ships with `TBD` in a metrics table is worse than no card.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ID = "hiroki-abe-58/sokudan-ja-310m"
SOKUDAN_NAME = "sokudan"

# (card column, results path) for the sokudan row of the bench_ja table.
COLUMNS: list[tuple[str, tuple[str, str]]] = [
    ("choice acc", ("choice", "accuracy")),
    ("choice ECE", ("choice", "ece")),
    ("score RPS", ("score", "rps")),
    ("score acc", ("score", "accuracy")),
    ("score MAE", ("score", "mae_argmax")),
    ("bool acc", ("bool", "accuracy")),
    ("bool ECE", ("bool", "ece")),
    ("bool AUROC", ("bool", "auroc")),
]


def read_seed(run_dir: Path, calibrated: bool) -> dict[tuple[str, str], float]:
    """Pull the sokudan row out of one seed's `results.json`."""
    blob = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    wanted = None
    for entry in blob["results"]:
        name = str(entry.get("name", ""))
        if SOKUDAN_NAME not in name.lower():
            continue
        is_calibrated = "較正" in name or "calibrat" in name.lower()
        if is_calibrated == calibrated:
            wanted = entry
            break
    if wanted is None:
        raise SystemExit(
            f"{run_dir}/results.json has no "
            f"{'calibrated ' if calibrated else ''}sokudan row"
        )
    out: dict[tuple[str, str], float] = {}
    for _label, key in COLUMNS:
        kind, metric = key
        value = (wanted.get(kind) or {}).get(metric)
        if value is not None:
            out[key] = float(value)
    return out


def aggregate(rows: list[dict[tuple[str, str], float]]) -> dict[tuple[str, str], str]:
    """mean ± sd per column. One seed gets no ± -- it does not have one."""
    out: dict[tuple[str, str], str] = {}
    for _label, key in COLUMNS:
        values = [row[key] for row in rows if key in row]
        if not values:
            out[key] = "TBD"
        elif len(values) == 1:
            out[key] = f"{values[0]:.3f}"
        else:
            out[key] = f"{mean(values):.3f} ± {stdev(values):.3f}"
    return out


def fill_bench_table(card: str, plain: dict, calibrated: dict, n_seeds: int) -> str:
    """Replace the two all-TBD sokudan rows with the aggregated numbers."""
    def row(label: str, values: dict) -> str:
        cells = " | ".join(values[key] for _l, key in COLUMNS)
        return f"| {label} | {cells} |"

    replacements = [
        (r"^\| \*\*sokudan-ja-310m\*\* \|(?: TBD \|)+\s*$",
         row("**sokudan-ja-310m**", plain)),
        (r"^\| \*\*sokudan-ja-310m \+ 温度較正\*\* \|(?: TBD \|)+\s*$",
         row("**sokudan-ja-310m + 温度較正**", calibrated)),
    ]
    for pattern, replacement in replacements:
        card, count = re.subn(pattern, replacement.replace("\\", "\\\\"), card,
                              flags=re.M)
        if count != 1:
            raise SystemExit(f"model card: expected 1 match for {pattern!r}, got {count}")

    caption = (
        f"> **{n_seeds} シードの平均 ± 標準偏差**"
        "（`bench_ja` 300 件、同一条件、同一メトリクス実装）。"
        if n_seeds >= 3 else
        f"> **⚠ {n_seeds} シードのみ。標準偏差は"
        + ("付いていません。" if n_seeds == 1 else
           "この本数から計算したもので、§9 の要件（3本以上）を満たしていません。")
        + "**"
    )
    card = card.replace("> **すべて TBD。測定後に埋める。**", caption)
    return card


def fill_data_table(card: str, manifest: dict[str, Any]) -> str:
    """The three-row training-data table, from counted values only."""
    documents = manifest.get("documents")
    pairs = manifest.get("pairs_after_rebalance", manifest.get("pairs_kept"))
    views = manifest.get("train_views", 0) + manifest.get("val_views", 0)
    swaps = [
        ("| 生成した文書 | TBD 文書 |", f"| 生成した文書 | {documents:,} 文書 |"),
        ("| ラベル付き (文書, 質問) ペア | TBD ペア |",
         f"| ラベル付き (文書, 質問) ペア | {pairs:,} ペア |"),
        ("| スキーマ拡張後のビュー | TBD ビュー |",
         f"| スキーマ拡張後のビュー | {views:,} ビュー |"),
        ("「TBD ビュー」を「TBD 例」と読み替えないでください。",
         f"「{views:,} ビュー」を「{views:,} 例」と読み替えないでください。"),
    ]
    for old, new in swaps:
        if old in card:
            card = card.replace(old, new)
    return card


def stage(
    out_dir: Path, checkpoint: Path, temperatures: Path | None, card: str
) -> list[str]:
    """Write the upload directory. Weights as safetensors, tokenizer from the backbone."""
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    stored = blob.get("config", {})
    backbone = stored.get("backbone", "sbintuitions/modernbert-ja-310m")

    state = {k: v.contiguous() for k, v in blob["state_dict"].items()}
    save_file(state, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "source": str(checkpoint)})

    encoding = stored.get("encoding", "separate")
    config = {
        "model_type": "sokudan",
        "architectures": ["SokudanJointModel" if encoding == "joint" else "SokudanModel"],
        "backbone": backbone,
        "encoding": encoding,
        "n_parameters": sum(v.numel() for v in state.values()),
        "question_types": ["choice", "score", "bool"],
        "bool_aliases": ["noul"],
        "license": "apache-2.0",
        "note": (
            "Custom architecture. v0.1 uses joint encoding: "
            "[CLS] instructions [SEP] options+markers [SEP] state [SEP] through the "
            "backbone, marker hidden states through a scorer. There is no decision "
            "head, and the state is re-encoded per question. Not an AutoModel -- "
            "load with `sokudan.load`, not `AutoModel.from_pretrained`."
        ),
    }
    if encoding != "joint":
        config["n_head_layers"] = stored.get("n_head_layers", 2)
    (out_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    tokenizer.save_pretrained(str(out_dir))

    if temperatures is not None and temperatures.exists():
        shutil.copy(temperatures, out_dir / "temperatures.json")

    (out_dir / "README.md").write_text(card, encoding="utf-8")
    return sorted(p.name for p in out_dir.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", default=["runs/s2a", "runs/s2b", "runs/s2c"],
                        help="run directories holding per-seed results.json")
    parser.add_argument("--checkpoint", default=None,
                        help="weights to publish; defaults to the first seed's model.pt")
    parser.add_argument("--temperatures", default=None,
                        help="defaults to the first seed's temperatures.json")
    parser.add_argument("--card", default="docs/model_card.md")
    parser.add_argument("--data-manifest", default="data/manifest_v2.json")
    parser.add_argument("--out-dir", default="runs/hf_upload")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=None,
                        help="branch to upload to (e.g. seed1); default is main")
    parser.add_argument("--push", action="store_true",
                        help="actually upload; without it nothing leaves the machine")
    parser.add_argument("--allow-tbd", action="store_true")
    args = parser.parse_args()

    seed_dirs = [Path(s) for s in args.seeds]
    missing = [str(d) for d in seed_dirs if not (d / "results.json").exists()]
    if missing:
        print(f"missing results.json: {missing}")
        return 1

    plain = aggregate([read_seed(d, calibrated=False) for d in seed_dirs])
    calibrated = aggregate([read_seed(d, calibrated=True) for d in seed_dirs])

    card = Path(args.card).read_text(encoding="utf-8")
    card = fill_bench_table(card, plain, calibrated, len(seed_dirs))

    manifest_path = Path(args.data_manifest)
    if manifest_path.exists():
        card = fill_data_table(card, json.loads(manifest_path.read_text(encoding="utf-8")))

    # The scaffold comment tells the reader the numbers are placeholders. Once they
    # are not, leaving it in is itself a false statement.
    card = re.sub(r"<!--\s*\n  Hugging Face モデルカードの骨格.*?-->\n*", "", card, flags=re.S)

    remaining = [
        f"  line {number}: {line.strip()}"
        for number, line in enumerate(card.splitlines(), start=1) if "TBD" in line
    ]

    checkpoint = Path(args.checkpoint) if args.checkpoint else seed_dirs[0] / "model.pt"
    temperatures = (
        Path(args.temperatures) if args.temperatures else seed_dirs[0] / "temperatures.json"
    )
    files = stage(Path(args.out_dir), checkpoint, temperatures, card)

    print(f"seeds:      {[str(d) for d in seed_dirs]}")
    print(f"checkpoint: {checkpoint}")
    print(f"staged ->   {args.out_dir}")
    for name in files:
        print(f"  {name}")
    print("\nbench_ja（sokudan 行、平均 ± 標準偏差）")
    for label, key in COLUMNS:
        print(f"  {label:12s} {plain[key]:>16s}   較正後 {calibrated[key]:>16s}")

    if remaining:
        print(f"\nカードに TBD が {len(remaining)} 箇所残っています:")
        for line in remaining[:20]:
            print(line)
        if not args.allow_tbd:
            print("\n未測定の数値を載せたカードは公開しない。--allow-tbd で上書き可。")
            return 2

    if not args.push:
        print(f"\nドライラン。公開するには --push（repo: {args.repo_id}）")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    if args.revision:
        # The alternate seeds live on branches of the same repository, so the card
        # on `main` can point at them without publishing three repositories whose
        # only difference is a seed.
        api.create_branch(
            args.repo_id, repo_type="model", branch=args.revision, exist_ok=True
        )
    api.upload_folder(
        folder_path=args.out_dir, repo_id=args.repo_id, repo_type="model",
        revision=args.revision or None,
        commit_message=(
            f"sokudan-ja-310m v0.1 weights, {args.revision}" if args.revision
            else f"sokudan-ja-310m v0.1 ({len(seed_dirs)} seeds)"
        ),
    )
    where = f"https://huggingface.co/{args.repo_id}"
    if args.revision:
        where += f"/tree/{args.revision}"
    print(f"\npushed -> {where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
