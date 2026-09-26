"""Three seeds of v0.4 against three of v0.1 and three of v0.3 (night 4, item 1).

    uv run python scripts/judge_3x3.py

The rule was registered before seeds 1 and 2 existed (docs/night_report_4.md §0):
v0.4 exceeds v0.1 on a metric only if **all three v0.4 seeds are above v0.1's best
seed**. Under a shared distribution that is one arrangement in twenty -- a one-sided
permutation p of 0.05, the strongest statement three against three can make.

The exact permutation p on the difference of means is printed beside it, over all
C(6,3) = 20 ways to split the six values, as context. With a sample this small it moves
in steps of 0.05.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path
from typing import Any

MODELS = {"v0.1": "v01", "v0.3": "v03", "v0.4": "v04"}


def frozen(tag: str, seed: int, key: str) -> float:
    run = json.loads(Path(f"runs/seedvar_{tag}_seed{seed}.json").read_text(encoding="utf-8"))
    return run["overall"]["auroc"] if key == "overall" else run["by_tier"][key]["auroc"]


def permutation_p(treated: list[float], control: list[float]) -> float:
    """One-sided: share of splits whose mean difference is at least the observed one."""
    pooled = treated + control
    observed = st.mean(treated) - st.mean(control)
    k = len(treated)
    count = total = 0
    for idx in itertools.combinations(range(len(pooled)), k):
        a = [pooled[i] for i in idx]
        b = [pooled[i] for i in range(len(pooled)) if i not in idx]
        total += 1
        count += st.mean(a) - st.mean(b) >= observed - 1e-12
    return count / total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--twelve-old", default="runs/diagnostics/heldout_compare_3x3.json")
    parser.add_argument("--twelve-new", default="runs/diagnostics/heldout_compare_v04_3.json")
    parser.add_argument("--out", default="runs/diagnostics/judge_3x3.json")
    args = parser.parse_args()

    twelve = {}
    for path in (args.twelve_old, args.twelve_new):
        twelve.update(json.loads(Path(path).read_text(encoding="utf-8"))["by_checkpoint"])

    metrics: list[tuple[str, str, Any]] = [
        ("凍結 / I 段", "主", lambda tag, s: frozen(tag, s, "I")),
        ("Day 3 追加 7 属性 / I 段", "主",
         lambda tag, s: twelve[f"runs/{tag}_seed{s}/model.pt"]["Day 3 で追加/I"]["auroc"]),
        ("凍結 / overall", "副", lambda tag, s: frozen(tag, s, "overall")),
        ("12 属性 / 全体 / all", "副",
         lambda tag, s: twelve[f"runs/{tag}_seed{s}/model.pt"]["全体/all"]["auroc"]),
        ("12 属性 / 全体 / I", "副",
         lambda tag, s: twelve[f"runs/{tag}_seed{s}/model.pt"]["全体/I"]["auroc"]),
    ]

    result = {}
    for name, role, get in metrics:
        values = {m: [get(tag, s) for s in range(3)] for m, tag in MODELS.items()}
        v01, v04 = values["v0.1"], values["v0.4"]
        top = max(v01)
        places = ["ABOVE" if v > top else ("BELOW" if v < min(v01) else "INSIDE") for v in v04]
        exceeds = all(v > top for v in v04)
        p = permutation_p(v04, v01)
        result[name] = {"role": role, "values": values, "v0.4_places_vs_v0.1": places,
                        "exceeds_v0.1": exceeds,
                        "mean_diff_vs_v0.1": st.mean(v04) - st.mean(v01),
                        "permutation_p_vs_v0.1": p,
                        "mean_diff_vs_v0.3": st.mean(v04) - st.mean(values["v0.3"]),
                        "permutation_p_vs_v0.3": permutation_p(v04, values["v0.3"])}
        print(f"== {name}（{role}）==")
        for m, vals in values.items():
            print(f"  {m}: " + " / ".join(f"{v:.4f}" for v in vals)
                  + f"   mean {st.mean(vals):.4f} ± {st.pstdev(vals):.4f}")
        print(f"  v0.4 の各シード vs v0.1 の範囲 [{min(v01):.4f}, {top:.4f}]: {places}")
        print(f"  判定（3 シードすべてが v0.1 の最大を上回る）: "
              f"{'上回る' if exceeds else '上回ったとは言えない'}")
        print(f"  平均差 vs v0.1 {result[name]['mean_diff_vs_v0.1']:+.4f}  並べ替え p = {p:.2f}   "
              f"vs v0.3 {result[name]['mean_diff_vs_v0.3']:+.4f}  p = "
              f"{result[name]['permutation_p_vs_v0.3']:.2f}\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
