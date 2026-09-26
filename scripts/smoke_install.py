"""What a clean install of `sokudan` must be able to do, with no extras present.

    python scripts/smoke_install.py            # in a fresh venv, or the Docker image
    python scripts/smoke_install.py --no-hub   # skip the one step that needs network

Every check here corresponds to a line the README or the model card tells a reader to
run. The bar is not that the code is correct -- the test suite covers that -- but that
the *core dependency set* is enough to reach it. Two releases have now been spent on
that exact gap:

  0.1.1  `pip install sokudan` + `predict()` -> ModuleNotFoundError: scipy
         (a module-level `from scipy.optimize import ...`, made lazy)
  later  `sokudan calibrate` -> the same import, reached at call time instead
         (the bounded minimisation was rewritten without scipy)

Both were invisible on a development machine, because the venv there has the optional
extras installed. Anything that only works with an extra has to fail here.

Exits non-zero on the first failure, naming the step.
"""

from __future__ import annotations

import argparse
import sys
import traceback

CHECKS: list[tuple[str, str]] = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn
    return register


@check("import the package")
def _import(args) -> str:
    import sokudan

    return f"sokudan {getattr(sokudan, '__version__', '(no __version__)')}"


@check("no optional extra leaked into the core install")
def _no_extras(args) -> str:
    # If one of these imports succeeds, the image is not the clean install it claims
    # to be, and every check after it is worthless as evidence.
    leaked = []
    for module in ("scipy", "sklearn", "fastapi", "pytest"):
        try:
            __import__(module)
        except ImportError:
            continue
        leaked.append(module)
    if leaked:
        raise AssertionError(
            f"{', '.join(leaked)} present; this environment cannot detect the bug "
            "this script exists for"
        )
    return "scipy, sklearn, fastapi, pytest all absent, as intended"


@check("build and validate a question schema")
def _schema(args) -> str:
    from sokudan.schema.question import BoolQuestion, ChoiceQuestion, ScoreQuestion

    BoolQuestion(instructions="解約の意図があるか")
    # `criteria` on every question type, not `choices` -- but a *mapping* here, since
    # a `choice` option carries a description of when it applies, while a `score`
    # level is just an ordered label.
    choice = ChoiceQuestion(instructions="どの部署か", criteria={
        "営業": "見積もり・契約・価格の話",
        "技術": "不具合・仕様・導入作業の話",
        "経理": "請求・支払い・精算の話",
    })
    score = ScoreQuestion(instructions="緊急度は", criteria=["低", "中", "高"])
    return (f"3 question types, choice has {len(choice.criteria)} options, "
            f"score has {len(score.criteria)} levels")


@check("fit and apply a temperature")
def _calibrate(args) -> str:
    import numpy as np

    from sokudan.calibration.metrics import ece, nll
    from sokudan.calibration.temperature import apply_temperature, fit_temperature

    rng = np.random.default_rng(0)
    logits = rng.normal(0, 3.0, size=(500, 3))
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    labels = np.array([rng.choice(3, p=row) for row in probs])
    # The model is overconfident by construction, so fitting has to soften it: a
    # temperature above one, and a lower NLL than the raw probabilities.
    temperature = fit_temperature(probs, labels)
    after = apply_temperature(probs, temperature)
    if not nll(after, labels) <= nll(probs, labels) + 1e-9:
        raise AssertionError("fitted temperature did not reduce NLL on its own data")
    return (f"T={temperature:.4f}, NLL {nll(probs, labels):.4f} -> {nll(after, labels):.4f}, "
            f"ECE {ece(probs, labels):.4f} -> {ece(after, labels):.4f}")


@check("build a model on a tiny random backbone")
def _model(args) -> str:
    import torch
    from transformers import AutoModel, ModernBertConfig

    from sokudan.model.backbone import Backbone, BackboneSpec
    from sokudan.model.joint import SokudanJointModel

    config = ModernBertConfig(
        vocab_size=256, hidden_size=32, num_attention_heads=4, num_hidden_layers=2,
        intermediate_size=64, local_attention=8, global_attn_every_n_layers=2,
        max_position_embeddings=512, pad_token_id=0, bos_token_id=1,
        eos_token_id=2, cls_token_id=3, sep_token_id=4,
    )
    backbone = Backbone(AutoModel.from_config(config), BackboneSpec.from_config(config))
    model = SokudanJointModel(backbone)
    ids = torch.randint(5, 200, (2, 24))
    out = model(ids, torch.ones(2, 24, dtype=torch.long),
                torch.tensor([[1, 3, 5], [1, 3, 3]]),
                torch.tensor([[1, 1, 1], [1, 1, 0]]),
                torch.tensor([True, False]))
    rows = (out.probs * torch.tensor([[1, 1, 1], [1, 1, 0]])).sum(1)
    if not torch.allclose(rows, torch.ones(2), atol=1e-5):
        raise AssertionError(f"probabilities do not sum to one: {rows.tolist()}")
    return f"forward OK, torch {torch.__version__}"


@check("encode a question with the real tokenizer")
def _encode(args) -> str:
    if args.no_hub:
        return "skipped (--no-hub)"
    from transformers import AutoTokenizer

    from sokudan.config import BACKBONE_MODEL_ID
    from sokudan.encoding.question import encode_joint
    from sokudan.schema.question import ScoreQuestion

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE_MODEL_ID)
    encoded = encode_joint(
        ScoreQuestion(instructions="この依頼の緊急度は", criteria=["低", "中", "高"]),
        "明日までに見積もりをいただけますか。",
        tokenizer,
    )
    if len(encoded.marker_positions) != 3:
        raise AssertionError(f"expected 3 markers, got {len(encoded.marker_positions)}")
    return f"{len(encoded.input_ids)} tokens, markers at {list(encoded.marker_positions)}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-hub", action="store_true",
                        help="skip the tokenizer check, which needs the hub once")
    args = parser.parse_args()

    width = max(len(name) for name, _ in CHECKS)
    for name, fn in CHECKS:
        try:
            detail = fn(args)
        except Exception as exc:
            print(f"FAIL {name:{width}}  {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 1
        print(f"ok   {name:{width}}  {detail}")
    print(f"\nall {len(CHECKS)} checks passed on a core-only install")
    return 0


if __name__ == "__main__":
    sys.exit(main())
