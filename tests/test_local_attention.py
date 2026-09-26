"""The `--local-attention` override (docs/local_attention_1024.md, Phase 1).

Three properties have to hold before a model trained at a wider window means
anything:

(a) Short sequences are untouched. At 128 (+-64) every token of a sequence under 64
    tokens already sees every other, so 1024 must give the same hidden states up to
    float noise. If it does not, the override changed something besides the window.
(b) Long sequences are not. At 300+ tokens the two masks differ, so the outputs must
    differ by far more than that noise -- otherwise the override is not reaching the
    attention mask and training "at 1024" would silently train at 128.
(c) The window survives a save and a reload through the same code the evaluation
    uses. The window is config, not a parameter, so the state dict does not carry
    it; a loader that fell back to the pretrained config would evaluate at 128.

These load the real backbone from the local HF cache (the tiny random config other
model tests use would say nothing about the pretrained config's window).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from sokudan.config import BACKBONE_MODEL_ID

ROOT = Path(__file__).resolve().parent.parent

# fp32 on the same device with an identical mask: the only differences left are
# kernel-level reassociation, which stays several orders of magnitude below this.
SAME_TOL = 1e-4
# What (b) must exceed: the largest per-token change in the final hidden state.
# Hidden states here are O(1); 0.05 is ~500x SAME_TOL.
DIFFERENT_MIN = 5e-2

pytestmark = [pytest.mark.slow]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def models():
    from sokudan.model.joint import SokudanJointModel

    try:
        narrow = SokudanJointModel.from_pretrained_backbone()
    except Exception as exc:  # pragma: no cover - depends on the local cache
        pytest.skip(f"backbone unavailable: {type(exc).__name__}: {exc}")
    torch.manual_seed(0)
    wide = SokudanJointModel.from_pretrained_backbone(local_attention=1024)
    device = _device()
    return narrow.to(device).eval(), wide.to(device).eval()


def _ids(tokenizer, n_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    text = "お問い合わせの件について、担当者から折り返しご連絡いたします。" * 200
    ids = tokenizer(text, add_special_tokens=True, truncation=True,
                    max_length=n_tokens, return_tensors="pt")["input_ids"]
    assert ids.shape[1] == n_tokens
    return ids.to(_device()), torch.ones_like(ids).to(_device())


def _hidden(model, ids, mask) -> torch.Tensor:
    with torch.no_grad():
        return model.backbone(ids, mask).float()


def _sliding_windows(model) -> list[int | None]:
    return [layer.attn.sliding_window for layer in model.backbone.model.layers]


def test_default_path_keeps_the_pretrained_window(models):
    narrow, _ = models
    assert narrow.backbone.spec.local_attention == 128
    assert narrow.backbone.model.config.sliding_window == 64
    assert sorted({w for w in _sliding_windows(narrow) if w is not None}) == [65]


def test_override_reaches_config_and_every_sliding_layer(models):
    narrow, wide = models
    assert wide.backbone.spec.local_attention == 1024
    assert wide.backbone.model.config.sliding_window == 512
    windows = _sliding_windows(wide)
    assert sorted({w for w in windows if w is not None}) == [513]
    # Same layers are sliding in both; the override changes the width, not the layout.
    assert [w is None for w in windows] == [w is None for w in _sliding_windows(narrow)]
    # Weights identical: the window is config only.
    for (name, a), (_, b) in zip(narrow.backbone.state_dict().items(),
                                 wide.backbone.state_dict().items(), strict=True):
        assert torch.equal(a, b), name


def test_a_short_sequences_are_unchanged(models, tokenizer):
    narrow, wide = models
    ids, mask = _ids(tokenizer, 60)
    diff = (_hidden(narrow, ids, mask) - _hidden(wide, ids, mask)).abs().max().item()
    print(f"(a) 60 tokens: max |h128 - h1024| = {diff:.3e} (tolerance {SAME_TOL:.0e})")
    assert diff <= SAME_TOL


def test_b_long_sequences_change(models, tokenizer):
    narrow, wide = models
    ids, mask = _ids(tokenizer, 320)
    delta = (_hidden(narrow, ids, mask) - _hidden(wide, ids, mask)).abs()
    per_token = delta.max(dim=-1).values[0]
    print(f"(b) 320 tokens: max {delta.max().item():.3e}, "
          f"mean {delta.mean().item():.3e}, "
          f"tokens over {DIFFERENT_MIN}: {(per_token > DIFFERENT_MIN).sum().item()}/320")
    assert delta.max().item() > DIFFERENT_MIN


def test_c_window_survives_save_and_reload(models, tokenizer, tmp_path):
    _, wide = models
    train = _load_script("train")
    eval_heldout = _load_script("eval_heldout")

    path = train.save_checkpoint(wide, tmp_path / "model.pt", head_layers=2,
                                 encoding="joint")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["config"]["local_attention"] == 1024
    assert blob["config"]["backbone"] == BACKBONE_MODEL_ID
    del blob

    reloaded, encoding, _ = eval_heldout.load_checkpoint(str(path))
    assert encoding == "joint"
    assert reloaded.backbone.spec.local_attention == 1024
    assert reloaded.backbone.model.config.sliding_window == 512
    assert sorted({w for w in _sliding_windows(reloaded) if w is not None}) == [513]

    ids, mask = _ids(tokenizer, 320)
    diff = (_hidden(wide, ids, mask) - _hidden(reloaded, ids, mask)).abs().max().item()
    assert diff <= SAME_TOL


def test_c_checkpoint_without_the_key_loads_at_the_pretrained_window(models, tmp_path):
    """v0.1's checkpoints predate the flag; they must still load at 128."""
    narrow, _ = models
    eval_heldout = _load_script("eval_heldout")
    path = tmp_path / "legacy.pt"
    torch.save({"state_dict": narrow.state_dict(),
                "config": {"n_head_layers": 2, "backbone": BACKBONE_MODEL_ID,
                           "encoding": "joint"}}, path)
    reloaded, _, _ = eval_heldout.load_checkpoint(str(path))
    assert reloaded.backbone.spec.local_attention == 128
    # ...and an explicit override (Phase 3 inference-only check) wins over it.
    overridden, _, _ = eval_heldout.load_checkpoint(str(path), local_attention=1024)
    assert overridden.backbone.spec.local_attention == 1024
