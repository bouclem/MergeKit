"""Tests for the MoE merge method."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from mergekit.methods import get_method  # noqa: E402


def _fake_state_dict(offset: float) -> dict:
    return {
        "model.embed_tokens.weight": torch.full((10, 8), offset),
        "model.layers.0.mlp.gate_proj.weight": torch.full((16, 8), offset),
        "model.layers.0.mlp.up_proj.weight": torch.full((16, 8), offset + 1),
        "model.layers.0.self_attn.q_proj.weight": torch.full((8, 8), offset),
    }


def test_moe_merge_creates_experts_and_router():
    models = [_fake_state_dict(0.0), _fake_state_dict(1.0), _fake_state_dict(2.0)]
    merged = get_method("moe").merge(models, expert_pattern=r"mlp")

    # shared tensors copied from the first model
    assert torch.equal(
        merged["model.embed_tokens.weight"],
        models[0]["model.embed_tokens.weight"],
    )
    assert "model.layers.0.self_attn.q_proj.weight" in merged

    # one expert per input model, values preserved
    for i in range(3):
        for proj in ("gate_proj", "up_proj"):
            key = f"model.layers.0.mlp.experts.{i}.{proj}.weight"
            assert torch.equal(
                merged[key],
                models[i][f"model.layers.0.mlp.{proj}.weight"],
            )

    router = merged["model.layers.0.mlp.router.weight"]
    assert router.shape == (3, 8)


def test_moe_merge_rejects_single_model():
    with pytest.raises(ValueError, match="at least 2"):
        get_method("moe").merge([_fake_state_dict(0.0)])


def test_moe_merge_rejects_mismatched_keys():
    bad = _fake_state_dict(0.0)
    bad["model.extra.weight"] = torch.zeros(4)
    with pytest.raises(ValueError, match="different tensor names"):
        get_method("moe").merge([_fake_state_dict(0.0), bad])


def test_moe_merge_rejects_bad_router_init():
    models = [_fake_state_dict(0.0), _fake_state_dict(1.0)]
    with pytest.raises(ValueError, match="router_init"):
        get_method("moe").merge(models, router_init="bogus")
