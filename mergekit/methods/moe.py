"""Mixture-of-Experts merge method (frankenMoE / BTX style)."""

import re

from mergekit.methods.base import MergeMethod

_DEFAULT_EXPERT_PATTERN = r"mlp|feed_forward|ffn"


class MoEMerge(MergeMethod):
    """Combine dense models into a Mixture-of-Experts state dict.

    Tensors whose names match ``expert_pattern`` become per-model
    experts; all other tensors are shared and copied from the first
    model. A router weight is added for each expert group::

        model.layers.0.mlp.gate_proj.weight        (from expert model 2)
            -> model.layers.0.mlp.experts.1.gate_proj.weight
        model.layers.0.mlp.router.weight           (n_experts x hidden_size)

    This naming is MergeKit's own convention — the result needs a
    MoE-aware architecture to load.

    Parameters (passed to :meth:`merge`):
        expert_pattern: Regex selecting which tensor names become
            experts. Default: ``mlp|feed_forward|ffn``.
        router_init: ``"random"`` (scaled normal) or ``"zeros"``.
        router_scale: Std-dev for ``"random"`` init. Default 0.02.
        hidden_size: Router input dimension. Inferred from a shared
            ``*embed*`` tensor when not given.
    """

    name = "moe"

    def merge(
        self,
        models: list[dict],
        expert_pattern: str = _DEFAULT_EXPERT_PATTERN,
        router_init: str = "random",
        router_scale: float = 0.02,
        hidden_size: int | None = None,
    ) -> dict:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for merging: pip install torch"
            ) from exc

        _check_compatible(models)
        pattern = re.compile(expert_pattern)
        base = models[0]

        if hidden_size is None:
            hidden_size, dtype = _infer_hidden_size(base, pattern)
        else:
            dtype = next(iter(base.values())).dtype

        out = {}
        router_keys = set()
        for name, tensor in base.items():
            match = pattern.search(name)
            if match is None:
                out[name] = tensor
                continue
            prefix = name[: match.end()]
            router_keys.add(f"{prefix}.router.weight")
            for i, state_dict in enumerate(models):
                out[f"{prefix}.experts.{i}{name[match.end():]}"] = state_dict[name]

        if router_init not in ("random", "zeros"):
            raise ValueError(
                f"unknown router_init {router_init!r} (use 'random' or 'zeros')"
            )
        num_experts = len(models)
        for key in sorted(router_keys):
            if router_init == "zeros":
                out[key] = torch.zeros(num_experts, hidden_size, dtype=dtype)
            else:
                out[key] = (
                    torch.randn(num_experts, hidden_size, dtype=dtype)
                    * router_scale
                )
        return out


def _check_compatible(models: list[dict]) -> None:
    """Ensure all state dicts have identical tensor names and shapes."""
    if len(models) < 2:
        raise ValueError("MoE merge needs at least 2 models")
    ref = models[0]
    for i, state_dict in enumerate(models[1:], start=1):
        missing = set(ref) - set(state_dict)
        extra = set(state_dict) - set(ref)
        if missing or extra:
            raise ValueError(
                f"model {i} has different tensor names "
                f"(missing: {len(missing)}, extra: {len(extra)})"
            )
        for name, tensor in ref.items():
            if tuple(state_dict[name].shape) != tuple(tensor.shape):
                raise ValueError(
                    f"model {i} shape mismatch on {name!r}: "
                    f"{tuple(tensor.shape)} vs "
                    f"{tuple(state_dict[name].shape)}"
                )


def _infer_hidden_size(base: dict, pattern: re.Pattern):
    """Infer router input dim from a shared ``*embed*`` tensor."""
    for name, tensor in base.items():
        if "embed" in name and pattern.search(name) is None and tensor.ndim >= 2:
            return tensor.shape[-1], tensor.dtype
    raise ValueError(
        "could not infer hidden_size (no shared '*embed*' tensor found); "
        "pass hidden_size explicitly"
    )
