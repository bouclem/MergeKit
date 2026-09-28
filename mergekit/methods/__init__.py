"""Merge method implementations."""

from mergekit.methods.base import MergeMethod
from mergekit.methods.moe import MoEMerge

METHODS: dict[str, type[MergeMethod]] = {
    "moe": MoEMerge,
}


def get_method(name: str) -> MergeMethod:
    """Instantiate a merge method by name."""
    try:
        return METHODS[name]()
    except KeyError:
        raise ValueError(
            f"unknown merge method {name!r} "
            f"(available: {', '.join(sorted(METHODS))})"
        ) from None


__all__ = ["MergeMethod", "MoEMerge", "METHODS", "get_method"]
