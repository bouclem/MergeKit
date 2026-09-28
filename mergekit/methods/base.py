"""Base interface for merge methods."""

from abc import ABC, abstractmethod


class MergeMethod(ABC):
    """Combines multiple model state dicts into one.

    Subclasses implement :meth:`merge`, which takes a list of state
    dicts (tensor name -> tensor) and returns the merged state dict.
    """

    name: str = "base"

    @abstractmethod
    def merge(self, models: list[dict], **params) -> dict:
        """Merge ``models`` into a single state dict.

        Args:
            models: State dicts to merge (at least two).
            **params: Method-specific options.

        Returns:
            The merged state dict.
        """
