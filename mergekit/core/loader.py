"""Model weight loading utilities.

Loads model state dicts from a single weight file or a directory of
shards. Supports .safetensors and PyTorch formats (.pt, .pth, .bin,
.ckpt).

torch and safetensors are imported lazily so that ``import mergekit``
does not require them to be installed.
"""

from pathlib import Path

_TORCH_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt"}


def load_state_dict(path: str | Path) -> dict:
    """Load a model state dict from a single weight file.

    Args:
        path: Path to a .safetensors, .pt, .pth, .bin, or .ckpt file.

    Returns:
        A dict mapping tensor names to tensors, on CPU.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file extension is unsupported or the file
            does not contain a state dict.
        ImportError: If the required backend (torch/safetensors) is
            not installed.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load .safetensors files: "
                "pip install safetensors"
            ) from exc
        return load_file(str(path))

    if suffix in _TORCH_SUFFIXES:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to load PyTorch weight files: "
                "pip install torch"
            ) from exc
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError(
                f"Expected a state dict in {path}, "
                f"got {type(state).__name__}"
            )
        return state

    raise ValueError(
        f"Unsupported weight file format: {path.suffix!r} "
        f"(supported: .safetensors, {', '.join(sorted(_TORCH_SUFFIXES))})"
    )


def load_model(path: str | Path) -> dict:
    """Load model weights from a file or a directory of shards.

    When given a directory, every supported weight file inside is
    loaded and merged into a single state dict
    (e.g. model-00001-of-00003.safetensors shards).

    Args:
        path: Path to a weight file or a model directory.

    Returns:
        A dict mapping tensor names to tensors, on CPU.

    Raises:
        FileNotFoundError: If the path does not exist or the directory
            contains no supported weight files.
        ValueError: If a file extension is unsupported.
    """
    path = Path(path)
    if path.is_dir():
        files = sorted(
            f
            for f in path.iterdir()
            if f.is_file()
            and (f.suffix.lower() == ".safetensors" or f.suffix.lower() in _TORCH_SUFFIXES)
        )
        if not files:
            raise FileNotFoundError(
                f"No supported weight files found in directory: {path}"
            )
        state = {}
        for file in files:
            state.update(load_state_dict(file))
        return state
    return load_state_dict(path)
