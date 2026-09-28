"""Model weight saving utilities.

torch and safetensors are imported lazily so that ``import mergekit``
does not require them to be installed.
"""

from pathlib import Path

_TORCH_SUFFIXES = {".pt", ".pth", ".bin", ".ckpt"}


def save_state_dict(state_dict: dict, path: str | Path) -> Path:
    """Save a state dict to a .safetensors or PyTorch weight file.

    Parent directories are created as needed.

    Args:
        state_dict: Tensor name -> tensor mapping.
        path: Output file (.safetensors, .pt, .pth, .bin, .ckpt).

    Returns:
        The output path.

    Raises:
        ValueError: If the file extension is unsupported.
        ImportError: If the required backend is not installed.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    if suffix == ".safetensors":
        try:
            from safetensors.torch import save_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to save .safetensors files: "
                "pip install safetensors"
            ) from exc
        # safetensors requires contiguous tensors
        contiguous = {
            name: tensor if tensor.is_contiguous() else tensor.contiguous()
            for name, tensor in state_dict.items()
        }
        save_file(contiguous, str(path))
        return path

    if suffix in _TORCH_SUFFIXES:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required to save PyTorch weight files: "
                "pip install torch"
            ) from exc
        torch.save(state_dict, path)
        return path

    raise ValueError(
        f"Unsupported output format: {path.suffix!r} "
        f"(supported: .safetensors, {', '.join(sorted(_TORCH_SUFFIXES))})"
    )
