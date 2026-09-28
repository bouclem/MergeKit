"""Command-line interface for MergeKit."""

import argparse
import sys
from pathlib import Path

from mergekit import __version__
from mergekit.core import load_model


def _cmd_info(model_path: Path) -> int:
    state = load_model(model_path)
    total_params = 0
    print(f"Model: {model_path}  ({len(state)} tensors)\n")
    print(f"{'name':<60} {'shape':<24} {'dtype':<12} {'params':>14}")
    for name in sorted(state):
        tensor = state[name]
        numel = tensor.numel()
        total_params += numel
        print(
            f"{name:<60} {str(tuple(tensor.shape)):<24} "
            f"{str(tensor.dtype):<12} {numel:>14,}"
        )
    print(f"\nTotal parameters: {total_params:,}")
    return 0


def _cmd_merge(models: list[Path]) -> int:
    print(
        "merge: no merge methods implemented yet "
        "(coming in a future release)",
        file=sys.stderr,
    )
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mergekit",
        description="MergeKit - a toolkit for merging AI models",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_info = sub.add_parser(
        "info", help="show tensor info for a model file or directory"
    )
    p_info.add_argument(
        "model", type=Path, help="path to a weight file or model directory"
    )

    p_merge = sub.add_parser(
        "merge", help="merge models (not implemented yet)"
    )
    p_merge.add_argument(
        "models", nargs="+", type=Path, help="model files or directories"
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            return _cmd_info(args.model)
        if args.command == "merge":
            return _cmd_merge(args.models)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
