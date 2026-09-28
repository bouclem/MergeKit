"""Command-line interface for MergeKit."""

import argparse
import sys
from pathlib import Path

from mergekit import __version__
from mergekit.core import load_model, save_state_dict
from mergekit.methods import METHODS, get_method


def _parse_params(raw: list[str]) -> dict:
    """Parse repeatable ``--param key=value`` items, coercing numbers."""
    params = {}
    for item in raw:
        if "=" not in item:
            raise ValueError(f"invalid --param {item!r} (expected key=value)")
        key, value = item.split("=", 1)
        try:
            params[key] = int(value)
        except ValueError:
            try:
                params[key] = float(value)
            except ValueError:
                params[key] = value
    return params


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


def _cmd_merge(
    model_paths: list[Path],
    output: Path,
    method_name: str,
    params: list[str],
) -> int:
    method = get_method(method_name)
    models = [load_model(path) for path in model_paths]
    merged = method.merge(models, **_parse_params(params))
    out = save_state_dict(merged, output)
    print(f"merged {len(models)} models -> {out} ({len(merged)} tensors)")
    return 0


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
        "merge", help="merge model weight files into one"
    )
    p_merge.add_argument(
        "models", nargs="+", type=Path, help="model files or directories"
    )
    p_merge.add_argument(
        "-o", "--output", type=Path, required=True,
        help="output weight file (.safetensors, .pt, ...)",
    )
    p_merge.add_argument(
        "--method", default="moe", choices=sorted(METHODS),
        help="merge method (default: moe)",
    )
    p_merge.add_argument(
        "--param", action="append", default=[], metavar="KEY=VALUE",
        help="method parameter, repeatable "
             "(e.g. --param router_init=zeros)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "info":
            return _cmd_info(args.model)
        if args.command == "merge":
            return _cmd_merge(args.models, args.output, args.method, args.param)
    except (FileNotFoundError, ValueError, ImportError, TypeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
