# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.0.3] - 2026-09-28

### Added

- MIT license (`LICENSE`, copyright Sqersters)
- First merge method: `MoEMerge` (mixture-of-experts / frankenMoE style) in `mergekit.methods.moe`
- `MergeMethod` base class and `METHODS` registry with `get_method()`
- `mergekit.core.saver.save_state_dict()` for writing merged weights
- `merge` CLI command now performs merges: `--method`, `-o/--output`, repeatable `--param key=value`
- `tests/test_moe.py`

## [0.0.2] - 2026-09-28

### Added

- Simple CLI (`mergekit` entry point) with `info` and `merge` (stub) subcommands
- `mergekit.core.loader` with `load_state_dict()` (`.safetensors`, `.pt`, `.pth`, `.bin`, `.ckpt`) and `load_model()` (file or sharded directory)
- Runtime dependencies: `torch>=2.0`, `safetensors>=0.4`

## [0.0.1] - 2026-09-28

### Added

- Initial project scaffolding
- `pyproject.toml` (setuptools backend)
- `README.md` and `CHANGELOG.md`
- `.gitignore` for Python, IDE, and model artifacts
- Package structure: `mergekit/` with `methods/`, `core/`, and `utils/` subpackages
- `tests/` and `examples/` folders (`.gitkeep` placeholder)
