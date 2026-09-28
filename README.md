# MergeKit

A toolkit for merging AI models — combine the weights of multiple
fine-tuned models into a single model.

> Early development (v0.0.2). No merge methods are implemented yet and
> the API is unstable.

## Planned features

- Multiple merge methods (linear weight averaging, SLERP, TIES,
  task arithmetic, ...)
- Config-driven merge recipes
- `merge` CLI command (currently a stub)

## Installation

```bash
pip install -e .
```

## Usage

```bash
mergekit --version
mergekit info path/to/model.safetensors   # list tensors in a weight file
mergekit info path/to/model-dir/          # or a directory of shards
mergekit merge model_a model_b            # not implemented yet
```

## Project structure

```
mergekit/
├── cli.py      # Command-line interface
├── methods/    # Merge method implementations (coming soon)
├── core/       # Merging engine and model loading
│   └── loader.py   # State dict loading (.safetensors, .pt, .bin, ...)
└── utils/      # Shared helpers
tests/          # Test suite
examples/       # Usage examples
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
