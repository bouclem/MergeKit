# MergeKit

A toolkit for merging AI models — combine the weights of multiple
fine-tuned models into a single model.

> Early development (v0.0.1). Scaffolding only — no merge methods are
> implemented yet and the API is unstable.

## Planned features

- Multiple merge methods (linear weight averaging, SLERP, TIES,
  task arithmetic, ...)
- Config-driven merge recipes
- CLI for running merges

## Installation

```bash
pip install -e .
```

## Project structure

```
mergekit/
├── methods/    # Merge method implementations (coming soon)
├── core/       # Merging engine and model loading (coming soon)
└── utils/      # Shared helpers
tests/          # Test suite
examples/       # Usage examples
```

## Development

```bash
pip install -e ".[dev]"
pytest
```
