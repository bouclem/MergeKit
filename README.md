# MergeKit

A toolkit for merging AI models — combine the weights of multiple
fine-tuned models into a single model.

> Early development (v0.0.3). The API is unstable.

## Features

- Merge methods: **MoE** (mixture-of-experts, frankenMoE style)
- Planned: linear weight averaging, SLERP, TIES, task arithmetic, ...
- Model weight loading (`.safetensors`, `.pt`, `.bin`, sharded dirs)
- CLI for inspecting and merging models

## Installation

```bash
pip install -e .
```

## Usage

```bash
mergekit --version
mergekit info path/to/model.safetensors   # list tensors in a weight file
mergekit info path/to/model-dir/          # or a directory of shards

# MoE merge: each model's MLP tensors become a separate expert,
# shared tensors come from the first model
mergekit merge model_a.safetensors model_b.safetensors \
    -o merged.safetensors --method moe

# method parameters via repeatable --param key=value
mergekit merge a.safetensors b.safetensors -o out.pt \
    --param expert_pattern="mlp" --param router_init=zeros
```

## Project structure

```
mergekit/
├── cli.py      # Command-line interface
├── methods/    # Merge method implementations
│   ├── base.py     # MergeMethod ABC + registry
│   └── moe.py      # MoE (mixture-of-experts) merge
├── core/       # Merging engine and model I/O
│   ├── loader.py   # State dict loading (.safetensors, .pt, .bin, ...)
│   └── saver.py    # State dict saving
└── utils/      # Shared helpers
tests/          # Test suite
examples/       # Usage examples
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

MIT — see [LICENSE](LICENSE).
