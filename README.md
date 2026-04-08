# DeepSeek-R1 Single-GPU Inference Engine

Streaming inference engine for DeepSeek-R1-0528 (685B, FP8) on a single GPU.
Expert weights load on demand from disk; non-expert weights (~40 GB) stay on GPU.

## Prerequisites

- Python 3.12+
- CUDA-capable GPU with sufficient VRAM (H200, H100, etc.)
- ~700 GB disk space for model weights
- [uv](https://docs.astral.sh/uv/) package manager

## Setup

```bash
uv venv --python 3.12.12
uv pip install torch safetensors transformers einops huggingface_hub
```

## Download the model

```bash
# Preview what will be downloaded (~689 GB)
uv run python download_model.py --dry-run

# Download to default location (~/models/DeepSeek-R1-0528)
uv run python download_model.py

# Or specify a path
uv run python download_model.py --output-dir /data/models/DeepSeek-R1-0528
```

## Run

```python
from inference.engine import InferenceEngine

engine = InferenceEngine(
    model_path="~/models/DeepSeek-R1-0528",
    expert_cache_gb=16.0,  # pinned host RAM for expert LRU cache
)

print(engine.generate("What is the sum of the first 100 prime numbers?"))
```

`expert_cache_gb` controls how much host RAM is used to cache recently-used
experts. Higher values reduce disk reads but require more RAM.
