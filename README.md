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

## Results

One will get results similar to this -

```
  [forward] Layer 59/60 [moe]
    [layer 59] MLA attention (seq_len=1, pos=166)
    [layer 59] MoE block (routing to 8 of 256 experts + 1 shared)
      [moe L59] token[0,0] -> experts [74, 26, 151, 15, 81, 239, 147, 24] (weights ['0.216', '0.165', '0.143', '0.128', '0.107', '0.092', '0.074', '0.074'])
      [gpu_transfer] L59: transferring experts [74, 26, 151, 15, 81, 239, 147, 24] (8 slots) host->GPU via async stream
        [expert_cache] MISS  L59/E074 — loading from disk (cache 256.0/256.0 GB, 6240 entries)
        [expert_cache] HIT   L59/E026
        [expert_cache] HIT   L59/E151
        [expert_cache] HIT   L59/E015
        [expert_cache] HIT   L59/E081
        [expert_cache] HIT   L59/E239
        [expert_cache] HIT   L59/E147
        [expert_cache] HIT   L59/E024
      [gpu_transfer] L59: transfer sync done
  [forward] Layer 60/60 [moe]
    [layer 60] MLA attention (seq_len=1, pos=166)
    [layer 60] MoE block (routing to 8 of 256 experts + 1 shared)
      [moe L60] token[0,0] -> experts [35, 122, 176, 217, 57, 209, 211, 38] (weights ['0.211', '0.176', '0.159', '0.157', '0.084', '0.073', '0.070', '0.070'])
      [gpu_transfer] L60: transferring experts [35, 122, 176, 217, 57, 209, 211, 38] (8 slots) host->GPU via async stream
        [expert_cache] HIT   L60/E035
        [expert_cache] HIT   L60/E122
        [expert_cache] MISS  L60/E176 — loading from disk (cache 256.0/256.0 GB, 6240 entries)
        [expert_cache] HIT   L60/E217
        [expert_cache] HIT   L60/E057
        [expert_cache] HIT   L60/E209
        [expert_cache] HIT   L60/E211
        [expert_cache] HIT   L60/E038
      [gpu_transfer] L60: transfer sync done
  [forward] Applying final norm + lm_head projection
[generate] EOS at step 150
decoding:  29%|███████████████████████████████████▊                                                                                       | 149/511 [22:44<55:14,  9.16s/tok, tok='!', tok_s=0.1, total=150]
[generate] Done. 150 tokens in 1364.45s (0.1 tok/s)
Q:
what are the top 5 largest cities in the world in terms of population?


------------

Here are the top 5 largest cities in the world by population as of 2023:

1. **Tokyo, Japan** - Approximately 37.4 million people in the Greater Tokyo Area.
2. **Delhi, India** - Around 32 million people in the National Capital Region.
3. **Shanghai, China** - About 29.2 million people in the metropolitan area.
4. **São Paulo, Brazil** - Roughly 22.6 million people in the metropolitan region.
5. **Mumbai, India** - Around 21.3 million people in the metropolitan area.

These figures can vary slightly depending on the definition of city limits versus metropolitan areas. Let me know if you'd like more details!
```
