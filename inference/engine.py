from pathlib import Path

import torch
from transformers import AutoTokenizer

from .config import DeepSeekConfig, load_config
from .forward import forward_pass
from .gpu_transfer import GPUTransfer
from .ops import precompute_rope_freqs
from .weight_store import WeightStore


class InferenceEngine:
    """Top-level entry point: tokenize, generate, decode."""

    def __init__(
        self,
        model_path: str | Path,
        max_seq_len: int = 4096,
        device: str = "cuda:0",
        expert_cache_gb: float = 16.0,
    ):
        self.device = torch.device(device)
        self.config = load_config(model_path)
        self.max_seq_len = max_seq_len

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.weight_store = WeightStore(
            model_path, self.config, self.device,
            expert_cache_bytes=int(expert_cache_gb * (1024 ** 3)),
        )
        self.gpu_transfer = GPUTransfer(self.config, self.weight_store, self.device)

        self.rope_freqs = precompute_rope_freqs(self.config, max_seq_len, self.device)
        self.kv_caches = self._init_kv_caches()

    def _init_kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Allocate KV caches for all layers.

        Each layer stores (compressed_kv, k_pe):
          compressed_kv: (1, max_seq_len, kv_lora_rank)
          k_pe: (1, max_seq_len, qk_rope_head_dim)
        """
        return [
            (
                torch.zeros(1, self.max_seq_len, self.config.kv_lora_rank,
                            dtype=torch.bfloat16, device=self.device),
                torch.zeros(1, self.max_seq_len, self.config.qk_rope_head_dim,
                            dtype=torch.bfloat16, device=self.device),
            )
            for _ in range(self.config.num_layers)
        ]

    def reset_kv_cache(self):
        for ckv, kpe in self.kv_caches:
            ckv.zero_()
            kpe.zero_()

    @torch.no_grad()
    def generate(
        self, prompt: str, max_tokens: int = 512, temperature: float = 0.6, top_p: float = 0.9,
    ) -> str:
        self.reset_kv_cache()
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        seq_len = input_ids.shape[1]

        # Prefill: process entire prompt
        logits = forward_pass(
            input_ids, self.kv_caches, 0,
            self.weight_store, self.gpu_transfer, self.config, self.rope_freqs,
        )
        next_token = sample(logits[:, -1, :], temperature, top_p)
        generated = [next_token.item()]

        # Decode: one token at a time
        for step in range(max_tokens - 1):
            position = seq_len + step
            logits = forward_pass(
                next_token.unsqueeze(0), self.kv_caches, position,
                self.weight_store, self.gpu_transfer, self.config, self.rope_freqs,
            )
            next_token = sample(logits[:, -1, :], temperature, top_p)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

            generated.append(next_token.item())

        return self.tokenizer.decode(generated, skip_special_tokens=True)


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample next token with temperature and nucleus (top-p) sampling.

    logits: (B, vocab_size)
    Returns: (B,) sampled token IDs
    """
    if temperature == 0:
        return logits.argmax(dim=-1)

    probs = torch.softmax(logits / temperature, dim=-1)

    # Top-p (nucleus) filtering
    sorted_probs, sorted_indices = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    mask = cumulative - sorted_probs > top_p
    sorted_probs[mask] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # Sample from filtered distribution
    token_idx = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, token_idx).squeeze(-1)
