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
        verbose: bool = False,
    ):
        self.device = torch.device(device)
        self.verbose = verbose

        if verbose:
            print(f"[init] Loading config from {model_path}")
        self.config = load_config(model_path)
        if verbose:
            print(f"[init] Config: {self.config.num_layers} layers, hidden={self.config.hidden_size}, "
                  f"experts={self.config.n_routed_experts} routed + {self.config.n_shared_experts} shared, "
                  f"top_k={self.config.num_experts_per_tok}")

        self.max_seq_len = max_seq_len

        if verbose:
            print(f"[init] Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if verbose:
            print(f"[init] Tokenizer loaded (vocab size={self.config.vocab_size})")

        if verbose:
            print(f"[init] Loading weights (expert cache={expert_cache_gb:.1f} GB, device={device})")
        self.weight_store = WeightStore(
            model_path, self.config, self.device,
            expert_cache_bytes=int(expert_cache_gb * (1024 ** 3)),
            verbose=verbose,
        )
        if verbose:
            print(f"[init] Weight store ready — {len(self.weight_store.gpu_weights)} non-expert tensors on GPU")

        if verbose:
            print(f"[init] Allocating GPU transfer buffers for {self.config.num_experts_per_tok} expert slots")
        self.gpu_transfer = GPUTransfer(self.config, self.weight_store, self.device, verbose=verbose)

        if verbose:
            print(f"[init] Precomputing RoPE frequencies (max_seq_len={max_seq_len})")
        self.rope_freqs = precompute_rope_freqs(self.config, max_seq_len, self.device)

        if verbose:
            print(f"[init] Allocating KV caches for {self.config.num_layers} layers "
                  f"(kv_lora_rank={self.config.kv_lora_rank}, rope_dim={self.config.qk_rope_head_dim})")
        self.kv_caches = self._init_kv_caches()
        if verbose:
            print("[init] Engine ready.")

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

        if self.verbose:
            print(f"[generate] Prompt encoded to {seq_len} tokens (max_new={max_tokens}, "
                  f"temp={temperature}, top_p={top_p})")

        # Prefill: process entire prompt
        if self.verbose:
            print(f"[generate] Prefill: running forward pass over {seq_len} prompt tokens ...")
        logits = forward_pass(
            input_ids, self.kv_caches, 0,
            self.weight_store, self.gpu_transfer, self.config, self.rope_freqs,
            verbose=self.verbose,
        )
        next_token = sample(logits[:, -1, :], temperature, top_p)
        generated = [next_token.item()]
        if self.verbose:
            first_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
            print(f"[generate] Prefill done. First token: {next_token.item()!r} ({first_text!r})")

        # Decode: one token at a time
        if self.verbose:
            print(f"[generate] Decode: generating up to {max_tokens - 1} more tokens ...")
        for step in range(max_tokens - 1):
            position = seq_len + step
            logits = forward_pass(
                next_token.unsqueeze(0), self.kv_caches, position,
                self.weight_store, self.gpu_transfer, self.config, self.rope_freqs,
                verbose=self.verbose,
            )
            next_token = sample(logits[:, -1, :], temperature, top_p)

            if next_token.item() == self.tokenizer.eos_token_id:
                if self.verbose:
                    print(f"[generate] EOS token hit at step {step + 1}. Stopping.")
                break

            generated.append(next_token.item())
            if self.verbose:
                tok_text = self.tokenizer.decode([next_token.item()], skip_special_tokens=True)
                print(f"[generate] step {step + 1}/{max_tokens - 1}: token {next_token.item()!r} ({tok_text!r}), "
                      f"total generated={len(generated)}")

        if self.verbose:
            print(f"[generate] Done. Generated {len(generated)} tokens total.")
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
