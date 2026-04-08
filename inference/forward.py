import torch

from .config import DeepSeekConfig
from .gpu_transfer import GPUTransfer
from .ops import (
    dense_ffn,
    expert_ffn,
    fp8_linear,
    mla_attention,
    precompute_rope_freqs,
    rms_norm,
    route,
)
from .weight_store import WeightStore


def forward_pass(
    input_ids: torch.Tensor,
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
    position: int,
    weight_store: WeightStore,
    gpu_transfer: GPUTransfer,
    config: DeepSeekConfig,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    """Run the full forward pass through all 61 layers.

    input_ids: (B, S) token IDs
    kv_caches: list of (compressed_kv, k_pe) tuples per layer
    position: current sequence position (for KV cache indexing)
    rope_freqs: precomputed RoPE frequencies

    Returns: (B, S, vocab_size) logits
    """
    # Embed
    embed_weight = weight_store.get("model.embed_tokens.weight")
    hidden = torch.nn.functional.embedding(input_ids, embed_weight)  # (B, S, hidden_size)

    for layer_idx in range(config.num_layers):
        hidden = transformer_layer(
            hidden, layer_idx, kv_caches[layer_idx], position,
            weight_store, gpu_transfer, config, rope_freqs,
        )

    # Final norm + lm_head
    final_norm = weight_store.get("model.norm.weight")
    hidden = rms_norm(hidden, final_norm, config.rms_norm_eps)

    lm_head_w = weight_store.get("lm_head.weight")
    # lm_head is typically not FP8, just a regular matmul
    return hidden.float() @ lm_head_w.float().T


def transformer_layer(
    hidden: torch.Tensor,
    layer_idx: int,
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    position: int,
    weight_store: WeightStore,
    gpu_transfer: GPUTransfer,
    config: DeepSeekConfig,
    rope_freqs: torch.Tensor,
) -> torch.Tensor:
    input_ln, post_attn_ln = weight_store.get_layer_norm_weights(layer_idx)
    attn_weights = weight_store.get_layer_attn_weights(layer_idx)

    # Attention block
    normed = rms_norm(hidden, input_ln, config.rms_norm_eps)
    attn_out = mla_attention(normed, attn_weights, kv_cache, position, rope_freqs, config)
    hidden = hidden + attn_out

    # FFN block
    normed = rms_norm(hidden, post_attn_ln, config.rms_norm_eps)

    if layer_idx < config.first_k_dense_replace:
        ffn_out = dense_ffn(normed, weight_store.get_dense_mlp_weights(layer_idx))
    else:
        ffn_out = moe_block(normed, layer_idx, weight_store, gpu_transfer, config)

    return hidden + ffn_out


def moe_block(
    hidden: torch.Tensor,
    layer_idx: int,
    weight_store: WeightStore,
    gpu_transfer: GPUTransfer,
    config: DeepSeekConfig,
) -> torch.Tensor:
    """MoE block: route -> load experts -> compute -> combine."""
    gate_weight = weight_store.get_gate_weight(layer_idx)
    expert_indices, expert_weights = route(hidden, gate_weight, config)
    # expert_indices: (B, S, top_k), expert_weights: (B, S, top_k)

    # Shared expert (always on GPU)
    shared_weights = weight_store.get_shared_expert_weights(layer_idx)
    shared_out = expert_ffn(hidden, shared_weights)

    # Routed experts: process each token's selected experts
    B, S, top_k = expert_indices.shape
    routed_out = torch.zeros_like(hidden)

    for b in range(B):
        for s in range(S):
            token_hidden = hidden[b, s].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            token_indices = expert_indices[b, s]  # (top_k,)
            token_weights = expert_weights[b, s]  # (top_k,)

            # Transfer this token's experts from host to GPU
            gpu_experts = gpu_transfer.transfer_experts(layer_idx, token_indices)

            for k in range(top_k):
                e_out = expert_ffn(token_hidden, gpu_experts[k])  # (1, 1, D)
                routed_out[b, s] += token_weights[k] * e_out.squeeze(0).squeeze(0)

    return shared_out + routed_out * config.routed_scaling_factor
