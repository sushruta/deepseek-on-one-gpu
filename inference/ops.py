import math

import torch
import torch.nn.functional as F
from einops import rearrange

from .config import DeepSeekConfig


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    return (x.float() * norm).to(x.dtype) * weight


def fp8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """FP8 block-quantized linear: dequantize weight blocks then matmul.

    weight: (out_features, in_features) in float8_e4m3fn
    scale_inv: (ceil(out_features/block), ceil(in_features/block)) in float32
    x: (*, in_features) in bfloat16/float16

    Block dequant on GPU: for each 128x128 block, multiply FP8 values by
    the corresponding scale_inv entry to recover approximate bfloat16 values.
    """
    out_features, in_features = weight.shape
    block = 128

    # Dequantize: expand scale_inv to match weight shape
    # scale_inv has shape (ceil(O/128), ceil(I/128))
    n_blocks_out = math.ceil(out_features / block)
    n_blocks_in = math.ceil(in_features / block)

    # Repeat each scale to cover its 128-element block, then trim
    # (n_blocks_out, n_blocks_in) -> (out_features, in_features)
    scale_expanded = scale_inv.repeat_interleave(block, dim=0)[:out_features]
    scale_expanded = scale_expanded.repeat_interleave(block, dim=1)[:, :in_features]

    w_dequant = weight.float() * scale_expanded
    return (x.float() @ w_dequant.T).to(out_dtype)


def yarn_get_mscale(scale: float = 1.0, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def precompute_rope_freqs(config: DeepSeekConfig, seq_len: int, device: torch.device) -> torch.Tensor:
    """Precompute YaRN-scaled RoPE frequencies.

    Returns: (seq_len, qk_rope_head_dim // 2) complex frequencies
    """
    dim = config.qk_rope_head_dim
    base = config.rope_theta
    factor = config.rope_scaling_factor
    beta_fast = config.rope_scaling_beta_fast
    beta_slow = config.rope_scaling_beta_slow
    original_max_pos = config.original_max_position_embeddings

    # Compute per-dimension wavelengths and interpolation ramps
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))

    # YaRN ramp: linear interpolation between low-freq (scaled) and high-freq (unscaled)
    low = max(math.floor(dim * math.log(original_max_pos / (beta_fast * 2 * math.pi)) / (2 * math.log(base))), 0)
    high = min(math.ceil(dim * math.log(original_max_pos / (beta_slow * 2 * math.pi)) / (2 * math.log(base))), dim // 2 - 1)

    smooth = torch.ones(dim // 2, device=device, dtype=torch.float32)
    if high > low:
        dims = torch.arange(dim // 2, device=device, dtype=torch.float32)
        smooth = ((dims - low) / (high - low)).clamp(0.0, 1.0)

    # Interpolate: high-freq dims keep original freq, low-freq dims get scaled down
    freqs = freqs * (1.0 - smooth) + (freqs / factor) * smooth

    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    # (seq_len, dim//2)
    angles = torch.outer(positions, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to x.

    x: (B, H, S, rope_dim)
    freqs: (S, rope_dim // 2)
    """
    # Split into pairs and view as complex
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # (1, 1, S, rope_dim//2) broadcast
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    rotated = x_complex * freqs
    return torch.view_as_real(rotated).flatten(-2).to(x.dtype)


def mla_attention(
    hidden: torch.Tensor,
    attn_weights: dict[str, tuple[torch.Tensor, torch.Tensor]],
    kv_cache: tuple[torch.Tensor, torch.Tensor],
    position: int,
    rope_freqs: torch.Tensor,
    config: DeepSeekConfig,
) -> torch.Tensor:
    """Multi-head Latent Attention.

    attn_weights: dict mapping name -> (weight_fp8, scale_inv) for each projection
    kv_cache: (compressed_kv_cache, k_pe_cache)
        compressed_kv_cache: (B, max_seq, kv_lora_rank)
        k_pe_cache: (B, max_seq, qk_rope_head_dim)

    Q path: hidden -> q_a_proj -> q_a_layernorm -> q_b_proj -> split(q_nope, q_pe) -> rope(q_pe)
    KV path: hidden -> kv_a_proj_with_mqa -> split(compressed_kv, k_pe) -> rope(k_pe)
             compressed_kv -> kv_a_layernorm -> kv_b_proj -> split(k_nope, v)
    """
    B, S, D = hidden.shape
    num_heads = config.num_attention_heads
    qk_nope_dim = config.qk_nope_head_dim
    qk_rope_dim = config.qk_rope_head_dim
    v_dim = config.v_head_dim
    kv_lora = config.kv_lora_rank

    compressed_kv_cache, k_pe_cache = kv_cache

    # Q path
    q_a = fp8_linear(hidden, *attn_weights["q_a_proj"])  # (B, S, q_lora_rank)
    q_a = rms_norm(q_a, attn_weights["q_a_layernorm"], config.rms_norm_eps)
    q = fp8_linear(q_a, *attn_weights["q_b_proj"])  # (B, S, num_heads * qk_head_dim)
    q = rearrange(q, "B S (H D) -> B H S D", H=num_heads)
    q_nope, q_pe = q.split([qk_nope_dim, qk_rope_dim], dim=-1)

    # KV path
    kv_a = fp8_linear(hidden, *attn_weights["kv_a_proj_with_mqa"])  # (B, S, kv_lora_rank + qk_rope_head_dim)
    compressed_kv, k_pe = kv_a.split([kv_lora, qk_rope_dim], dim=-1)

    # Cache compressed_kv and k_pe
    compressed_kv_cache[:B, position:position + S] = compressed_kv
    k_pe_cache[:B, position:position + S] = k_pe

    # Read full cache up to current position
    cached_ckv = compressed_kv_cache[:B, :position + S]  # (B, T, kv_lora_rank)
    cached_k_pe = k_pe_cache[:B, :position + S]  # (B, T, qk_rope_head_dim)

    # Decompress KV
    kv_a_normed = rms_norm(cached_ckv, attn_weights["kv_a_layernorm"], config.rms_norm_eps)
    kv_b = fp8_linear(kv_a_normed, *attn_weights["kv_b_proj"])  # (B, T, num_heads * (qk_nope_head_dim + v_head_dim))
    kv_b = rearrange(kv_b, "B T (H D) -> B H T D", H=num_heads)
    k_nope, v = kv_b.split([qk_nope_dim, v_dim], dim=-1)

    # Apply RoPE
    rope_start = position
    rope_end = position + S
    q_pe = apply_rope(q_pe, rope_freqs[rope_start:rope_end])

    cached_k_pe_heads = rearrange(cached_k_pe, "B T D -> B 1 T D").expand(-1, num_heads, -1, -1)
    cached_k_pe_heads = apply_rope(cached_k_pe_heads, rope_freqs[:position + S])

    # Concatenate nope and rope parts for full Q and K
    q_full = torch.cat([q_nope, q_pe], dim=-1)  # (B, H, S, qk_head_dim)
    k_full = torch.cat([k_nope, cached_k_pe_heads], dim=-1)  # (B, H, T, qk_head_dim)

    # Attention: softmax(Q @ K^T / sqrt(qk_head_dim)) @ V
    mscale = yarn_get_mscale(config.rope_scaling_factor, config.rope_scaling_mscale_all_dim)
    scale = mscale / math.sqrt(config.qk_head_dim)

    attn_scores = torch.matmul(q_full, k_full.transpose(-2, -1)) * scale  # (B, H, S, T)

    # Causal mask
    if S > 1:
        causal_mask = torch.triu(
            torch.full((S, position + S), float("-inf"), device=hidden.device),
            diagonal=position + 1,
        )
        attn_scores = attn_scores + causal_mask.unsqueeze(0).unsqueeze(0)

    attn_probs = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(hidden.dtype)
    attn_out = torch.matmul(attn_probs, v)  # (B, H, S, v_head_dim)

    attn_out = rearrange(attn_out, "B H S D -> B S (H D)")
    return fp8_linear(attn_out, *attn_weights["o_proj"])  # (B, S, hidden_size)


def swiglu(x: torch.Tensor, gate_proj: torch.Tensor, gate_scale: torch.Tensor,
           up_proj: torch.Tensor, up_scale: torch.Tensor,
           down_proj: torch.Tensor, down_scale: torch.Tensor) -> torch.Tensor:
    """SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))"""
    gate = F.silu(fp8_linear(x, gate_proj, gate_scale))
    up = fp8_linear(x, up_proj, up_scale)
    return fp8_linear(gate * up, down_proj, down_scale)


def dense_ffn(hidden: torch.Tensor, mlp_weights: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return swiglu(
        hidden,
        *mlp_weights["gate_proj"], *mlp_weights["up_proj"], *mlp_weights["down_proj"],
    )


def expert_ffn(hidden: torch.Tensor, expert_weights: dict[str, tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    return swiglu(
        hidden,
        *expert_weights["gate_proj"], *expert_weights["up_proj"], *expert_weights["down_proj"],
    )


def route(
    hidden: torch.Tensor,
    gate_weight: torch.Tensor,
    config: DeepSeekConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid-based routing with group-limited top-k selection.

    hidden: (B, S, D) — for single-token generation, S=1
    gate_weight: (n_routed_experts, hidden_size) in bfloat16 (router is not FP8)

    Returns:
        indices: (B, S, num_experts_per_tok) expert indices
        weights: (B, S, num_experts_per_tok) normalized routing weights
    """
    # (B, S, n_routed_experts)
    logits = hidden.float() @ gate_weight.float().T
    scores = torch.sigmoid(logits)

    # Group-limited selection: divide experts into groups, pick topk_group groups first,
    # then pick experts within those groups
    n_experts = config.n_routed_experts
    n_group = config.n_group
    topk_group = config.topk_group
    top_k = config.num_experts_per_tok
    group_size = n_experts // n_group

    # (B, S, n_group, group_size)
    scores_grouped = scores.view(*scores.shape[:-1], n_group, group_size)

    # Max score per group -> select top groups
    group_scores = scores_grouped.max(dim=-1).values  # (B, S, n_group)
    top_group_indices = group_scores.topk(topk_group, dim=-1).indices  # (B, S, topk_group)

    # Mask out experts not in selected groups
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(-1, top_group_indices, True)
    # (B, S, n_group, 1) -> broadcast to (B, S, n_group, group_size)
    expert_mask = group_mask.unsqueeze(-1).expand_as(scores_grouped)
    masked_scores = scores_grouped.where(expert_mask, torch.tensor(0.0, device=scores.device))
    masked_scores = masked_scores.view(*scores.shape[:-1], n_experts)

    # Top-k from masked scores
    topk_weights, topk_indices = masked_scores.topk(top_k, dim=-1)

    # Normalize
    if config.norm_topk_prob:
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-9)

    return topk_indices, topk_weights
