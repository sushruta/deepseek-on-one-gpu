import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DeepSeekConfig:
    hidden_size: int = 7168
    num_layers: int = 61
    num_attention_heads: int = 128
    num_key_value_heads: int = 128
    vocab_size: int = 129280

    # MLA dimensions
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    qk_head_dim: int = 192  # qk_nope_head_dim + qk_rope_head_dim

    # RoPE
    rope_theta: float = 10000.0
    max_position_embeddings: int = 163840
    rope_scaling_factor: float = 40.0
    rope_scaling_beta_fast: int = 32
    rope_scaling_beta_slow: int = 1
    rope_scaling_mscale: float = 1.0
    rope_scaling_mscale_all_dim: float = 1.0
    original_max_position_embeddings: int = 4096

    # MoE
    n_routed_experts: int = 256
    num_experts_per_tok: int = 8
    n_group: int = 8
    topk_group: int = 4
    moe_intermediate_size: int = 2048
    routed_scaling_factor: float = 2.5
    norm_topk_prob: bool = True

    # Dense MLP (layers 0-2)
    intermediate_size: int = 18432
    first_k_dense_replace: int = 3

    # Shared expert
    n_shared_experts: int = 1

    # Norms
    rms_norm_eps: float = 1e-6

    # FP8 block quantization
    fp8_block_size: int = 128


def load_config(model_path: str | Path) -> DeepSeekConfig:
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        raw = json.load(f)

    rope_scaling = raw.get("rope_scaling", {})

    return DeepSeekConfig(
        hidden_size=raw.get("hidden_size", 7168),
        num_layers=raw.get("num_hidden_layers", 61),
        num_attention_heads=raw.get("num_attention_heads", 128),
        num_key_value_heads=raw.get("num_key_value_heads", 128),
        vocab_size=raw.get("vocab_size", 129280),
        q_lora_rank=raw.get("q_lora_rank", 1536),
        kv_lora_rank=raw.get("kv_lora_rank", 512),
        qk_nope_head_dim=raw.get("qk_nope_head_dim", 128),
        qk_rope_head_dim=raw.get("qk_rope_head_dim", 64),
        v_head_dim=raw.get("v_head_dim", 128),
        qk_head_dim=raw.get("qk_nope_head_dim", 128) + raw.get("qk_rope_head_dim", 64),
        rope_theta=raw.get("rope_theta", 10000.0),
        max_position_embeddings=raw.get("max_position_embeddings", 163840),
        rope_scaling_factor=rope_scaling.get("factor", 40.0),
        rope_scaling_beta_fast=rope_scaling.get("beta_fast", 32),
        rope_scaling_beta_slow=rope_scaling.get("beta_slow", 1),
        rope_scaling_mscale=rope_scaling.get("mscale", 1.0),
        rope_scaling_mscale_all_dim=rope_scaling.get("mscale_all_dim", 1.0),
        original_max_position_embeddings=raw.get("original_max_position_embeddings", 4096),
        n_routed_experts=raw.get("n_routed_experts", 256),
        num_experts_per_tok=raw.get("num_experts_per_tok", 8),
        n_group=raw.get("n_group", 8),
        topk_group=raw.get("topk_group", 4),
        moe_intermediate_size=raw.get("moe_intermediate_size", 2048),
        routed_scaling_factor=raw.get("routed_scaling_factor", 2.5),
        norm_topk_prob=raw.get("norm_topk_prob", True),
        intermediate_size=raw.get("intermediate_size", 18432),
        first_k_dense_replace=raw.get("first_k_dense_replace", 3),
        n_shared_experts=raw.get("n_shared_experts", 1),
        rms_norm_eps=raw.get("rms_norm_eps", 1e-6),
        fp8_block_size=raw.get("quantization_config", {}).get("weight_block_size", [128, 128])[0],
    )
