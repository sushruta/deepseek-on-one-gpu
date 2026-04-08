import json
import re
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open

from .config import DeepSeekConfig

EXPERT_PROJ_NAMES = ("gate_proj", "up_proj", "down_proj")

EXPERT_PATTERN = re.compile(
    r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|weight_scale_inv)"
)


def is_expert_weight(key: str) -> bool:
    return EXPERT_PATTERN.match(key) is not None


def parse_expert_key(key: str) -> tuple[int, int, str, str] | None:
    """Returns (layer, expert_idx, proj_name, param_type) or None."""
    m = EXPERT_PATTERN.match(key)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), m.group(3), m.group(4)


def expert_tensor_keys(layer: int, expert_idx: int) -> list[str]:
    """All safetensor keys for one expert (weight + scale for each projection)."""
    prefix = f"model.layers.{layer}.mlp.experts.{expert_idx}"
    return [
        f"{prefix}.{proj}.{suffix}"
        for proj in EXPERT_PROJ_NAMES
        for suffix in ("weight", "weight_scale_inv")
    ]


class ExpertCache:
    """LRU cache of expert weights in pinned host RAM.

    Each entry is keyed by (layer, expert_idx) and holds
    {proj_name: {"weight": pinned_tensor, "weight_scale_inv": pinned_tensor}}.
    """

    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.used = 0
        self.entries: OrderedDict[tuple[int, int], dict] = OrderedDict()

    def get(self, layer: int, expert_idx: int) -> dict | None:
        key = (layer, expert_idx)
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key]

    def put(self, layer: int, expert_idx: int, expert_dict: dict):
        key = (layer, expert_idx)
        entry_bytes = sum(
            t.nbytes
            for proj in expert_dict.values()
            for t in proj.values()
        )
        self._evict_until(entry_bytes)
        self.entries[key] = expert_dict
        self.used += entry_bytes

    def _evict_until(self, needed: int):
        while self.used + needed > self.capacity and self.entries:
            _, evicted = self.entries.popitem(last=False)
            self.used -= sum(
                t.nbytes for proj in evicted.values() for t in proj.values()
            )


class WeightStore:
    """Loads DeepSeek-R1 weights with a disk-backed architecture.

    Non-expert weights (~40 GB): loaded to GPU at startup (streamed through RAM).
    Expert weights (~627 GB): read on demand from safetensors on disk, with an
    LRU cache in pinned host RAM to avoid re-reading hot experts.
    """

    def __init__(
        self,
        model_path: str | Path,
        config: DeepSeekConfig,
        device: torch.device,
        expert_cache_bytes: int = 16 * (1024 ** 3),
    ):
        self.model_path = Path(model_path)
        self.config = config
        self.device = device

        # Non-expert weights on GPU: key -> tensor
        self.gpu_weights: dict[str, torch.Tensor] = {}

        # Expert key -> shard filename (for lazy loading from disk)
        self.expert_key_to_shard: dict[str, str] = {}

        # LRU cache of experts in pinned host RAM
        self.expert_cache = ExpertCache(expert_cache_bytes)

        self._load()

    def _load(self):
        """Load non-expert weights to GPU; build index for expert weights."""
        index_path = self.model_path / "model.safetensors.index.json"
        with open(index_path) as f:
            index = json.load(f)

        # Separate expert keys (index only) from non-expert keys (load to GPU)
        non_expert_keys_by_shard: dict[str, list[str]] = {}
        for key, shard_file in index["weight_map"].items():
            if is_expert_weight(key):
                self.expert_key_to_shard[key] = shard_file
            else:
                non_expert_keys_by_shard.setdefault(shard_file, []).append(key)

        # Load non-expert weights to GPU, one shard at a time (streams through RAM)
        for shard_file, keys in non_expert_keys_by_shard.items():
            shard_path = self.model_path / shard_file
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in keys:
                    self.gpu_weights[key] = f.get_tensor(key).to(self.device)

    def _load_expert_from_disk(self, layer: int, expert_idx: int) -> dict:
        """Read one expert's tensors from safetensors shards into pinned host RAM.

        Returns {proj_name: {"weight": pinned, "weight_scale_inv": pinned}}.
        """
        result = {}
        # Group keys by shard to minimize file opens
        keys = expert_tensor_keys(layer, expert_idx)
        keys_by_shard: dict[str, list[str]] = {}
        for key in keys:
            shard = self.expert_key_to_shard[key]
            keys_by_shard.setdefault(shard, []).append(key)

        tensors: dict[str, torch.Tensor] = {}
        for shard_file, shard_keys in keys_by_shard.items():
            shard_path = self.model_path / shard_file
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for key in shard_keys:
                    tensor = f.get_tensor(key)
                    pinned = torch.empty_like(tensor).pin_memory()
                    pinned.copy_(tensor)
                    tensors[key] = pinned

        prefix = f"model.layers.{layer}.mlp.experts.{expert_idx}"
        for proj in EXPERT_PROJ_NAMES:
            result[proj] = {
                "weight": tensors[f"{prefix}.{proj}.weight"],
                "weight_scale_inv": tensors[f"{prefix}.{proj}.weight_scale_inv"],
            }
        return result

    def get(self, key: str) -> torch.Tensor:
        return self.gpu_weights[key]

    def get_expert(self, layer: int, expert_idx: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Returns {proj_name: (weight_pinned, scale_pinned)} for one expert.

        Checks pinned-RAM LRU cache first; on miss, reads from disk.
        """
        cached = self.expert_cache.get(layer, expert_idx)
        if cached is None:
            cached = self._load_expert_from_disk(layer, expert_idx)
            self.expert_cache.put(layer, expert_idx, cached)

        return {
            proj: (cached[proj]["weight"], cached[proj]["weight_scale_inv"])
            for proj in EXPERT_PROJ_NAMES
        }

    def get_layer_attn_weights(self, layer: int) -> dict[str, tuple[torch.Tensor, torch.Tensor] | torch.Tensor]:
        """Returns attention projection weights for a layer.

        FP8 projections return (weight, scale_inv) tuples.
        LayerNorm weights return plain tensors.
        """
        prefix = f"model.layers.{layer}.self_attn."
        proj_names = ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "kv_b_proj", "o_proj"]
        result = {}

        for name in proj_names:
            w_key = f"{prefix}{name}.weight"
            s_key = f"{prefix}{name}.weight_scale_inv"
            result[name] = (self.gpu_weights[w_key], self.gpu_weights[s_key])

        # LayerNorm weights (plain tensors, not FP8)
        result["q_a_layernorm"] = self.gpu_weights[f"{prefix}q_a_layernorm.weight"]
        result["kv_a_layernorm"] = self.gpu_weights[f"{prefix}kv_a_layernorm.weight"]
        return result

    def get_layer_norm_weights(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (input_layernorm, post_attention_layernorm) for a layer."""
        prefix = f"model.layers.{layer}"
        return (
            self.gpu_weights[f"{prefix}.input_layernorm.weight"],
            self.gpu_weights[f"{prefix}.post_attention_layernorm.weight"],
        )

    def get_dense_mlp_weights(self, layer: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Returns dense MLP weights (gate_proj, up_proj, down_proj) for layers 0-2."""
        prefix = f"model.layers.{layer}.mlp."
        return {
            proj: (self.gpu_weights[f"{prefix}{proj}.weight"], self.gpu_weights[f"{prefix}{proj}.weight_scale_inv"])
            for proj in EXPERT_PROJ_NAMES
        }

    def get_shared_expert_weights(self, layer: int) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """Returns shared expert weights for MoE layers."""
        prefix = f"model.layers.{layer}.mlp.shared_experts."
        return {
            proj: (self.gpu_weights[f"{prefix}{proj}.weight"], self.gpu_weights[f"{prefix}{proj}.weight_scale_inv"])
            for proj in EXPERT_PROJ_NAMES
        }

    def get_gate_weight(self, layer: int) -> torch.Tensor:
        """Returns router gate weight (not FP8, plain bfloat16)."""
        return self.gpu_weights[f"model.layers.{layer}.mlp.gate.weight"]
