import torch

from .config import DeepSeekConfig
from .weight_store import WeightStore


class GPUTransfer:
    """Manages GPU-side buffers and host->device transfers for expert weights.

    Non-expert weights live permanently on GPU (loaded by WeightStore).
    This class handles the dynamic part: packing and transferring the 8 router-selected
    experts per layer from pinned host RAM to a reusable GPU buffer.
    """

    def __init__(self, config: DeepSeekConfig, weight_store: WeightStore, device: torch.device,
                 verbose: bool = False):
        self.config = config
        self.weight_store = weight_store
        self.device = device
        self.verbose = verbose
        self.transfer_stream = torch.cuda.Stream(device=device)

        # Pre-allocate GPU buffer for 8 experts.
        # Each expert has 3 projections (gate, up, down).
        # gate/up: (moe_intermediate_size, hidden_size) = (2048, 7168) FP8 = ~14.3 MB each
        # down: (hidden_size, moe_intermediate_size) = (7168, 2048) FP8 = ~14.3 MB each
        # Total per expert: ~42.9 MB weight + small scale tensors
        # 8 experts: ~343 MB
        self._expert_gpu_buffers = self._allocate_expert_buffers()

    def _allocate_expert_buffers(self) -> list[dict[str, dict[str, torch.Tensor]]]:
        """Pre-allocate GPU tensors for 8 experts to avoid repeated allocation."""
        h = self.config.hidden_size
        ffn = self.config.moe_intermediate_size
        block = self.config.fp8_block_size
        import math

        buffers = []
        for _ in range(self.config.num_experts_per_tok):
            expert_buf = {}
            for proj, shape in [("gate_proj", (ffn, h)), ("up_proj", (ffn, h)), ("down_proj", (h, ffn))]:
                scale_shape = (math.ceil(shape[0] / block), math.ceil(shape[1] / block))
                expert_buf[proj] = {
                    "weight": torch.empty(shape, dtype=torch.float8_e4m3fn, device=self.device),
                    "weight_scale_inv": torch.empty(scale_shape, dtype=torch.float32, device=self.device),
                }
            buffers.append(expert_buf)
        return buffers

    def transfer_experts(
        self, layer: int, expert_indices: torch.Tensor, verbose: bool = False,
    ) -> list[dict[str, tuple[torch.Tensor, torch.Tensor]]]:
        """Pack and transfer selected experts from pinned host RAM to GPU buffers.

        expert_indices: (num_experts_per_tok,) 1D tensor of expert indices for one token.

        Returns list of dicts, one per expert:
            {proj_name: (weight_gpu, scale_gpu)}
        """
        indices = expert_indices.tolist()
        results = []

        _verbose = verbose or self.verbose
        if _verbose:
            print(f"      [gpu_transfer] L{layer:02d}: transferring experts {indices} "
                  f"({len(indices)} slots) host->GPU via async stream")

        with torch.cuda.stream(self.transfer_stream):
            for slot, eidx in enumerate(indices):
                host_expert = self.weight_store.get_expert(layer, eidx)
                gpu_buf = self._expert_gpu_buffers[slot]

                for proj in ("gate_proj", "up_proj", "down_proj"):
                    host_w, host_s = host_expert[proj]
                    gpu_buf[proj]["weight"].copy_(host_w, non_blocking=True)
                    gpu_buf[proj]["weight_scale_inv"].copy_(host_s, non_blocking=True)

        # Sync to ensure transfers complete before compute
        self.transfer_stream.synchronize()
        if _verbose:
            print(f"      [gpu_transfer] L{layer:02d}: transfer sync done")

        for slot in range(len(indices)):
            buf = self._expert_gpu_buffers[slot]
            results.append({
                proj: (buf[proj]["weight"], buf[proj]["weight_scale_inv"])
                for proj in ("gate_proj", "up_proj", "down_proj")
            })
        return results
