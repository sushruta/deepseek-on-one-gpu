import torch
import triton
import triton.language as tl

QUANT_BLOCK = 128


@triton.jit
def fp8_block_matmul_kernel(
    x_ptr, w_ptr, scale_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_scale_n, stride_scale_k,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused FP8 block-dequantize + matmul kernel.

    Computes out = x @ w^T where w is FP8 with per-block scales.
    Each (BLOCK_N, BLOCK_K) weight tile shares one scale value,
    since BLOCK_N=BLOCK_K=QUANT_BLOCK=128.

    x: (M, K) bfloat16
    w: (N, K) float8_e4m3fn
    scale: (ceil(N/128), ceil(K/128)) float32
    out: (M, N) bfloat16
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in range(tl.cdiv(K, BLOCK_K)):
        offs_k = k_step * BLOCK_K + tl.arange(0, BLOCK_K)

        # Load x tile: (BLOCK_M, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)  # (BLOCK_M, BLOCK_K) bf16

        # Load w tile: (BLOCK_N, BLOCK_K)
        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # (BLOCK_N, BLOCK_K) fp8

        # Load scale: one scalar per (n_block, k_block) since BLOCK_N=BLOCK_K=128
        scale = tl.load(scale_ptr + pid_n * stride_scale_n + k_step * stride_scale_k)

        # Dequantize and accumulate: x @ w^T
        # w_f32: (BLOCK_N, BLOCK_K) -> transpose to (BLOCK_K, BLOCK_N) for dot
        w_f32 = w_tile.to(tl.float32) * scale
        acc += tl.dot(x_tile.to(tl.float32), tl.trans(w_f32))

    # Store output: (BLOCK_M, BLOCK_N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.bfloat16), mask=out_mask)


def fp8_linear_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Drop-in replacement for fp8_linear using a fused Triton kernel.

    x: (*, K) bfloat16
    weight: (N, K) float8_e4m3fn
    scale_inv: (ceil(N/128), ceil(K/128)) float32

    Returns: (*, N) in out_dtype
    """
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])  # (M, K)
    M, K = x_2d.shape
    N = weight.shape[0]

    out = torch.empty((M, N), dtype=torch.bfloat16, device=x.device)

    BLOCK_K = QUANT_BLOCK
    BLOCK_N = QUANT_BLOCK
    if M <= 16:
        BLOCK_M = 16
    elif M <= 128:
        BLOCK_M = 64
    else:
        BLOCK_M = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    fp8_block_matmul_kernel[grid](
        x_2d, weight, scale_inv, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        scale_inv.stride(0), scale_inv.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    result = out.to(out_dtype) if out_dtype != torch.bfloat16 else out
    return result.reshape(*original_shape[:-1], N)
