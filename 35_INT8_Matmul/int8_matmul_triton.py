import torch
import triton
import triton.language as tl


# A_int8[M, K] @ B_int8[K, N] -> C_int8[M, N]
# A_real = scale_A * (A_int8 - zero_point_A)
# B_real = scale_B * (B_int8 - zero_point_B)
# C_real = A_real @ B_real
# C_int8 = round(C_real / scale_C) + zero_point_C
@triton.jit
def _int8_matmul_kernel(
    a,
    b,
    c,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    scale_A,
    scale_B,
    scale_C,
    zero_point_A,
    zero_point_B,
    zero_point_C,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # num_pid_in_group = GROUP_M * num_pid_n
    # group_id = pid // num_pid_in_group
    # first_pid_m = group_id * GROUP_M
    # group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    # pid_in_group = pid - group_id * num_pid_in_group
    # pid_m = first_pid_m + (pid_in_group % group_size_m)
    # pid_n = pid_in_group // group_size_m
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    a_block_ptr = tl.make_block_ptr(
        base=a,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b,
        shape=(K, N),
        strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    for k_start in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1), padding_option="zero")

        raw_dot = tl.dot(a_tile, b_tile, out_dtype=tl.int32)
        sum_a = tl.sum(a_tile.to(tl.int32), axis=1)
        sum_b = tl.sum(b_tile.to(tl.int32), axis=0)
        valid_k = tl.minimum(BLOCK_K, K - k_start)

        # (A - zA)(B - zB) = A B - zB * sum(A) - zA * sum(B) + zA * zB * K
        acc += (
            raw_dot
            - zero_point_B * sum_a[:, None]
            - zero_point_A * sum_b[None, :]
            + zero_point_A * zero_point_B * valid_k
        )

        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    scaled = acc.to(tl.float32) * (scale_A * scale_B / scale_C)
    rounded = tl.where(scaled >= 0.0, tl.floor(scaled + 0.5), tl.ceil(scaled - 0.5))
    quantized = rounded + zero_point_C
    clipped = tl.minimum(tl.maximum(quantized, -128.0), 127.0)

    tl.store(c_block_ptr, clipped.to(tl.int8), boundary_check=(0, 1))


# a, b, c are tensors on the GPU
def solve(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    M: int,
    N: int,
    K: int,
    scale_A: float,
    scale_B: float,
    scale_C: float,
    zero_point_A: int,
    zero_point_B: int,
    zero_point_C: int,
):
    block_m, block_n, block_k = 32, 64, 64
    group_m = 8

    grid = (triton.cdiv(M, block_m) * triton.cdiv(N, block_n),)
    _int8_matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        scale_A,
        scale_B,
        scale_C,
        zero_point_A,
        zero_point_B,
        zero_point_C,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        GROUP_M=group_m,
        num_warps=4,
        num_stages=2,
    )
