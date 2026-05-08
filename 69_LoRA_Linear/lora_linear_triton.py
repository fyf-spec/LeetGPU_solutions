import torch
import triton
import triton.language as tl


@triton.jit
def _lora_down_kernel(
    x,
    A,
    down,
    batch: tl.constexpr,
    d_in: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_r = tl.program_id(1)

    acc = tl.zeros((BLOCK_M, BLOCK_R), dtype=tl.float32)

    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(batch, d_in),
        strides=(d_in, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    a_ptr = tl.make_block_ptr(
        base=A,
        shape=(rank, d_in),
        strides=(d_in, 1),
        offsets=(pid_r * BLOCK_R, 0),
        block_shape=(BLOCK_R, BLOCK_K),
        order=(1, 0),
    )
    down_ptr = tl.make_block_ptr(
        base=down,
        shape=(batch, rank),
        strides=(rank, 1),
        offsets=(pid_m * BLOCK_M, pid_r * BLOCK_R),
        block_shape=(BLOCK_M, BLOCK_R),
        order=(1, 0),
    )

    for _ in range(0, d_in, BLOCK_K):
        x_tile = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        a_tile = tl.load(a_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(x_tile, tl.trans(a_tile), input_precision="tf32x3")

        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        a_ptr = tl.advance(a_ptr, (0, BLOCK_K))

    tl.store(down_ptr, acc, boundary_check=(0, 1))


@triton.jit
def _lora_linear_kernel(
    x,
    W,
    down,
    B,
    output,
    lora_scale,
    batch: tl.constexpr,
    d_in: tl.constexpr,
    d_out: tl.constexpr,
    rank: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(batch, BLOCK_M)
    num_pid_n = tl.cdiv(d_out, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    x_ptr = tl.make_block_ptr(
        base=x,
        shape=(batch, d_in),
        strides=(d_in, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K),
        order=(1, 0),
    )
    w_ptr = tl.make_block_ptr(
        base=W,
        shape=(d_out, d_in),
        strides=(d_in, 1),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_K),
        order=(1, 0),
    )
    down_ptr = tl.make_block_ptr(
        base=down,
        shape=(batch, rank),
        strides=(rank, 1),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_R),
        order=(1, 0),
    )
    b_ptr = tl.make_block_ptr(
        base=B,
        shape=(d_out, rank),
        strides=(rank, 1),
        offsets=(pid_n * BLOCK_N, 0),
        block_shape=(BLOCK_N, BLOCK_R),
        order=(1, 0),
    )
    output_ptr = tl.make_block_ptr(
        base=output,
        shape=(batch, d_out),
        strides=(d_out, 1),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    for _ in range(0, d_in, BLOCK_K):
        x_tile = tl.load(x_ptr, boundary_check=(0, 1), padding_option="zero")
        w_tile = tl.load(w_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += tl.dot(x_tile, tl.trans(w_tile), input_precision="tf32x3")

        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        w_ptr = tl.advance(w_ptr, (0, BLOCK_K))

    for _ in range(0, rank, BLOCK_R):
        down_tile = tl.load(down_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += lora_scale * tl.dot(down_tile, tl.trans(b_tile), input_precision="tf32x3")

        down_ptr = tl.advance(down_ptr, (0, BLOCK_R))
        b_ptr = tl.advance(b_ptr, (0, BLOCK_R))

    tl.store(output_ptr, acc, boundary_check=(0, 1))


# x, W, A, B, output are tensors on the GPU
def solve(
    x: torch.Tensor,
    W: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    output: torch.Tensor,
    batch: int,
    d_in: int,
    d_out: int,
    rank: int,
    lora_scale: float,
):
    down = torch.empty((batch, rank), device=x.device, dtype=torch.float32)

    block_r = 64
    _lora_down_kernel[(triton.cdiv(batch, 32), triton.cdiv(rank, block_r))](
        x,
        A,
        down,
        batch,
        d_in,
        rank,
        BLOCK_M=32,
        BLOCK_R=block_r,
        BLOCK_K=64,
        num_warps=4,
        num_stages=4,
    )

    grid = (triton.cdiv(batch, 32) * triton.cdiv(d_out, 64),)
    _lora_linear_kernel[grid](
        x,
        W,
        down,
        B,
        output,
        lora_scale,
        batch,
        d_in,
        d_out,
        rank,
        BLOCK_M=32,
        BLOCK_N=64,
        BLOCK_K=64,
        BLOCK_R=block_r,
        GROUP_M=8,
        num_warps=4,
        num_stages=4,
    )
