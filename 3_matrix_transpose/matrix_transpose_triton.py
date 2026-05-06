import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(
    input,
    output,
    rows,
    cols,
    stride_ir,
    stride_ic,
    stride_or,
    stride_oc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(rows, BLOCK_M)
    num_pid_n = tl.cdiv(cols, BLOCK_N)

    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)

    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    input_block = tl.make_block_ptr(
        base=input,
        shape=(rows, cols),
        strides=(stride_ir, stride_ic),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output_block = tl.make_block_ptr(
        base=output,
        shape=(cols, rows),
        strides=(stride_or, stride_oc),
        offsets=(pid_n * BLOCK_N, pid_m * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )

    tile = tl.load(input_block, boundary_check=(0, 1))
    tl.store(output_block, tl.trans(tile), boundary_check=(0, 1))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = input.stride(0), input.stride(1)
    stride_or, stride_oc = output.stride(0), output.stride(1)
    block_m, block_n = 32, 32
    group_m = 8

    grid = (triton.cdiv(rows, block_m) * triton.cdiv(cols, block_n),)
    matrix_transpose_kernel[grid](
        input,
        output,
        rows,
        cols,
        stride_ir,
        stride_ic,
        stride_or,
        stride_oc,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GROUP_M=group_m,
        num_warps=8,
    )
