import torch
import triton
import triton.language as tl


@triton.jit
def _local_prefix_sum_kernel(data, output, block_sums, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    data_block = tl.make_block_ptr(
        base=data,
        shape=(n,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    output_block = tl.make_block_ptr(
        base=output,
        shape=(n,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    values = tl.load(data_block, boundary_check=(0,), padding_option="zero")
    prefix = tl.cumsum(values, axis=0)

    tl.store(output_block, prefix, boundary_check=(0,))
    tl.store(block_sums + pid, tl.sum(values, axis=0))


@triton.jit
def _add_block_offsets_kernel(output, scanned_block_sums, block_sums, n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE

    output_block = tl.make_block_ptr(
        base=output,
        shape=(n,),
        strides=(1,),
        offsets=(block_start,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    values = tl.load(output_block, boundary_check=(0,), padding_option="zero")
    offset = tl.load(scanned_block_sums + pid) - tl.load(block_sums + pid)
    tl.store(output_block, values + offset, boundary_check=(0,))


def _prefix_sum(data: torch.Tensor, output: torch.Tensor, n: int):
    block_size = 1024
    num_blocks = triton.cdiv(n, block_size)
    block_sums = data.new_empty((num_blocks,))

    _local_prefix_sum_kernel[(num_blocks,)](
        data,
        output,
        block_sums,
        n,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    if num_blocks == 1:
        return

    scanned_block_sums = data.new_empty((num_blocks,))
    _prefix_sum(block_sums, scanned_block_sums, num_blocks)

    _add_block_offsets_kernel[(num_blocks,)](
        output,
        scanned_block_sums,
        block_sums,
        n,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )


# data and output are tensors on the GPU
def solve(data: torch.Tensor, output: torch.Tensor, n: int):
    _prefix_sum(data, output, n)
