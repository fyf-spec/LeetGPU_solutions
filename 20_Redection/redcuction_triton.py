import torch
import triton
import triton.language as tl


@triton.jit
def _block_reduce_sum(input, partial, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    input_block = tl.make_block_ptr(
        base=input,
        shape=(N,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    partial_block = tl.make_block_ptr(
        base=partial,
        shape=(tl.cdiv(N, BLOCK_SIZE),),
        strides=(1,),
        offsets=(pid,),
        block_shape=(1,),
        order=(0,),
    )

    values = tl.load(input_block, boundary_check=(0,), padding_option="zero")
    block_sum = tl.sum(values, axis=0)
    tl.store(partial_block, block_sum + tl.zeros((1,), dtype=tl.float32), boundary_check=(0,))


@triton.jit
def _final_reduce_sum(partial, output, N, BLOCK_SIZE: tl.constexpr):
    partial_block = tl.make_block_ptr(
        base=partial,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    output_block = tl.make_block_ptr(
        base=output,
        shape=(1,),
        strides=(1,),
        offsets=(0,),
        block_shape=(1,),
        order=(0,),
    )

    values = tl.load(partial_block, boundary_check=(0,), padding_option="zero")
    total = tl.sum(values, axis=0)
    tl.store(output_block, total + tl.zeros((1,), dtype=tl.float32), boundary_check=(0,))


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    block_size = 2048
    final_block_size = 4096

    num_blocks = triton.cdiv(N, block_size)
    partial = input.new_empty((num_blocks,))

    _block_reduce_sum[(num_blocks,)](
        input,
        partial,
        N,
        BLOCK_SIZE=block_size,
        num_warps=8,
    )

    if num_blocks <= final_block_size:
        _final_reduce_sum[(1,)](
            partial,
            output,
            num_blocks,
            BLOCK_SIZE=final_block_size,
            num_warps=8,
        )
    else:
        next_blocks = triton.cdiv(num_blocks, final_block_size)
        partial2 = input.new_empty((next_blocks,))

        _block_reduce_sum[(next_blocks,)](
            partial,
            partial2,
            num_blocks,
            BLOCK_SIZE=final_block_size,
            num_warps=8,
        )
        _final_reduce_sum[(1,)](
            partial2,
            output,
            next_blocks,
            BLOCK_SIZE=final_block_size,
            num_warps=8,
        )
    # torch.sum(input,dim=0,keepdim=True,out=output)
