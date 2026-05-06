import math
import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    a_block_ptr = tl.make_block_ptr(
        base=a,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c,
        shape=(n_elements,),
        strides=(1,),
        offsets=(pid * BLOCK_SIZE,),
        block_shape=(BLOCK_SIZE,),
        order=(0,),
    )

    a_vals = tl.load(a_block_ptr, boundary_check=(0,))
    b_vals = tl.load(b_block_ptr, boundary_check=(0,))
    tl.store(c_block_ptr, a_vals + b_vals, boundary_check=(0,))



# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, N: int):
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](a, b, c, N, BLOCK_SIZE)
