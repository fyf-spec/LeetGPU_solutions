import torch
import triton
import triton.language as tl


@triton.jit
def swiglu(input, output, N, BLOCK_SIZE: tl.constexpr):
    half_n = N // 2
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < half_n

    x1 = tl.load(input + offsets, mask=mask, other=0.0)
    x2 = tl.load(input + half_n + offsets, mask=mask, other=0.0)

    sigmoid = 1.0 / (1.0 + tl.exp(-x1))
    result = x1 * sigmoid * x2
    tl.store(output + offsets, result, mask=mask)


# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N // 2, BLOCK_SIZE),)
    swiglu[grid](input, output, N, BLOCK_SIZE=BLOCK_SIZE, num_warps=4)
