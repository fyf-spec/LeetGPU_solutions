import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 128, 'BLOCK_N': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_K': 64,  'BLOCK_N': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 128, 'BLOCK_N': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_K': 64,  'BLOCK_N': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K,
    stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # L2 Cache Swizzling：将 1D PID 映射为 2D 坐标 (pid_m, pid_k)
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_k = tl.cdiv(K, BLOCK_K)
    num_pid_in_group = GROUP_M * num_pid_k
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_k = (pid % num_pid_in_group) // group_size_m

    c_block_ptr = tl.make_block_ptr(
        base=c, shape=(M, K), strides=(stride_cm, stride_ck),
        offsets=(pid_m * BLOCK_M, pid_k * BLOCK_K), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    a_block_ptr = tl.make_block_ptr(
        base=a, shape=(M, N), strides=(stride_am, stride_an),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b, shape=(N, K), strides=(stride_bn, stride_bk),
        offsets=(0, pid_k * BLOCK_K), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0)
    )

    accumulator = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    # 循环累加
    for _ in range(tl.cdiv(N, BLOCK_N)):
        a_vals = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_vals = tl.load(b_block_ptr, boundary_check=(0, 1))
        
        accumulator += tl.dot(a_vals, b_vals, allow_tf32=False)
        
        a_block_ptr = tl.advance(a_block_ptr, offsets=(0, BLOCK_N))
        b_block_ptr = tl.advance(b_block_ptr, offsets=(BLOCK_N, 0))

    c_vals = accumulator.to(c.dtype.element_ty)
    tl.store(c_block_ptr, c_vals, boundary_check=(0, 1))


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = a.stride(0), a.stride(1)
    stride_bn, stride_bk = b.stride(0), b.stride(1)
    stride_cm, stride_ck = c.stride(0), c.stride(1)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(K, META['BLOCK_K']), )
    
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K,
        stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck
    )