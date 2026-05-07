import torch
import triton
import triton.language as tl


@triton.jit
def fwd_kernel_triton(
    q,
    k,
    v,
    o,
    M: tl.constexpr,
    N: tl.constexpr,
    d: tl.constexpr,
    stride_qm,
    stride_qd,
    stride_kn,
    stride_kd,
    stride_vn,
    stride_vd,
    stride_om,
    stride_od,
    scale,
    Q_TILE_SIZE: tl.constexpr,
    N_TILE_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    q_tile_index = tl.program_id(0)
    q_offset = q_tile_index * Q_TILE_SIZE

    q_block_ptr = tl.make_block_ptr(
        base=q,
        shape=(M, d),
        strides=(stride_qm, stride_qd),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, BLOCK_D),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        base=k,
        shape=(N, d),
        strides=(stride_kn, stride_kd),
        offsets=(0, 0),
        block_shape=(N_TILE_SIZE, BLOCK_D),
        order=(1, 0),
    )
    v_block_ptr = tl.make_block_ptr(
        base=v,
        shape=(N, d),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(N_TILE_SIZE, BLOCK_D),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        base=o,
        shape=(M, d),
        strides=(stride_om, stride_od),
        offsets=(q_offset, 0),
        block_shape=(Q_TILE_SIZE, BLOCK_D),
        order=(1, 0),
    )

    q_tile = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    m_i = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)
    acc = tl.zeros((Q_TILE_SIZE, BLOCK_D), dtype=tl.float32)

    for n_start in range(0, N, N_TILE_SIZE):
        k_tile = tl.load(k_block_ptr, boundary_check=(0, 1), padding_option="zero")
        v_tile = tl.load(v_block_ptr, boundary_check=(0, 1), padding_option="zero")

        s_tile = tl.dot(q_tile, tl.trans(k_tile), allow_tf32=False) * scale
        n_offsets = n_start + tl.arange(0, N_TILE_SIZE)
        s_tile = tl.where(n_offsets[None, :] < N, s_tile, -float("inf"))

        m_ij = tl.max(s_tile, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s_tile - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v_tile, allow_tf32=False)
        m_i = m_new

        k_block_ptr = tl.advance(k_block_ptr, (N_TILE_SIZE, 0))
        v_block_ptr = tl.advance(v_block_ptr, (N_TILE_SIZE, 0))

    output = acc / l_i[:, None]
    tl.store(o_block_ptr, output, boundary_check=(0, 1))


# Q, K, V, output are tensors on the GPU.
def solve(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, output: torch.Tensor, M: int, N: int, d: int
):
    q_tile_size = 16
    n_tile_size = 64
    block_d = max(32, 1 << (d - 1).bit_length())

    grid = (triton.cdiv(M, q_tile_size),)
    fwd_kernel_triton[grid](
        Q,
        K,
        V,
        output,
        M,
        N,
        d,
        Q.stride(0),
        Q.stride(1),
        K.stride(0),
        K.stride(1),
        V.stride(0),
        V.stride(1),
        output.stride(0),
        output.stride(1),
        d**-0.5,
        Q_TILE_SIZE=q_tile_size,
        N_TILE_SIZE=n_tile_size,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
