import torch
import triton
import triton.language as tl


@triton.jit
def _cross_entropy_rows_kernel(
    logits,
    true_labels,
    row_losses,
    N: tl.constexpr,
    C: tl.constexpr,
    stride_ln,
    stride_lc,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    row_start = pid_n * BLOCK_N
    row_offsets = row_start + tl.arange(0, BLOCK_N)
    col_offsets = tl.arange(0, BLOCK_C)
    row_mask = row_offsets < N

    logits_block = tl.make_block_ptr(
            base=logits,
            shape=(N, C),
            strides=(stride_ln, stride_lc),
            offsets=(row_start, 0),
            block_shape=(BLOCK_N, BLOCK_C),
            order=(1, 0),
        )

    m_i = tl.full((BLOCK_N,), -float("inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for col_start in range(0, C, BLOCK_C):
        values = tl.load(logits_block, boundary_check=(0, 1), padding_option="zero")
        values = tl.where(
            row_mask[:, None] & (col_start + col_offsets[None, :] < C),
            values,
            -float("inf"),
        )

        tile_max = tl.max(values, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(values - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        logits_block.advance((0, BLOCK_C))
        
    logsumexp = tl.log(l_i) + m_i

    labels = tl.load(true_labels + row_offsets, mask=row_mask, other=0)
    target_logit = tl.load(
        logits + row_offsets * stride_ln + labels * stride_lc,
        mask=row_mask,
        other=0.0,
    )
    row_loss = logsumexp - target_logit

    tl.store(row_losses + row_offsets, row_loss, mask=row_mask)


@triton.jit
def _mean_loss_kernel(row_losses, loss, N, BLOCK_N: tl.constexpr):
    losses_block = tl.make_block_ptr(
        base=row_losses,
        shape=(N,),
        strides=(1,),
        offsets=(0,),
        block_shape=(BLOCK_N,),
        order=(0,),
    )

    values = tl.load(losses_block, boundary_check=(0,), padding_option="zero")
    total = tl.sum(values, axis=0)
    tl.store(loss, total / N)


# logits, true_labels, loss are tensors on the GPU
def solve(logits: torch.Tensor, true_labels: torch.Tensor, loss: torch.Tensor, N: int, C: int):
    batch_block = 8
    block_c = 4096
    block_n = triton.next_power_of_2(N)
    reduce_num_warps = 1 if block_n <= 64 else 4 if block_n <= 2048 else 8
    row_losses = logits.new_empty((N,))

    _cross_entropy_rows_kernel[(triton.cdiv(N, batch_block),)](
        logits,
        true_labels,
        row_losses,
        N,
        C,
        logits.stride(0),
        logits.stride(1),
        BLOCK_N=batch_block,
        BLOCK_C=block_c,
        num_warps=8,
    )
    _mean_loss_kernel[(1,)](
        row_losses,
        loss,
        N,
        BLOCK_N=block_n,
        num_warps=reduce_num_warps,
    )
