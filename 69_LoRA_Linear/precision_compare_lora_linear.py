"""Compare tl.dot input_precision choices for the LoRA linear kernel.

Fixed case:
    batch=256, d_in=4096, d_out=4096, rank=64, lora_scale=0.5
"""

import statistics

import torch
import triton
import triton.language as tl


BATCH = 256
D_IN = 4096
D_OUT = 4096
RANK = 64
LORA_SCALE = 0.5

BLOCK_M = 32
BLOCK_N = 64
BLOCK_K = 64
BLOCK_R = 64

VARIANTS = [
    ("all_tf32", "tf32", "tf32", "tf32"),
    ("all_tf32x3", "tf32x3", "tf32x3", "tf32x3"),
    ("all_ieee", "ieee", "ieee", "ieee"),
    ("down_tf32x3_rest_tf32", "tf32x3", "tf32", "tf32"),
    ("main_tf32x3_rest_tf32", "tf32", "tf32x3", "tf32"),
    ("up_tf32x3_rest_tf32", "tf32", "tf32", "tf32x3"),
    ("down_ieee_rest_tf32", "ieee", "tf32", "tf32"),
    ("main_ieee_rest_tf32", "tf32", "ieee", "tf32"),
    ("up_ieee_rest_tf32", "tf32", "tf32", "ieee"),
    ("main_ieee_lora_tf32x3", "tf32x3", "ieee", "tf32x3"),
]


@triton.jit
def _lora_down_kernel(
    x,
    A,
    down,
    batch: tl.constexpr,
    d_in: tl.constexpr,
    rank: tl.constexpr,
    DOWN_PRECISION: tl.constexpr,
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
        acc += tl.dot(x_tile, tl.trans(a_tile), input_precision=DOWN_PRECISION)

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
    MAIN_PRECISION: tl.constexpr,
    UP_PRECISION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(d_out, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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
        acc += tl.dot(x_tile, tl.trans(w_tile), input_precision=MAIN_PRECISION)

        x_ptr = tl.advance(x_ptr, (0, BLOCK_K))
        w_ptr = tl.advance(w_ptr, (0, BLOCK_K))

    for _ in range(0, rank, BLOCK_R):
        down_tile = tl.load(down_ptr, boundary_check=(0, 1), padding_option="zero")
        b_tile = tl.load(b_ptr, boundary_check=(0, 1), padding_option="zero")
        acc += lora_scale * tl.dot(down_tile, tl.trans(b_tile), input_precision=UP_PRECISION)

        down_ptr = tl.advance(down_ptr, (0, BLOCK_R))
        b_ptr = tl.advance(b_ptr, (0, BLOCK_R))

    tl.store(output_ptr, acc, boundary_check=(0, 1))


def solve_with_precision(x, W, A, B, output, down_precision, main_precision, up_precision):
    down = torch.empty((BATCH, RANK), device=x.device, dtype=torch.float32)

    _lora_down_kernel[(triton.cdiv(BATCH, BLOCK_M), triton.cdiv(RANK, BLOCK_R))](
        x,
        A,
        down,
        BATCH,
        D_IN,
        RANK,
        down_precision,
        BLOCK_M=BLOCK_M,
        BLOCK_R=BLOCK_R,
        BLOCK_K=BLOCK_K,
        num_warps=4,
        num_stages=4,
    )

    _lora_linear_kernel[(triton.cdiv(BATCH, BLOCK_M) * triton.cdiv(D_OUT, BLOCK_N),)](
        x,
        W,
        down,
        B,
        output,
        LORA_SCALE,
        BATCH,
        D_IN,
        D_OUT,
        RANK,
        main_precision,
        up_precision,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        BLOCK_R=BLOCK_R,
        num_warps=4,
        num_stages=4,
    )


def reference(x, W, A, B):
    return x @ W.T + LORA_SCALE * ((x @ A.T) @ B.T)


def make_inputs():
    torch.manual_seed(0)
    x = torch.empty(BATCH, D_IN, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    W = torch.empty(D_OUT, D_IN, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    A = torch.empty(RANK, D_IN, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    B = torch.empty(D_OUT, RANK, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    return x, W, A, B


def benchmark_variant(x, W, A, B, expected, variant, warmup=3, repeat=10):
    _, down_precision, main_precision, up_precision = variant
    output = torch.empty(BATCH, D_OUT, device="cuda", dtype=torch.float32)

    for _ in range(warmup):
        solve_with_precision(x, W, A, B, output, down_precision, main_precision, up_precision)
    torch.cuda.synchronize()

    times = []
    for _ in range(repeat):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        solve_with_precision(x, W, A, B, output, down_precision, main_precision, up_precision)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    diff = (output - expected).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_l2 = (output - expected).norm().div(expected.norm()).item()
    ok = torch.allclose(output, expected, atol=1e-1, rtol=1e-2)
    ok_loose = torch.allclose(output, expected, atol=5e-1, rtol=1e-2)

    return statistics.median(times), max_err, mean_err, rel_l2, ok, ok_loose


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping precision comparison.")
        return

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    x, W, A, B = make_inputs()
    expected = reference(x, W, A, B)
    torch.cuda.synchronize()

    print(f"Fixed case: ({BATCH},{D_IN},{D_OUT},{RANK}) | lora_scale={LORA_SCALE}")
    print("Reference: PyTorch matmul with torch.backends.cuda.matmul.allow_tf32 = False")
    print(
        f"{'variant':>26} | {'down':>6} | {'main':>6} | {'up':>6} | "
        f"{'ok@1e-1':>8} | {'ok@5e-1':>8} | {'max err':>10} | "
        f"{'mean err':>10} | {'rel l2':>10} | {'median ms':>10}"
    )
    print("-" * 128)

    for variant in VARIANTS:
        name, down_precision, main_precision, up_precision = variant
        try:
            elapsed, max_err, mean_err, rel_l2, ok, ok_loose = benchmark_variant(x, W, A, B, expected, variant)
            print(
                f"{name:>26} | {down_precision:>6} | {main_precision:>6} | {up_precision:>6} | "
                f"{str(ok):>8} | {str(ok_loose):>8} | {max_err:>10.4e} | "
                f"{mean_err:>10.4e} | {rel_l2:>10.4e} | {elapsed:>10.4f}"
            )
        except Exception as exc:
            torch.cuda.synchronize()
            print(
                f"{name:>26} | {down_precision:>6} | {main_precision:>6} | {up_precision:>6} | "
                f"FAILED: {type(exc).__name__}: {exc}"
            )


if __name__ == "__main__":
    main()
