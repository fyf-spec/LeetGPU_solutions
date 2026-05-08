"""LoRA linear correctness check + timing."""

import time

import torch

from lora_linear_triton import solve


CASES = [
    (2, 4, 3, 2, 0.5),
    (7, 17, 13, 5, 0.75),
    (32, 128, 96, 16, 1.0),
    (128, 1024, 1024, 32, 0.25),
    (256, 4096, 4096, 64, 0.5),
]


def reference(x, W, A, B, lora_scale):
    return x @ W.T + lora_scale * ((x @ A.T) @ B.T)


def make_case(batch, d_in, d_out, rank):
    x = torch.empty(batch, d_in, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    W = torch.empty(d_out, d_in, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    A = torch.empty(rank, d_in, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
    B = torch.empty(d_out, rank, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)

    if (batch, d_in, d_out, rank) == (2, 4, 3, 2):
        x = torch.tensor(
            [[1.0, 0.0, -1.0, 2.0], [0.0, 1.0, 1.0, -1.0]],
            device="cuda",
            dtype=torch.float32,
        )
        W = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        A = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        B = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )

    return x, W, A, B


def run_case(batch, d_in, d_out, rank, lora_scale):
    x, W, A, B = make_case(batch, d_in, d_out, rank)
    output = torch.empty(batch, d_out, device="cuda", dtype=torch.float32)

    solve(x, W, A, B, output, batch, d_in, d_out, rank, lora_scale)
    torch.cuda.synchronize()

    start = time.perf_counter()
    solve(x, W, A, B, output, batch, d_in, d_out, rank, lora_scale)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000

    expected = reference(x, W, A, B, lora_scale)
    ok = torch.allclose(output, expected, atol=1e-1, rtol=1e-2)
    max_err = (output - expected).abs().max().item()

    return ok, max_err, elapsed


def test():
    if not torch.cuda.is_available():
        print("CUDA is not available; skipping LoRA linear Triton test.")
        return

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False

    print(
        f"{'(batch,d_in,d_out,rank)':>28} | {'scale':>7} | {'correct':>7} | "
        f"{'max err':>10} | {'time (ms)':>10}"
    )
    print("-" * 76)

    for batch, d_in, d_out, rank, lora_scale in CASES:
        ok, max_err, elapsed = run_case(batch, d_in, d_out, rank, lora_scale)
        shape = f"({batch},{d_in},{d_out},{rank})"
        print(f"{shape:>28} | {lora_scale:>7.2f} | {str(ok):>7} | {max_err:>10.4e} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
