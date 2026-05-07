"""int8_matmul 正确性验证 + 计时"""

import time
import torch
from int8_matmul_triton import solve


CASES = [
    (1, 1, 2, 1.0, 1.0, 1.0, 1, 3, 5),
    (2, 2, 2, 0.1, 0.2, 0.05, 0, 0, 0),
    (32, 48, 64, 0.02, 0.03, 0.04, -3, 5, 2),
    (128, 128, 128, 0.02, 0.02, 0.05, 0, 0, 0),
    (512, 512, 512, 0.02, 0.03, 0.04, -2, 4, -1),
    (1024, 1024, 1024, 0.02, 0.03, 0.04, -2, 4, -1)
]


def reference(a, b, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C):
    acc = (a.float() - zero_point_A) @ (b.float() - zero_point_B)
    scaled = acc.float() * (scale_A * scale_B / scale_C)
    rounded = torch.where(scaled >= 0, torch.floor(scaled + 0.5), torch.ceil(scaled - 0.5))
    quantized = rounded + zero_point_C
    return torch.clamp(quantized, -128, 127).to(torch.int8)


def make_case(M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C):
    a = torch.randint(-128, 128, (M, K), device="cuda", dtype=torch.int8)
    b = torch.randint(-128, 128, (K, N), device="cuda", dtype=torch.int8)

    if (M, N, K) == (1, 1, 2):
        a = torch.tensor([[1, 2]], device="cuda", dtype=torch.int8)
        b = torch.tensor([[3], [4]], device="cuda", dtype=torch.int8)
    elif (M, N, K) == (2, 2, 2):
        a = torch.tensor([[1, 2], [3, 4]], device="cuda", dtype=torch.int8)
        b = torch.tensor([[5, 6], [7, 8]], device="cuda", dtype=torch.int8)

    return a, b


def test():
    print(f"{'(M,N,K)':>16} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 40)

    for M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C in CASES:
        a, b = make_case(M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C)
        c = torch.empty(M, N, device="cuda", dtype=torch.int8)

        # warmup
        solve(a, b, c, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(a, b, c, M, N, K, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = reference(a, b, scale_A, scale_B, scale_C, zero_point_A, zero_point_B, zero_point_C)
        ok = torch.equal(c, expected)

        print(f"({M:>4},{N:>4},{K:>4}) | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
