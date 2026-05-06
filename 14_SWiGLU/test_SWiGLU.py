"""SWiGLU 正确性验证 + 计时"""

import time
import torch
from SWiGLU_triton import solve


N_VALUES = [2, 4, 1_000, 16_384, 100_000]


def test():
    print(f"{'N':>12} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 35)

    for N in N_VALUES:
        input = torch.empty(N, device="cuda", dtype=torch.float32).uniform_(-100.0, 100.0)
        output = torch.empty(N // 2, device="cuda", dtype=torch.float32)

        # warmup
        solve(input, output, N)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(input, output, N)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        half_n = N // 2
        x1 = input[:half_n]
        x2 = input[half_n:]
        expected = x1 * torch.sigmoid(x1) * x2
        ok = torch.allclose(output, expected, atol=1e-4, rtol=1e-4)

        print(f"{N:>12,} | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
