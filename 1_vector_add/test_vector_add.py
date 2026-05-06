"""vector_add 正确性验证 + 计时"""

import time
import torch
import triton
from vector_add_triton import solve

N_VALUES = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

def test():
    print(f"{'N':>12} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 35)

    for N in N_VALUES:
        a = torch.randn(N, device="cuda", dtype=torch.float32)
        b = torch.randn(N, device="cuda", dtype=torch.float32)
        c = torch.empty(N, device="cuda", dtype=torch.float32)

        # warmup
        solve(a, b, c, N)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(a, b, c, N)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = a + b
        ok = torch.allclose(c, expected, atol=1e-5, rtol=1e-5)

        print(f"{N:>12,} | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()