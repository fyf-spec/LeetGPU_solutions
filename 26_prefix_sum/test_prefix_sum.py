"""prefix_sum 正确性验证 + 计时"""

import time
import torch
from prefix_sum_triton import solve


N_VALUES = [1, 2, 4, 5, 1024, 1025, 100_000, 250_000]


def test():
    print(f"{'N':>12} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 35)

    for n in N_VALUES:
        data = torch.empty(n, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        output = torch.empty(n, device="cuda", dtype=torch.float32)

        if n == 4:
            data = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda", dtype=torch.float32)
        elif n == 5:
            data = torch.tensor([5.0, -2.0, 3.0, 1.0, -4.0], device="cuda", dtype=torch.float32)

        # warmup
        solve(data, output, n)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(data, output, n)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = torch.cumsum(data, dim=0)
        ok = torch.allclose(output, expected, atol=1e-2, rtol=1e-4)

        print(f"{n:>12,} | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
