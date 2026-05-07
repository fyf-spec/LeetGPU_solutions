"""reduction 正确性验证 + 计时"""

import time
import torch
from redcuction_triton import solve


N_VALUES = [1, 2, 8, 1_000, 100_000, 4_194_304, 4_194_304 * 4,  4_194_304 * 16]


def test():
    print(f"{'N':>12} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 35)

    for N in N_VALUES:
        input = torch.empty(N, device="cuda", dtype=torch.float32).uniform_(-1.0, 1.0)
        output = torch.empty(1, device="cuda", dtype=torch.float32)

        if N == 8:
            input = torch.arange(1, 9, device="cuda", dtype=torch.float32)

        # warmup
        solve(input, output, N)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(input, output, N)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = torch.sum(input)
        ok = torch.allclose(output[0], expected, atol=1e-2, rtol=1e-4)

        print(f"{N:>12,} | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
