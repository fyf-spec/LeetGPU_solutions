"""matrix_transpose 正确性验证 + 计时"""

import time
import torch
from matrix_transpose_triton import solve


SIZES = [(2, 3), (3, 1), (128, 256), (1024, 1024), (7000, 6000)]


def test():
    print(f"{'(rows,cols)':>16} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 40)

    for rows, cols in SIZES:
        input = torch.randn(rows, cols, device="cuda", dtype=torch.float32)
        output = torch.empty(cols, rows, device="cuda", dtype=torch.float32)

        # warmup
        solve(input, output, rows, cols)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(input, output, rows, cols)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = input.t()
        ok = torch.allclose(output, expected, atol=1e-5, rtol=1e-5)

        print(f"({rows:>4},{cols:>4}) | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
