"""matrix_multiplication 正确性验证 + 计时"""

import time
import torch
from matrix_multiplication_triton import solve

SIZES = [(128, 128, 128), (512, 512, 512), (1024, 1024, 1024),(2048, 2048, 2048)]

def test():
    print(f"{'(M,N,K)':>16} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 40)

    for M, N, K in SIZES:
        a = torch.randn(M, N, device="cuda", dtype=torch.float16)
        b = torch.randn(N, K, device="cuda", dtype=torch.float16)
        c = torch.empty(M, K, device="cuda", dtype=torch.float16)

        solve(a, b, c, M, N, K)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(a, b, c, M, N, K)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = a @ b
        ok = torch.allclose(c, expected, atol=1e-2, rtol=1e-2)

        print(f"({M:>4},{N:>4},{K:>4}) | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")

if __name__ == "__main__":
    test()