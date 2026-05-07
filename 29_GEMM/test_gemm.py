"""GEMM 正确性验证 + 计时"""

import time
import torch
from gemm_triton import solve


CASES = [
    (16, 16, 16, 1.0, 0.0),
    (32, 48, 64, 0.75, 0.25),
    (128, 128, 128, 1.0, 1.0),
    (1024, 1024, 1024, 1.0, 0.0),
]


def test():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"{'(M,N,K)':>16} | {'alpha':>6} | {'beta':>6} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 62)

    for M, N, K, alpha, beta in CASES:
        a = torch.empty(M, K, device="cuda", dtype=torch.float16).uniform_(-1.0, 1.0)
        b = torch.empty(K, N, device="cuda", dtype=torch.float16).uniform_(-1.0, 1.0)
        c = torch.empty(M, N, device="cuda", dtype=torch.float16).uniform_(-1.0, 1.0)
        c_initial = c.clone()

        # warmup
        solve(a, b, c, M, N, K, alpha, beta)
        torch.cuda.synchronize()

        c.copy_(c_initial)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(a, b, c, M, N, K, alpha, beta)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = (alpha * (a.float() @ b.float()) + beta * c_initial.float()).half()
        ok = torch.allclose(c, expected, atol=1e-1, rtol=1e-2)

        print(f"({M:>4},{N:>4},{K:>4}) | {alpha:>6.2f} | {beta:>6.2f} | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
