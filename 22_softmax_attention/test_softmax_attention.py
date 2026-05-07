"""softmax_attention 正确性验证 + 计时"""

import time
import torch
from softmax_attention_triton import solve


SIZES = [(1, 2, 2), (2, 3, 4), (16, 32, 32), (128, 128, 64), (512, 256, 128)]


def test():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print(f"{'(M,N,d)':>18} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 44)

    for M, N, d in SIZES:
        Q = torch.empty(M, d, device="cuda", dtype=torch.float32).uniform_(-0.5, 0.5)
        K = torch.empty(N, d, device="cuda", dtype=torch.float32).uniform_(-0.5, 0.5)
        V = torch.empty(N, d, device="cuda", dtype=torch.float32).uniform_(-0.5, 0.5)
        output = torch.empty(M, d, device="cuda", dtype=torch.float32)

        # warmup
        solve(Q, K, V, output, M, N, d)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(Q, K, V, output, M, N, d)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        scale = d**-0.5
        expected = torch.softmax((Q @ K.T) * scale, dim=1) @ V
        ok = torch.allclose(output, expected, atol=1e-3, rtol=1e-3)

        print(f"({M:>4},{N:>4},{d:>4}) | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
