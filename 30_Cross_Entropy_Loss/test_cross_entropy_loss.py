"""cross_entropy_loss 正确性验证 + 计时"""

import time
import torch
import torch.nn.functional as F
from cross_entropy_loss_triton import solve


CASES = [(2, 3), (3, 4), (128, 64), (1024, 256), (10_000, 1_0000), (16184, 16184)]


def make_case(N: int, C: int):
    logits = torch.empty(N, C, device="cuda", dtype=torch.float32).uniform_(-5.0, 5.0)
    true_labels = torch.randint(0, C, (N,), device="cuda", dtype=torch.long)

    if (N, C) == (2, 3):
        logits = torch.tensor(
            [[1.0, 2.0, 0.5], [0.1, 3.0, 1.5]],
            device="cuda",
            dtype=torch.float32,
        )
        true_labels = torch.tensor([1, 1], device="cuda", dtype=torch.long)
    elif (N, C) == (3, 4):
        logits = torch.tensor(
            [[-0.5, 1.5, 0.0, 1.0], [2.0, -1.0, 0.5, 0.5], [0.0, 0.0, 0.0, 0.0]],
            device="cuda",
            dtype=torch.float32,
        )
        true_labels = torch.tensor([3, 0, 1], device="cuda", dtype=torch.long)

    return logits, true_labels


def test():
    print(f"{'(N,C)':>16} | {'正确性':>6} | {'耗时 (ms)':>10}")
    print("-" * 40)

    for N, C in CASES:
        logits, true_labels = make_case(N, C)
        loss = torch.empty(1, device="cuda", dtype=torch.float32)

        # warmup
        solve(logits, true_labels, loss, N, C)
        torch.cuda.synchronize()

        # 计时
        start = time.perf_counter()
        solve(logits, true_labels, loss, N, C)
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - start) * 1000

        # 正确性
        expected = F.cross_entropy(logits, true_labels, reduction="mean")
        ok = torch.allclose(loss[0], expected, atol=1e-2, rtol=1e-2)

        print(f"({N:>5},{C:>5}) | {'✅' if ok else '❌':>6} | {elapsed:>10.4f}")


if __name__ == "__main__":
    test()
