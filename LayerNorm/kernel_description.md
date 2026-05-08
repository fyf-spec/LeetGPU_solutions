# LayerNorm

来源教程: https://triton-lang.org/main/getting-started/tutorials/05-layer-norm.html

LayerNorm 通常沿着张量最后一维做归一化。假设最后一维长度是 `N`，每一行 `x` 会先用这一行自己的均值和方差归一化，然后再乘以可学习的 `weight`，加上可学习的 `bias`。

## 前向公式

对单行输入:

$$
\mu = \frac{1}{N}\sum_{i=0}^{N-1} x_i
$$

$$
\sigma^2 = \frac{1}{N}\sum_{i=0}^{N-1}(x_i - \mu)^2
$$

$$
rstd = \frac{1}{\sqrt{\sigma^2 + \epsilon}}
$$

$$
\hat{x_i} = (x_i - \mu) \cdot rstd
$$

$$
y_i = \hat{x_i} \cdot w_i + b_i
$$

其中 `eps` 是为了数值稳定性加在方差上的小常数，`w` 对应 PyTorch LayerNorm 的 `weight`，`b` 对应 `bias`。

直观理解:

- `mean` 把这一行的整体偏移移除。
- `variance` 和 `rstd` 把这一行缩放到稳定尺度。
- `weight` 和 `bias` 再把归一化后的值线性映射回模型需要的分布。

## 反向公式

设 `dy` 是上游传回来的梯度，先定义:

$$
wdy_i = dy_i \cdot w_i
$$

输入梯度可以写成:

$$
c_1 = \frac{1}{N}\sum_{i=0}^{N-1}\hat{x_i}\cdot wdy_i
$$

$$
c_2 = \frac{1}{N}\sum_{i=0}^{N-1}wdy_i
$$

$$
dx_i = rstd \cdot (wdy_i - \hat{x_i}\cdot c_1 - c_2)
$$

对可学习参数:

$$
dw_i = \sum_m dy_{m,i}\cdot \hat{x}_{m,i}
$$

$$
db_i = \sum_m dy_{m,i}
$$

`dw` 和 `db` 需要沿 batch 中所有行求和，因为每一行共享同一组 `weight` 和 `bias`。

## Triton 实现要点

- 前向 kernel 中，一个 Triton program 处理一行。
- 先沿最后一维做 reduction 得到 `mean`，再做第二次 reduction 得到 `variance`。
- 前向会把 `mean` 和 `rstd` 存下来，反向时直接复用。
- 第一个反向 kernel 按行计算 `dx`，同时把局部 `dw`/`db` 累积到分组 buffer 中。
- 因为多个 program 可能写同一个分组 buffer，所以官方教程使用 lock 做保护。
- 第二个反向 kernel 再把分组 buffer reduce 成最终 `dw` 和 `db`。
- 官方教程的实现限制单行 feature 数据小于 64KB，代码中通过 `MAX_FUSED_SIZE = 65536 // x.element_size()` 控制。
