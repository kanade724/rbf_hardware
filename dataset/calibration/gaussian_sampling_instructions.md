# 高斯采样说明

> 历史说明：当前在线推理已切换到
> `dataset/empirical/hardware_response_samples.npz` 的400 Cycle实测经验重采样，
> 本文及Gaussian参数表不再被运行时加载，仅保留用于追溯旧方案。

## 数据来源

`高斯拟合参数.xlsx` — 16 个 Sheet（Group_1 ~ Group_16），每个 Sheet 有 256 行（Sheet_Index 0-255），每行记录该位置 100 个数据点的高斯拟合参数：

| 列 | 含义 |
|----|------|
| Sheet_Index | 原始数据 Sheet 编号 (0-255) |
| A (Amplitude) | 高斯拟合幅值 |
| μ (Mean) | 高斯拟合均值 |
| σ (Std Dev) | 高斯拟合标准差 |

## 采样方法

从一个特定 Group 和 Sheet 的高斯分布中随机生成一个值：

```python
import numpy as np
import openpyxl

# 加载参数
wb = openpyxl.load_workbook("高斯拟合参数.xlsx", data_only=True)
ws = wb["Group_1"]  # Group 编号: Group_1 ~ Group_16

# 读取指定 Sheet 的参数（row = sheet_index + 2）
sheet_index = 100  # 0-255
mu = float(ws.cell(row=sheet_index + 2, column=3).value)
sigma = float(ws.cell(row=sheet_index + 2, column=4).value)
wb.close()

# 采样
value = np.random.normal(mu, sigma)
print(value)
```

## 封装函数

```python
import openpyxl
import numpy as np

# ---- 预加载（启动时执行一次） ----
wb = openpyxl.load_workbook("高斯拟合参数.xlsx", data_only=True)
params = {}  # params[group_idx][sheet_idx] = {"mu": ..., "sigma": ...}

for g in range(16):
    ws = wb[f"Group_{g + 1}"]
    params[g] = {}
    for sheet_idx in range(256):
        r = sheet_idx + 2
        params[g][sheet_idx] = {
            "mu": float(ws.cell(row=r, column=3).value),
            "sigma": float(ws.cell(row=r, column=4).value),
        }
wb.close()


def sample(group, sheet):
    """从 Group(1-16), Sheet(0-255) 的高斯分布中随机采一个值"""
    p = params[group - 1][sheet]
    mu, sigma = p["mu"], p["sigma"]
    if sigma <= 0:
        return mu
    return float(np.random.normal(mu, sigma))


def sample_batch(group, sheet, n):
    """批量采样 n 个值"""
    p = params[group - 1][sheet]
    mu, sigma = p["mu"], p["sigma"]
    if sigma <= 0:
        return np.full(n, mu)
    return np.random.normal(mu, sigma, size=n)
```

## 使用示例

```python
# 单次采样
v = sample(group=1, sheet=100)
print(v)  # 例如: 4.16e-06

# 批量采样
vs = sample_batch(group=1, sheet=100, n=10)
print(vs)  # array([4.15e-06, 4.17e-06, ...])
```

## 原理

对每个 Group × 每个 Sheet 的 100 个数据点做直方图后，用 `scipy.optimize.curve_fit` 拟合高斯函数：

$$f(x) = A \cdot e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$

提取 μ 和 σ 后，调用 `np.random.normal(mu, sigma)` 即可生成符合该分布的随机值。

## 硬件推理中的集中采样约束

`A` 是拟合曲线在 `μ` 处的峰高，并不是 `0～1` 概率；将 Gaussian 归一化为
概率分布后，A 会被约掉。原始 `σ` 描述采集数据的拟合宽度。为了满足当前硬件组
要求，在线模拟额外采用以下集中约束：

```text
radius = abs(μ) × 20%
relative_A = A / max(A)  # 同一输入维度的16个Group内归一化
effective_sigma = min(σ, radius / (3 × (1 + relative_A)))
value ~ TruncatedNormal(μ, effective_sigma, μ-radius, μ+radius)
```

相对 A 越高，分布越集中于中心。超出边界的随机值会重新采样，最终再做边界
保护。因此每个 Group 的模拟值严格在其中心上下20%以内，同时避免简单裁剪导致
大量数据堆积在边界。
