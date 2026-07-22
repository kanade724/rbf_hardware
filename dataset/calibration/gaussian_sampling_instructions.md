# 高斯采样说明

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
