# 硬件 RBF 后级全连接分类器

本项目用于训练器件产生的 256 维 RBF 特征之后的全连接分类头。网络结构和参数参考 `rbf-network-benchmark` 中的 `pen_digits`（OpenML 的规范名称）实验，但用器件测得的 RBF 输出替换 PC 计算的 Gaussian RBF 特征。

`train.py` 是唯一的训练入口；具体实现按职责拆分在 `src/rbf_hardware/` 中。

## 网络结构与训练方法

基准项目的计算链路为：

```text
16 维量化笔迹坐标
→ 256 维 Gaussian RBF 特征
→ 拼接 1 维常数偏置
→ 257 × 10 多输出岭回归权重
→ 对数字类别 0–9 做 argmax
```

分类头沿用基准项目的岭回归闭式目标：

```text
W = (Phi.T @ Phi + alpha * I)^(-1) @ Phi.T @ one_hot(y)
```

偏置不参与正则化。基准模型没有 optimizer、learning rate、batch size 或 epoch，不能为它虚构这些超参数。

PC 基准值 `ridge_alpha: 0.001` 仍保留为候选基线。器件特征的共线性远高于 PC Gaussian 中心，因此默认程序只在 `train_out` 内进行确定性的 5 折分层交叉验证，自动选择有效正则强度；当前数据选择 `alpha: 0.3`，选择过程不读取测试标签。

器件输出约为 `-5e-9` 到 `4.36e-6`，而理想 Gaussian 激活是非负、约为一数量级。默认配置会：

1. 将接近零的负器件噪声裁剪为零；
2. 只在 `train_out.csv` 上拟合一个全局最大值；
3. 将同一个训练集尺度应用到训练集和测试集；
4. 将尺度和裁剪规则写入 `weights.pt`。

这样可以避免测试数据泄漏。程序随后使用训练集折外预测为重点类别 8 选择一个小的分数偏置。该操作只修改现有全连接层的 bias，不会增加网络层；当前数据选择 `class_8_bias: 0.07`，并满足配置中的训练折召回率下限。

## 原始正确率较低的原因与解决结果

原始固定参数的训练准确率为 100%，测试准确率却只有 75.60%。归一化设计矩阵有 257 列，但训练样本只有 439 条，条件数约为 4,596，而且器件响应之间高度共线。使用 `alpha: 0.001` 时联合权重最大绝对值约为 9.24，表现为明显的高维小样本过拟合。

数字 8 的类内离散度也是所有类别中最大的；它最近的类别中心是数字 0，8 与 0 的中心距离只有数字 8 类内 RMS 离散度的约 46%。因此原分类头会少预测数字 8，并经常把 8 判成 0。

训练集内部的 alpha 搜索把最大权重降到约 0.665；训练折外的类别偏置校准进一步修正数字 8 被低估的问题。当前结果为：

| 运行方式 | 测试准确率 | Macro-F1 | 数字 8 召回率 |
|---|---:|---:|---:|
| 固定基准 alpha `0.001` | 75.60% | 74.99% | 54.48% |
| 交叉验证 alpha `0.3` | 84.73% | 84.30% | 68.82% |
| 交叉验证 alpha + 训练内数字 8 偏置（之前439/3000划分） | **84.90%** | **84.55%** | **75.27%** |
| 当前交换后的3000/439划分 | **84.28%** | **84.13%** | **73.81%** |

最后两行使用的测试样本不同，不能视为严格同一测试集上的直接对比。当前交换后的运行选择 `alpha: 0.3`、`class_8_bias: 0.1`，训练集内部五折准确率为86.86%。

## 数据约定

所有 CSV 都没有表头，最后一列是整数标签。

| 文件 | 尺寸 | 含义 | 用途 |
|---|---:|---|---|
| `dataset/csv/train_in.csv` | 10,553 × 17 | 16 维 RBF 前输入 + 标签 | 校验前 3,000 个训练标签 |
| `dataset/csv/train_out.csv` | 3,000 × 257 | 256 维器件 RBF 输出 + 标签 | 训练分类头 |
| `dataset/csv/test_in.csv` | 439 × 17 | 16 维 RBF 前输入 + 标签 | 校验测试标签 |
| `dataset/csv/test_out.csv` | 439 × 257 | 256 维器件 RBF 输出 + 标签 | 测试分类头 |

RBF 后的分类头必须使用 `train_out.csv` 和 `test_out.csv` 作为特征。`*_in.csv` 只有 16 个特征，不能直接送入需要 256 个输入的全连接层。

程序会用 `train_in.csv` 的前 3,000 行校验训练标签，并用 `test_in.csv` 的全部 439 行校验器件测试标签。`train_in.csv` 剩余 7,553 个参考输入目前没有对应的器件输出。

加载器还会拒绝训练集和测试集之间完全相同的特征行。当前隔离审计得到：共享特征行 0 条、共享带标签整行 0 条；四个数据文件的 SHA-256 指纹和审计结果都会保存到 `metrics.json`。

需要严格区分“没有混合训练”和“完全未观察的盲测”：当前 `test_out.csv` 已在迭代开发中被查看。84.90% 是这 3,000 行上可精确复算的真实结果，但不能再视为从未观察过的最终盲测估计。若用于论文或最终对外结论，应冻结当前 checkpoint，再采集新样本的器件输出，或补齐 `test_in` 剩余 7,553 行的器件输出，并只评估一次。

## 项目结构

```text
rbf-hardware/
├── config.yaml                  # 所有运行、训练和输出配置
├── train.py                     # 唯一含 main() 的训练入口
├── requirements.txt
├── README.md
├── README_cn.md
├── dataset/
│   ├── csv/
│   │   ├── train_in.csv
│   │   ├── train_out.csv
│   │   ├── test_in.csv
│   │   └── test_out.csv
│   └── origin_xslx/
└── src/rbf_hardware/
    ├── config.py                # 配置校验和路径解析
    ├── data.py                  # CSV、维度、标签校验和训练集尺度拟合
    ├── model.py                 # 与基准一致的岭回归分类头
    ├── metrics.py               # 指标、报告、CSV/SVG 混淆矩阵
    ├── logging_utils.py         # 终端与 app.log 共享日志
    └── training.py              # 训练和产物编排
```

## 在当前工作区运行

从 `SURF` 上一级项目目录使用已有虚拟环境：

```powershell
.\venv\Scripts\python.exe -u .\rbf-hardware\train.py --config .\rbf-hardware\config.yaml
```

或进入 `rbf-hardware` 后运行：

```powershell
..\venv\Scripts\python.exe -u train.py --config config.yaml
```

`-u` 表示不缓冲终端日志。

在第二个 VS Code PowerShell 终端实时查看同一个共享日志：

```powershell
Get-Content ..\app.log -Wait -Encoding UTF8
```

训练终端和 `SURF/app.log` 会收到完全相同的时间戳消息。日志中的 `[LOOK]` 与 `[CHANGE]` 标记展示数据检查、修改/训练以及训练后评估，形成可复现的“看 → 改 → 看”闭环。

## 部署到另一台 PC

将整个 `rbf-hardware` 文件夹（必须包含 `dataset/csv`）复制到一个可写的父目录：

```text
<workspace>/
└── rbf-hardware/
```

不要把已有虚拟环境复制到另一台 PC；应在目标机器重新创建虚拟环境。

### Windows PowerShell

建议使用 64 位 Python 3.10–3.12。

```powershell
cd <workspace>\rbf-hardware
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -u train.py --config config.yaml
```

上述命令直接调用虚拟环境中的 Python，因此不需要修改 PowerShell 脚本执行策略。

再打开一个 VS Code 终端：

```powershell
cd <workspace>\rbf-hardware
Get-Content ..\app.log -Wait -Encoding UTF8
```

### Linux 或 macOS

```bash
cd <workspace>/rbf-hardware
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -u train.py --config config.yaml
```

在第二个终端查看日志：

```bash
cd <workspace>/rbf-hardware
tail -F ../app.log
```

默认依赖会由 `pip` 选择可用的 PyTorch 版本。如果需要指定 CUDA 版本，应先按照目标机器驱动和 CUDA 版本安装 PyTorch 官方 wheel，再安装其余依赖。`config.yaml` 的 `runtime.device` 可设为 `auto`、`cpu` 或 `cuda`。

## 输出结果

所有运行结果都输出到 `rbf-hardware` 的上一级工作区：

```text
<workspace>/output/rbf_hardware/pen_digits_hardware_fc_<时间戳>/
├── weights.pt
├── metrics.json
├── predictions.csv
├── confusion_matrix.csv
├── confusion_matrix.txt
├── confusion_matrix.svg
├── classification_report.txt
├── alpha_search.csv
├── class_bias_search.csv
└── config.yaml
```

共享实时日志为 `<workspace>/app.log`。

`weights.pt` 中的张量全部保存在 CPU，因而没有 CUDA 的机器也可以加载。主要内容为：

- `state_dict.weight`：形状 `[10, 256]`；
- `state_dict.bias`：形状 `[10]`；
- `combined_weights`：与基准兼容的 `[257, 10]`；
- 类别、基准与最终岭回归参数、类别偏置、器件缩放参数、指标、数据来源和完整配置。

`metrics.json` 保存数据和运行环境信息、预处理参数、集合隔离哈希与审计、评估协议状态、完整的训练内 alpha/类别偏置搜索、数值诊断、逐类别召回率以及最终训练/测试指标。`alpha_search.csv` 和 `class_bias_search.csv` 是参数选择表的紧凑版本。

检查权重文件示例：

```python
import torch

checkpoint = torch.load("weights.pt", map_location="cpu", weights_only=False)
weight = checkpoint["state_dict"]["weight"]
bias = checkpoint["state_dict"]["bias"]
scale = checkpoint["preprocessing"]["scale"]
```

推理时必须先按照 checkpoint 中保存的 `negative_policy` 和 `scale` 处理器件特征，再使用 `weight` 和 `bias`。

## 配置说明

所有可调整项都集中在 `config.yaml`：

- `paths`：可移植的工作区相对路径、数据、输出和日志路径；
- `data`：CSV 分隔符、编码、特征数、类别和参考标签校验；
- `preprocessing`：负噪声处理和仅在训练集拟合的缩放；
- `classifier`：岭回归结构、基准 alpha、训练内 alpha 选择和类别分数校准；
- `runtime`：设备、数值类型、随机种子和确定性设置；
- `output`：运行目录和产物文件名；
- `logging`：日志级别和写入模式；
- `diagnostics`：泛化差距和类别召回率告警阈值。

相对路径根据项目和工作区结构解析，不依赖启动命令所在的当前目录。

## 常见问题

- **提示“17 columns … cannot be sent to the 256-input classifier”**：把 `test_in.csv` 或 `train_in.csv` 错当成了 RBF 后特征；请改用 `test_out.csv` 或 `train_out.csv`。
- **参考行数不同**：当前 `train_in` 的 10,553 行中仅前 3,000 行有器件输出，这是现有数据的正常情况。
- **指定 CUDA 但不可用**：将 `runtime.device` 改为 `cpu`，或安装兼容的显卡驱动和 CUDA PyTorch。
- **无法写入父目录**：输出和 `app.log` 按设计位于 `rbf-hardware` 上一级，因此父目录必须可写。
- **CSV 解析或列数错误**：保持现有 CSV 无表头，并保留最后一列标签。
