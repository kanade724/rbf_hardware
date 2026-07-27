# 硬件 RBF Pen Digits 系统

本项目包含三个且仅有三个带 `main()` 的运行入口：

- `train_hardware_model.py`：训练 16×16 硬件 Gaussian 联合模型；
- `collect_pen_digits.py`：通过鼠标轨迹采集 16 维 Pen Digits 原始数据；
- `run_hardware_inference.py`：持续处理新增行、模拟硬件响应并完成识别。

其余 Python 文件均为高内聚的库模块，不包含命令行入口。

## 在线推理流程

```text
绘图采集产生16维0～100数据
→ 按train_in.csv/test_in.csv规则归一化到0～1
→ 映射到differential_levels.csv中的256个离散等级
→ 追加到16维差分中间表
→ 按16组Gaussian均值和标准差生成16×16硬件响应
→ 追加到256维模拟硬件表
→ 将本次实验的“差分等级+对应16个硬件值”合并到独立17列表
→ checkpoints/weights.pt完成联合Gaussian变换和分类
→ 追加预测表与详细报告
```

推理程序为单进程三阶段流水线。每个阶段根据下游已有行数，仅处理上游新增行；程序重启后不会重复写入已经完成的行。所有运行状态、共享文件路径、处理行号和预测结果都会追加到 `SURF/app.log`。

## 数据约定

- `dataset/csv/train_in.csv` 和 `test_in.csv`：16 个 0～1 特征加标签，无表头；
- `dataset/csv/differential_levels.csv`：表头加 256 个严格递增等级，不是 265 个；
- `dataset/calibration/gaussian_fitting_parameters.csv`：16 个 Group × 256 个等级，共 4096 行；
- `dataset/calibration/gaussian_fitting_parameters_group_1.csv`：用户上传的单 Group 原表备份；
- `checkpoints/weights.pt`：`format_version=2`，输入 256 维模拟或真实硬件响应。

规范校准表列名统一为：

```text
group_index,differential_level,amplitude,mean,std_dev
```

每个输入维度按 dimension-major 顺序生成 16 个 Group 响应。`sampling_mode=gaussian` 时根据说明文件使用 `normal(mean, std_dev)` 采样；`mean` 模式用于确定性验证。

## 目录结构

```text
rbf-hardware/
├── train_hardware_model.py
├── collect_pen_digits.py
├── run_hardware_inference.py
├── config.yaml
├── checkpoints/
├── dataset/
├── tests/
└── src/rbf_hardware/
    ├── configuration/     # 配置校验与路径解析
    ├── data/              # 训练数据与追加式CSV存储
    ├── inference/         # 量化、硬件模拟、流式调度
    ├── infrastructure/    # 共享日志
    ├── modeling/          # 联合Gaussian、岭分类器、checkpoint推理
    ├── reporting/         # 指标与报告
    ├── training/          # 训练、交叉验证与参数选择
    ├── ui/                # Pen Digits绘图采集器
    └── utilities/         # 无main函数的数据迁移工具
```

## 运行

在 `SURF` 目录执行：

```powershell
.\venv\Scripts\Activate.ps1

# 1. 启动绘图采集器
python .\rbf-hardware\collect_pen_digits.py

# 2. 另开终端启动持续推理
python .\rbf-hardware\run_hardware_inference.py

# 3. 需要时重新训练
python .\rbf-hardware\train_hardware_model.py
```

只处理当前已有行并退出：

```powershell
python .\rbf-hardware\run_hardware_inference.py --once
```

确定性模拟验证：

```powershell
python .\rbf-hardware\run_hardware_inference.py --once --sampling-mode mean
```

实时查看程序日志：

```powershell
Get-Content .\app.log -Tail 100 -Wait
```

不要在推理运行时使用 WPS 或 Excel 直接打开运行时 CSV；这类软件可能在 Windows
上独占文件。持续推理模式检测到占用后会在 `app.log` 中记录中文告警并自动重试，
关闭对应表格窗口后会从未完成阶段继续，不会重复处理已经写入的上游行。

查看代理实施记录：

```powershell
Get-Content .\agent.log -Tail 100 -Wait
```

## 运行时共享文件

默认保存在仓库外部的 `SURF/runtime/`，因此 `rbf-hardware` 可以直接作为独立仓库上传 GitHub：

- `pen_digits_raw.csv`：绘图产生的 16 维原始行；
- `pen_digits_differential.csv`：16 维 0～1 差分等级行；
- `pen_digits_hardware.csv`：256 维模拟硬件行；
- `hardware_experiments/`：每保存一个手写数字时新建一张独立实验聚合表；
- `pen_digits_predictions.csv`：样本序号和预测数字；
- `pen_digits_inference_report.csv`：时间、得分、分类边界和完整来源路径。

这些文件构成同一条追加式数据链，不能单独删除中间文件中的部分行。需要重置时，应在停止推理程序后成组归档或清空全部运行时 CSV。

每张实验表命名为
`pen_digits_hardware_experiment_YYYYMMDD_HHMMSS_ffffff.csv`，共有 17
列：第一列为整型 `differential_level_index`，后 16 列为
`hardware_value_01`～`hardware_value_16`。每保存一个数字，就使用该数字唯一的
一行差分数据和一行 256 维硬件数据立即创建一张新表。差分行第 N 个值与硬件行
第 N 个连续 16 值块对应；仅在这一个数字的 16 个差分值内部，将相同差分等级的
硬件块逐列求和并合并为一行，最终按差分等级从小到大排序。不同数字的数据绝不
写入或累计到同一张实验表。程序一次补处理多条历史新增行时，也会为每条样本
分别创建一张表。

`differential_level_index` 按该值在 `differential_levels.csv` 中的位置进行
零基映射，严格输出整数 `0`～`255`：第一项为 `0`，最后一项为 `255`。CSV 中
不会写成浮点数 `0.0` 或 `255.0`，可直接交给硬件组使用。

需要修改实验表目录时可使用：

```powershell
python .\rbf-hardware\run_hardware_inference.py `
  --experiment-output-dir C:\path\to\experiment_tables
```

## 验证结果

当前 checkpoint 在真实 `test_out_400.csv` 上准确率为 93.62%。使用 16 组 Gaussian 参数均值模拟全部 439 行时，准确率为 94.31%；按校准标准差随机采样时为 93.85%。两种模拟与真实硬件预测的一致率均为 99.32%，硬件向量相关系数分别为 0.99982 和 0.99964。

因此，该流程可以体现拟合校准分布下的硬件统计响应和噪声结果，且最终 checkpoint 只消费生成后的 256 维硬件表，不会绕过硬件层直接使用 16 维原始输入。但它仍是基于 `mean/std_dev` 的模拟硬件结果，不能替代某一次真实器件采集值；连接真实硬件时，只需让设备按相同 dimension-major 顺序写入 `pen_digits_hardware.csv`。
