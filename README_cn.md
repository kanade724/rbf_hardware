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
→ 从400次实测响应库选择同一个物理Cycle
→ 按16个差分等级分别查出该Cycle的16组真实硬件响应
→ 追加到256维模拟硬件表
→ 将本次实验的“差分等级+对应16个硬件值”合并到独立17列表
→ checkpoints/weights.pt完成联合Gaussian变换和分类
→ 追加预测表与详细报告
```

推理程序为单进程三阶段流水线。每个阶段根据下游已有行数，仅处理上游新增行；程序重启后不会重复写入已经完成的行。所有运行状态、共享文件路径、处理行号和预测结果都会追加到 `SURF/app.log`。

## 数据约定

- `dataset/csv/train_in.csv` 和 `test_in.csv`：16 个 0～1 特征加标签，无表头；
- `dataset/csv/differential_levels.csv`：表头加 256 个严格递增等级，不是 265 个；
- `dataset/empirical/hardware_response_samples.npz`：256等级 × 400 Cycle × 16 Group的完整实测响应库；
- `dataset/empirical/hardware_response_empirical_mapping.csv`：256等级 × 16 Group的可读统计映射表；
- `dataset/empirical/hardware_response_metadata.json`：实测源文件哈希、维度和采样规则；
- `dataset/empirical/hardware_response_validation.json`：与439条真实硬件测试数据的对照指标；
- `dataset/calibration/gaussian_fitting_parameters.csv`：旧Gaussian拟合表，仅保留作历史参考，运行时不再加载；
- `checkpoints/weights.pt`：`format_version=2`，输入 256 维模拟或真实硬件响应。

经验统计映射表每一行对应一个等级和一个Group，列为：

```text
level_index,differential_level,group_index,sample_count,mean,std_dev,
minimum,q01,q05,q25,median,q75,q95,q99,maximum
```

`sampling_mode=empirical` 不拟合Gaussian，也不人工限制正负或宽度。每保存一个
数字，程序从400次物理Cycle中随机选择一个Cycle；该数字的16个差分维度都使用
同一个Cycle，再分别按等级查出完整16 Group响应。随后对每个响应增加独立乘性
均匀噪声。每个Group以自身实测绝对幅值95分位作为大信号参考，噪声范围按绝对
响应大小从小信号±5%线性下降到大信号±1%。这使输出不会逐值复制源数据，同时
保留响应符号、实测形状、偏态、长尾、通道相关性和Cycle漂移。`mean` 模式不加
随机噪声，使用400次Cycle逐等级均值，用于确定性验证。最终256维布局仍为
dimension-major，与真实硬件和checkpoint一致。

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

当前 checkpoint 在真实 `test_out_400.csv` 上准确率为 93.62%。使用400 Cycle
经验库均值映射439条数据时，准确率为93.85%，与真实硬件预测一致率99.32%，
硬件向量相关系数0.99983。使用10个随机种子进行整Cycle经验重采样时，平均
准确率93.53%（91.80%～94.08%），平均预测一致率98.38%，平均硬件相关系数
0.99938。

经验模拟以真实400次硬件响应为基底并加入1%～5%的幅值自适应随机扰动，不再
依赖旧Gaussian参数或人为±20%截断，也不会逐值照抄源表。最终checkpoint只消费
生成后的256维硬件表，不会绕过硬件层直接使用16维原始输入。连接在线真实硬件
时，只需让设备按相同dimension-major顺序写入 `pen_digits_hardware.csv`。
