# RBF Hardware Pen Digits：中文使用与维护指南

本项目用于完成 Pen Digits 手写数字的采集、模拟硬件响应、模型训练和在线识别。

这份文档是项目的主要交接文档。第一次拿到仓库的开发者或 Agent，请先完整阅读“快速开始”“数据处理流程”“维护约束”三部分，再修改代码。

## 1. 项目当前实现

项目只有以下 4 个正式 Python 入口：

| 入口 | 职责 |
| --- | --- |
| `ui.py` | 一键启动统一科研 GUI，在同一窗口通过实验一保存、实验二手动推理并展示结果 |
| `collect_pen_digits.py` | 单独启动手写数据采集器，只向原始 CSV 追加 16 维数据 |
| `run_hardware_inference.py` | 无界面批处理入口，持续读取新行并完成完整推理，适用于自动化任务 |
| `train_hardware_model.py` | 使用真实硬件训练集重新训练模型并生成训练报告 |

`src/rbf_hardware/` 中的其他 Python 文件都是可复用模块，不应再增加 `main()`。

当前推理使用真实器件的 400 次循环响应作为经验响应库：

- 不使用旧 `gaussian_fitting_parameters.csv` 生成模拟响应；
- 每个手写数字随机选择一个真实物理 Cycle；
- 该数字的全部 16 个差分维度共用同一个 Cycle，保留通道相关性和 Cycle 漂移；
- 在实测响应上增加随信号幅值变化的随机噪声；
- 小信号最多约 ±5%，大信号约 ±1%；
- 每保存一个数字，原子覆盖固定硬件实验表，不与前一次实验累加。

注意：模型内部仍有“联合高斯特征变换”。它是分类器的特征处理方法，与旧的“使用高斯参数模拟器件响应”不是一回事。

## 2. 推荐目录布局

配置默认假定仓库位于某个工作区目录下：

```text
工作区/
├── rbf-hardware/                    # Git 仓库
│   ├── checkpoints/
│   ├── dataset/
│   ├── src/
│   ├── tests/
│   ├── ui.py
│   ├── collect_pen_digits.py
│   ├── run_hardware_inference.py
│   ├── train_hardware_model.py
│   └── config.yaml
├── runtime/                         # 自动生成，不进入 Git
├── output/                          # 训练输出，不进入 Git
├── app.log                          # 四个程序共用的运行日志
├── agent.log                        # Agent 工作记录
└── venv/                            # 推荐的 Python 虚拟环境
```

`config.yaml` 中的 `paths.workspace_root: ..` 表示所有相对路径都以 `rbf-hardware` 的父目录为基准。因此建议从工作区目录运行命令。

仓库可以直接上传 GitHub。运行数据、日志和训练输出位于仓库外，不会污染版本历史。

## 3. 环境要求

- Windows 10 或 Windows 11；
- Python 3.10 及以上版本，推荐 Python 3.12；
- Tkinter，用于手写数字绘图界面，一般随 Windows Python 一起安装；
- CPU 可以完成采集和推理，不强制要求 GPU；
- 依赖见 `requirements.txt`。

从工作区目录创建环境：

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\rbf-hardware\requirements.txt
```

检查 Tkinter：

```powershell
python -m tkinter
```

如果弹出一个测试窗口，说明绘图环境可用。

## 4. 五分钟快速开始

先进入包含 `rbf-hardware` 的工作区目录并激活虚拟环境：

```powershell
cd C:\path\to\workspace
.\venv\Scripts\Activate.ps1
```

启动统一科研 GUI：

```powershell
python .\rbf-hardware\ui.py
```

不需要再单独启动 `run_hardware_inference.py`。统一 GUI 已经在同一进程中加载采集与推理流水线，但两项实验由两个按钮分别触发。不要同时运行 GUI 和无界面持续推理入口，避免两个进程竞争同一组 CSV。

在手写板中写一个数字后，按以下顺序操作：

1. 点击 `Experiment 1 · Save` 或按 `Enter`，只在 `runtime/pen_digits_raw.csv` 追加一行；
2. 确认样本已保存后，点击 `Experiment 2 · Infer` 或按 `F5`；
3. 实验二生成 16 维差分等级；
4. 实验二生成 256 维硬件响应；
5. 用当前处理样本覆盖固定实验表；
6. 使用 `checkpoints/weights.pt` 识别数字；
7. 将预测结果和处理报告追加到对应 CSV；
8. 在右侧展示数字、最高得分、256 维硬件输出和最近记录；
9. 将两个实验的运行状态分别追加到 `app.log`。

保存动作不会自动触发推理。GUI 侦测到外部新增行时只显示“实验二已就绪”，仍需用户点击实验二按钮。没有待处理数据时实验二按钮保持禁用；任一流水线阶段落后于上游时按钮自动启用。实验二运行期间关闭窗口时，GUI 会等待各阶段 CSV 安全写完再退出，避免中途终止后台线程。关闭 GUI 窗口即可结束采集和推理。

实时查看日志：

```powershell
Get-Content .\app.log -Encoding UTF8 -Tail 100 -Wait
```

查看 Agent 工作记录：

```powershell
Get-Content .\agent.log -Encoding UTF8 -Tail 100 -Wait
```

## 5. 统一科研 GUI

统一 GUI 会根据当前屏幕方向选择初始尺寸。宽屏采用左右双栏；纵屏或窄窗口自动切换为上下布局，并在内容超过可用高度时提供整页垂直滚动。页头、流程状态卡和底部状态栏也会同步切换为窄屏排版。拖动窗口跨过布局断点时只重排现有控件，不会重建画板或丢失当前笔迹。

手写板带有网格和中心参考线，会随窗口可用空间等比放大或缩小，已绘制的轨迹、网格和采样点会同步重绘。内部始终使用固定逻辑坐标，因此调整窗口大小或横竖屏切换不会改变最终 16 维数据。程序从鼠标轨迹中等距选择 8 个点，将每个点的 x、y 坐标组成 16 个数，并缩放到 0～100。

常用操作：

- 鼠标左键拖动画线；
- 点击 `Experiment 1 · Save` 或按 `Enter`，只保存当前手写样本；
- 点击 `Experiment 2 · Infer` 或按 `F5`，手动运行待处理样本的硬件推理；
- 点击“清空”，清除当前画布；
- 按 `Ctrl+Z`，撤销上一段轨迹。

右侧科研面板展示：

- 当前识别数字；
- 样本序号；
- 模型最高得分；
- 采集、差分量化、16×16 硬件响应和模型识别四阶段状态；
- 本地样本数和经验响应模式；
- 最新一行 256 维硬件输出，使用科学计数法并支持横向滚动；
- 最近识别记录。

GUI 的可见标题、按钮、状态和弹窗统一使用英文。实验二在后台线程运行，模型计算时界面不会冻结。GUI 会侦测原始 CSV 的外部新增行并提示实验二可运行，但不会自动执行推理。

“Hardware Output”只替换原来的“Latest experiment table”界面展示。后台会在实验二处理每个样本时覆盖固定的 `hardware_experiments` CSV，不会改变硬件组的数据列结构。

默认输出：

```text
工作区/runtime/pen_digits_raw.csv
```

指定其他输出文件：

```powershell
python .\rbf-hardware\ui.py `
  --output C:\path\to\pen_digits_raw.csv
```

临时使用确定性的均值响应模式：

```powershell
python .\rbf-hardware\ui.py --sampling-mode mean
```

正常实验应使用默认的 `empirical` 模式。

### 5.1 分开启动采集和推理

只需要采集原始手写数据、不加载模型时：

```powershell
python .\rbf-hardware\collect_pen_digits.py
```

只需要推理已有或后续新增的数据时：

```powershell
python .\rbf-hardware\run_hardware_inference.py
```

也可以分别打开两个终端，先启动无界面推理，再启动独立采集器。独立采集器只追加原始 CSV，推理程序会自动侦测新行。

统一 `ui.py` 已经包含推理功能，因此运行 `ui.py` 时不要再对同一组 CSV 启动 `run_hardware_inference.py`。

## 6. 完整数据处理流程

```text
绘图保存
  ↓
16 维原始坐标，范围 0～100
  ↓ 按 train_in.csv / test_in.csv 的规则归一化
16 维 0～1 数值
  ↓ 就近量化到 differential_levels.csv 的 256 个等级
16 维差分等级
  ↓ 选择一个实测物理 Cycle，并查出每个等级的 16 组响应
16 × 16 = 256 维模拟硬件响应
  ↓
原子覆盖固定的 17 列硬件实验表
  ↓
checkpoints/weights.pt
  ↓
预测数字、分类分数、处理报告
```

整个流程在一个推理进程内完成，不需要启动多个中间处理程序。

### 6.1 追加式处理和断点恢复

所有运行时主表都是追加式 CSV。第 N 行在各阶段代表同一个样本：

```text
raw 第 N 行
↔ differential 第 N 行
↔ hardware 第 N 行
↔ predictions 第 N 行
↔ report 第 N 行
```

推理程序通过比较上下游行数，只处理尚未完成的新行。因此正常重启不会重复处理已完成样本。

不要只删除某一张下游表或只删除部分行，否则会破坏行号对应关系。需要清空实验时：

1. 先关闭绘图和推理程序；
2. 将整个 `runtime/` 归档，或成组清空全部运行时 CSV；
3. 再重新启动程序。

## 7. 实测硬件响应模拟

### 7.1 响应库

核心响应库：

```text
dataset/empirical/hardware_response_samples.npz
```

其逻辑形状为：

```text
256 个差分等级 × 400 个物理 Cycle × 16 个 Group
```

它来自硬件实测工作簿“400次循环归一化_新规则.xlsx”。仓库已经包含转换后的响应库，新用户运行推理时不需要原始 Excel 文件。

配套文件：

| 文件 | 用途 |
| --- | --- |
| `hardware_response_samples.npz` | 推理实际加载的完整实测响应 |
| `hardware_response_empirical_mapping.csv` | 256×16 个等级/通道的统计信息，供人工检查 |
| `hardware_response_metadata.json` | 原始文件哈希、维度和迁移信息 |
| `hardware_response_validation.json` | 模拟响应与真实硬件测试数据的验证指标 |
| `README.md` | 经验响应数据目录的补充说明 |

旧文件 `dataset/calibration/gaussian_fitting_parameters.csv` 仅保留作历史参考，当前推理不会加载它。

### 7.2 默认 empirical 模式

每处理一个新数字，程序进行以下操作：

1. 从 400 个真实 Cycle 中随机选一个；
2. 16 个差分维度共用该 Cycle；
3. 每个差分维度根据其等级，取出该 Cycle 的 16 个 Group 响应；
4. 按 `dimension_major` 顺序展开为 256 维；
5. 在每个实测值上加入独立的乘性均匀噪声。

256 维布局如下：

```text
维度 1 的 Group 1～16,
维度 2 的 Group 1～16,
...
维度 16 的 Group 1～16
```

这个顺序必须和真实硬件输出及 checkpoint 保持一致。

### 7.3 噪声规则

每个 Group 使用自身 400 次实测响应的绝对值第 95 百分位作为“大信号”参考：

```text
幅值比例 = clamp(|响应值| / Group绝对值q95, 0, 1)
噪声比例 = 5% - 4% × 幅值比例
输出值 = 实测响应 × uniform(1 - 噪声比例, 1 + 噪声比例)
```

因此：

- 接近零的小信号最多约 ±5%；
- 幅值越大，噪声比例越低；
- 达到或超过大信号参考值时约 ±1%；
- 噪声保留响应正负号；
- 生成值不会逐项原样复制实测表。

噪声上下限由 `config.yaml` 中以下字段控制：

```yaml
inference:
  empirical_noise_minimum_rate: 0.01
  empirical_noise_maximum_rate: 0.05
```

### 7.4 mean 验证模式

`mean` 模式使用每个等级 400 次 Cycle 的均值，不增加随机噪声，适合确定性回归检查：

```powershell
python .\rbf-hardware\run_hardware_inference.py `
  --once `
  --sampling-mode mean
```

`mean` 模式只建议用于验证，不应代替默认的 `empirical` 实验数据生成。

## 8. 固定覆盖的 17 列硬件实验表

每保存一个手写数字，程序立即覆盖以下固定表：

```text
runtime/hardware_experiments/pen_digits_hardware_experiment.csv
```

程序启动本身不会改写实验表。只有发现并处理一行新的手写数字数据时才覆盖。写入采用临时文件替换方式，不会向旧表尾部追加，也不会生成带时间戳的新文件。升级前已经存在的时间戳文件不会被程序自动删除，可按需要手工归档。

表结构：

| 列 | 内容 |
| --- | --- |
| `differential_level_index` | 差分等级在 `differential_levels.csv` 中的零基位置，严格为整数 0～255 |
| `hardware_value_01`～`hardware_value_16` | 该等级对应的 16 个硬件响应之和 |

聚合规则：

1. 当前数字有 16 个差分等级和 16 个对应的 16 值硬件块；
2. 如果当前数字内部有相同等级，将对应硬件块逐列相加；
3. 按 `differential_level_index` 从小到大排序；
4. 每个新数字都从空聚合状态开始，然后整体覆盖固定表，不累计前一个数字；
5. 一次补处理多行历史数据时按行依次覆盖，处理结束后固定表保留最后一行样本的聚合结果；
6. 第一列写成 `0`、`1`、…、`255`，不会写成 `0.0` 等浮点形式。

指定其他实验表目录：

```powershell
python .\rbf-hardware\run_hardware_inference.py `
  --experiment-output-dir C:\path\to\hardware_experiments
```

## 9. 运行时文件说明

默认路径都位于工作区的 `runtime/`：

| 文件或目录 | 列数 | 内容 |
| --- | ---: | --- |
| `pen_digits_raw.csv` | 16 | 绘图生成的原始坐标 |
| `pen_digits_differential.csv` | 16 | 量化后的差分值 |
| `pen_digits_hardware.csv` | 256 | 模拟硬件响应 |
| `hardware_experiments/pen_digits_hardware_experiment.csv` | 17 | 当前最新数字的固定覆盖聚合实验表 |
| `pen_digits_predictions.csv` | 由报告模块定义 | 样本序号和预测数字 |
| `pen_digits_inference_report.csv` | 由报告模块定义 | 时间、分数、分类边界及数据来源 |

`pen_digits_hardware.csv` 是 checkpoint 的直接输入。模型不会绕过硬件模拟层直接使用 16 维原始输入。

## 10. 推理命令

本节命令用于无界面批处理、服务器运行或自动化验证。日常手写实验优先使用统一 GUI。无界面持续推理程序与 GUI 不应同时处理同一组运行时文件。

持续监听新行：

```powershell
python .\rbf-hardware\run_hardware_inference.py
```

只处理当前已有的新行，然后退出：

```powershell
python .\rbf-hardware\run_hardware_inference.py --once
```

常用覆盖参数：

```text
--config
--sampling-mode empirical|mean
--raw-input
--differential-output
--hardware-output
--experiment-output-dir
--predictions-output
--report-output
--checkpoint
--response-bank
```

查看完整帮助：

```powershell
python .\rbf-hardware\run_hardware_inference.py --help
```

## 11. 训练模型

训练数据：

- `dataset/csv/train_out_400.csv`：真实硬件训练输出；
- `dataset/csv/test_out_400.csv`：真实硬件测试输出；
- `dataset/csv/train_in.csv`、`test_in.csv`：对应的 16 维参考输入和标签。

运行训练：

```powershell
python .\rbf-hardware\train_hardware_model.py
```

训练终端会显示两个 `tqdm` 进度条：

- `联合参数搜索`：按照交叉验证折数和每个真实参数组合持续更新；
- `最终模型训练`：展示联合特征变换、硬件分类器、PC 基准以及权重与报告保存 4 个阶段。

进度条显示当前折、联合高斯 `sigma` 和分类头 `alpha`。训练日志仍然同时追加到 `app.log`，进度条不会改变模型选择结果。

训练结果默认写入：

```text
工作区/output/rbf_hardware/<本次运行目录>/
```

其中包括权重、指标、预测、混淆矩阵、参数搜索结果和配置快照。确认新模型验证结果后，再有意识地替换：

```text
rbf-hardware/checkpoints/weights.pt
```

不要仅因为训练脚本运行成功就自动覆盖正式 checkpoint。

## 12. 配置文件

主配置为 `config.yaml`，主要分区：

| 分区 | 说明 |
| --- | --- |
| `paths` | 工作区、训练数据、输出和日志路径 |
| `data` | 数据列数、标签、编码和训练集约束 |
| `feature_transform` | 模型内部联合高斯特征变换 |
| `classifier` | Ridge 分类器及参数搜索 |
| `runtime` | 设备、数据类型和随机种子 |
| `inference` | 在线推理路径、经验响应、噪声和轮询间隔 |
| `output` | 训练产物命名 |
| `logging` | 日志级别和追加模式 |

修改配置时优先修改路径或参数，不要把本机绝对路径写进仓库。

## 13. 验证和测试

从工作区目录运行全部测试：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "$PWD\rbf-hardware\src"
python -m unittest discover -s .\rbf-hardware\tests -v
```

当前基准结果：

- 测试：15 项全部通过；
- Python 文件：36 个可以解析；
- 正式 `main()`：仅 4 个；
- checkpoint 在真实 `test_out_400.csv` 上准确率约 93.62%；
- 经验均值模拟准确率约 93.85%，与真实硬件预测一致率约 99.32%；
- 加噪经验模拟的 10 个随机种子平均准确率约 93.53%；
- 加噪经验模拟与真实硬件的平均预测一致率约 98.38%。

这些指标用于工程回归，不应被描述成全新盲测结论；当前测试集在开发过程中已经被检查过。

## 14. 常见问题

### 14.1 `PermissionError: [Errno 13] Permission denied`

原因通常是 WPS、Excel 或 CSV 预览器在 Windows 上独占了运行时 CSV。

处理方法：

1. 关闭正在查看的 CSV；
2. 保持 GUI 或无界面持续推理程序运行，它会记录警告并自动重试；
3. 如果使用 `--once`，关闭占用程序后重新执行命令。

不要在 GUI 或持续推理运行时直接用 WPS 或 Excel 打开：

```text
pen_digits_raw.csv
pen_digits_differential.csv
pen_digits_hardware.csv
pen_digits_predictions.csv
pen_digits_inference_report.csv
```

需要观察内容时，优先复制文件后查看，或使用：

```powershell
Get-Content .\runtime\pen_digits_hardware.csv -Tail 2
```

### 14.2 保存数字后没有生成实验表

检查：

- 推理程序是否正在运行；
- `app.log` 是否报告文件占用或列数错误；
- 原始表是否确实增加了新行；
- 下游表行数是否因手工删除而与上游失去对应。

实验表在“处理新行”时覆盖，不是在程序启动时改写。

### 14.3 输出重复或阶段行数不一致

停止所有程序，将整个 `runtime/` 成组归档或清空后重试。不要只删除一张中间表。

### 14.4 找不到模块

确认已经安装 `requirements.txt`，并从工作区目录使用仓库顶层入口运行。正式入口会自动把 `src/` 加入 Python 模块路径。

### 14.5 找不到响应库或 checkpoint

确认以下文件存在：

```text
rbf-hardware/dataset/empirical/hardware_response_samples.npz
rbf-hardware/checkpoints/weights.pt
```

并检查 `config.yaml` 中的 `workspace_root` 是否仍与目录布局匹配。

### 14.6 中文日志或 README 乱码

本项目面向 Windows 用户的中文 README 和日志使用 UTF-8 with BOM。读取时显式指定 UTF-8：

```powershell
Get-Content .\rbf-hardware\README_cn.md -Encoding UTF8
Get-Content .\app.log -Encoding UTF8 -Tail 100 -Wait
Get-Content .\agent.log -Encoding UTF8 -Tail 100 -Wait
```

文件前三个字节应为 UTF-8 BOM：

```powershell
Format-Hex .\rbf-hardware\README_cn.md | Select-Object -First 1
```

输出开头应包含：

```text
EF BB BF
```

## 15. 日志约定

### 15.1 `app.log`

四个正式程序共用 `工作区/app.log`，记录：

- 程序启动和停止；
- 共享文件路径；
- 各阶段新增行数；
- 响应模式、Cycle、噪声和预测结果；
- 文件占用、格式错误和重试信息。

日志必须追加写入，不应在程序启动时覆盖。

### 15.2 `agent.log`

Agent 在 `工作区/agent.log` 中用中文记录：

- 当前开始做什么；
- 修改了什么；
- 检查或测试的结果；
- 当前问题；
- 下一步要做什么。

示例：

```text
10:01 Developer:
开始检查推理流水线

10:03 Developer:
完成经验响应模块修改

10:04 Tester:
开始运行单元测试

10:06 Tester:
15 项测试全部通过
```

任务较长时应提高记录频率，不要在同一条“处理中”记录上停留太久。日志记录结论和工程动作，不记录隐藏的逐字推理过程。

## 16. 给接手 Agent 的维护约束

接手任务时按以下顺序检查：

1. 阅读本文件；
2. 阅读 `config.yaml`；
3. 执行 `Get-Content .\agent.log -Encoding UTF8 -Tail 100`；
4. 执行 `Get-Content .\app.log -Encoding UTF8 -Tail 100`；
5. 查看 `git status --short`，保留用户已有修改；
6. 明确修改的是采集、推理、训练还是共享模块；
7. 修改后运行相关测试和完整测试。

必须保持以下不变量：

- 正式 `main()` 始终只有统一 GUI、采集、推理、训练 4 个；
- 可复用代码放在 `src/rbf_hardware/` 的对应子包中；
- 单一模块应职责明确，入口文件只做参数解析和组装；
- 运行时文件继续位于仓库外的 `runtime/`；
- CSV 各阶段第 N 行必须代表同一样本；
- 256 维硬件响应必须保持 `dimension_major` 顺序；
- 默认响应模拟加载实测经验库，不得悄悄恢复旧高斯映射；
- 一个保存动作必须原子覆盖固定实验表，不能追加旧实验或生成时间戳文件；
- 实验表第一列必须是 0～255 的整数；
- 相同差分等级只在当前数字内部合并；
- 中文用户文件和日志使用 UTF-8 with BOM；
- `app.log` 和 `agent.log` 只能追加，不能覆盖；
- 不得提交 `runtime/`、`output/`、日志、缓存或虚拟环境；
- 修改实测响应源时，响应库、映射表、元数据和验证报告必须成组更新；
- 不得在没有验证的情况下覆盖正式 checkpoint。

建议提交前执行：

```powershell
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONPATH = "$PWD\rbf-hardware\src"
python -m unittest discover -s .\rbf-hardware\tests -v
rg -n 'if __name__ == .__main__.' .\rbf-hardware --glob "*.py"
git -C .\rbf-hardware status --short
git -C .\rbf-hardware diff --check
```

测试文件自身可能包含 `unittest.main()`，统计“正式入口”时应排除 `tests/`。

## 17. 代码目录职责

```text
src/rbf_hardware/
├── configuration/      # 配置读取、校验和路径解析
├── data/               # CSV 存储和训练数据集
├── inference/          # 预处理、经验响应、实验聚合、流式流水线
├── infrastructure/     # 日志等基础设施
├── modeling/           # 联合高斯变换、Ridge 分类器、checkpoint 推理
├── reporting/          # 指标和报告
├── training/           # 训练、交叉验证和参数选择
├── ui/                 # 统一GUI、独立采集应用、手写板和窗口工具
└── utilities/          # 无 main 的一次性迁移和格式转换工具
```

新增功能时，将业务逻辑放入最相关的子包，入口文件只负责：

1. 解析命令行参数；
2. 加载配置；
3. 解析路径；
4. 组装服务；
5. 启动流程并报告结果。

这样可以保证高内聚、低耦合，也便于单元测试。

## 18. 数据和编码约定

- 运行时数值表使用逗号分隔；
- 配置中的默认文本编码为 `utf-8-sig`；
- 含中文且供 Windows 用户直接查看的 Markdown、CSV 和日志优先使用 UTF-8 with BOM；
- JSON 和 Python 源文件使用标准 UTF-8，并由程序显式指定编码；
- 不依赖系统默认编码；
- CSV 列数和顺序属于数据契约，修改时必须同步更新验证和测试；
- 差分等级共有 256 个，不是 265 个；
- `differential_level_index` 是 0～255 的零基整数索引，不是原始浮点差分值。

## 19. 当前状态摘要

当前版本已经实现：

- 采集、手动触发推理和结果展示一体化的科研风格 GUI；
- 实验一保存、实验二推理的两个独立按钮；
- 后台推理线程、外部新行提示和最近识别记录；
- 16 维绘图数据追加采集；
- 与训练参考输入一致的归一化和 256 等级差分量化；
- 基于 400 Cycle 实测响应的 16×16 硬件模拟；
- 小信号约 ±5%、大信号约 ±1% 的幅值自适应随机噪声；
- 固定覆盖、17 列、等级合并并排序的最新硬件实验表；
- checkpoint 手动触发推理、预测表和详细报告；
- Windows 文件占用重试；
- 仓库外运行目录；
- 中文 UTF-8 日志；
- 统一 GUI、采集、推理、训练四个正式入口；
- 自动化测试和工程回归检查。

如需理解某一阶段的实现，请先从 `src/rbf_hardware/inference/pipeline.py` 查看流水线编排，再进入对应模块，不要从运行时 CSV 的偶然内容反推接口规则。
