# 400 Cycle 实测硬件响应库

本目录由硬件源文件 `400次循环归一化_新规则.xlsx` 生成。源工作簿保持只读，
其 SHA-256 记录在 `hardware_response_metadata.json`。

- `hardware_response_samples.npz`：运行时使用的完整经验样本，数组布局为
  `256 levels × 400 cycles × 16 groups`，数据类型为 `float32`。
- `hardware_response_empirical_mapping.csv`：便于人工审阅的统计映射表，
  每个等级和Group一行，包含均值、标准差、最值及1%～99%关键分位数。
- `hardware_response_metadata.json`：源文件哈希、维度和运行时采样规则。
- `hardware_response_validation.json`：与439条真实硬件测试数据的对照结果。

在线经验采样为每个保存数字随机选择一个物理Cycle。该数字的16个输入维度共享
此Cycle，再分别按量化等级查出完整16 Group响应。查表后增加独立乘性均匀噪声：
小信号最大±5%，达到各Group绝对响应95分位的大信号为±1%，中间线性递减。
这样不会逐值复制源数据，同时保留实测负值、非Gaussian尾部、Group相关性和
Cycle漂移。
