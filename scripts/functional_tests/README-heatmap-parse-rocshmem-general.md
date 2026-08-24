# rocSHMEM 测试结果解析使用文档
> 配套脚本 `heatmap-parse-rocshmem-general.py`，用于解析 rocSHMEM 功能测试输出的日志文件，生成包含延迟与带宽的 Excel 热力图报告。

## 1. 功能概述
脚本读取一个或多个目录下的 rocSHMEM 测试结果日志，按操作类型（op）、GPU 数、workgroup 数、线程数分组，将各消息大小对应的延迟（Lat）、带宽（Bw）、算法带宽（AlgBw）、总线带宽（BusBw）汇总到一张 Excel 工作表中，便于横向对比性能。

主要特性：
- 支持按 **消息大小（msgsize）** 或 **Rank 级消息总大小（volume）** 两种维度展示
- 支持 **仅延迟 / 仅带宽 / 同时显示延迟和带宽** 三种输出模式
- 自动检测日志中的 `error`、`failed`、`segfault` 关键字，失败用例标记为 `Failed` 并附关键错误片段
- 自动收集各目录下的 `env_info.log`，统一汇总到独立的 `EnvInfo` 工作表
- 失败用例与正常用例均按操作类型分组，便于对比

## 2. 环境依赖
```bash
pip3 install xlsxwriter
```

## 3. 使用方法
### 3.1 命令格式
```bash
python3 heatmap-parse-rocshmem-general.py [--lat|--bw|--both] [--s|--v] [-o=out_file] 目录1 [目录2 ...]
```

### 3.2 参数说明
| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--lat` | 只显示延迟（单位 us） | 否 |
| `--bw` | 只显示带宽（Bw/AlgBw/BusBw，单位 GB/s） | 否 |
| `--both` | 同时显示延迟和带宽 | 是 |
| `--s` | 按消息大小（msgsize）展示 | 是 |
| `--v` | 按 Rank 级消息总大小（volume = 并发消息数 × size）展示 | 否 |
| `-o=名称` | 指定输出 Excel 文件名（不含扩展名） | `rocshmem_test` |
| 目录 | 一个或多个包含测试日志的目录路径 | 必填 |

### 3.3 使用示例
```bash
# 默认模式：同时显示延迟和带宽，按消息大小展示
python3 heatmap-parse-rocshmem-general.py ./results/op1 ./results/op2

# 只看带宽，按 volume 展示，输出到 my_report.xlsx
python3 heatmap-parse-rocshmem-general.py --bw --v -o=my_report ./results

# 只看延迟
python3 heatmap-parse-rocshmem-general.py --lat ./results
```

## 4. 输入文件要求
### 4.1 日志文件命名规范
脚本通过正则 `^(.+?)_n(\d+)_w(\d+)_z(\d+)` 解析文件名，命名格式必须为：
```
<op>_n<ngpus>_w<nwgs>_z<nthreads>
```
- `op`：操作类型名称（如 `put`、`get`、`allreduce` 等）
- `n`：GPU 数量
- `w`：workgroup 数量
- `z`：线程数

不符合命名规范的文件将被跳过并打印 `⏭️ Skip` 提示。

### 4.2 日志文件内容格式
每行一条数据记录，以空白分隔。以 `#`、`[` 开头或包含 `mpirun` 的行会被跳过。支持以下三种列格式：

**5 列格式（无 volume）**
```
Msg Size (B)  # of timed Msgs  Latency (us)  Bandwidth (GB/s)  Msg Rate (Msg/s)
```

**6 列格式（含 volume）**
```
Volume (B)  Msg Size (B)  # of timed Msgs  Latency (us)  Bandwidth (GB/s)  Msg Rate (Msg/s)
```

**8 列格式（含 RCCL 对齐带宽）**
```
Volume (B)  Msg Size (B)  # of timed Msgs  Latency (us)  Bandwidth (GB/s)  AlgBwAlignRccl (GB/s)  BusBwAlignRccl (GB/s)  Msg Rate (Msg/s)
```

> 提示：`volume < 8` 的记录会被过滤（由脚本顶部 `minmsgsize` 控制，可按需修改）。

### 4.3 环境信息文件（可选）
在每个输入目录下放置 `env_info.log`，脚本会将其内容收集到 Excel 末尾的 `EnvInfo` 工作表，并在各目录工作表中放置跳转链接。

## 5. 输出结果
### 5.1 文件输出
在执行目录下生成 `<out_file>.xlsx`（默认 `rocshmem_test.xlsx`），包含：
- 每个输入目录对应一个工作表（工作表名取目录名前 31 个字符）
- 末尾一个 `EnvInfo` 工作表（若存在 `env_info.log`）

### 5.2 工作表结构
- **第 2 行**：数据目录路径、单位说明（Lat: us / Bw: GB/s）、当前展示模式
- **第 4 行**：环境信息跳转链接
- **第 5 行**：按操作类型分组的标签列（合并单元格）与列头
- **数据行**：每个用例一行，列依次为 `num-gpus`、`num-wgs`、`num-threads`，随后是各消息大小对应的数据

消息大小覆盖范围（共 27 档）：
```
8B, 16B, 32B, 64B, 128B, 256B, 512B, 1KB, 2KB, 4KB, 8KB, 16KB, 32KB, 64B,
128KB, 256KB, 512KB, 1MB, 2MB, 4MB, 8MB, 16MB, 32MB, 64MB, 128MB, 256MB, 512MB, 1GB
```

### 5.3 单元格内容格式
- **`--lat` 模式**：仅显示延迟值（us）
- **`--bw` 模式**：三行显示 `Bw / AlgBw / BusBw`（GB/s）
- **`--both` 模式**：四行显示 `Bw / AlgBw / BusBw / Lat`；仅有延迟或仅有带宽时分别加 `Lat:` / `Bw:` 前缀
- **失败用例**：合并整行数据列，红色显示 `Failed：<错误片段>`（最多 500 字符）

## 6. 常见问题
1. **报错 `ModuleNotFoundError: No module named 'xlsxwriter'`**
```bash
pip3 install xlsxwriter
```

2. **文件被跳过并提示 `⏭️ Skip`**
文件名不符合 `<op>_n<ngpus>_w<nwgs>_z<nthreads>` 命名规范，请检查日志文件命名。

3. **数据单元格为空**
该消息大小在日志中无对应记录，或 `volume` 小于 `minmsgsize`（默认 8）被过滤。可调整脚本顶部 `minmsgsize` 值。

4. **Excel 工作表名重复或被截断**
Excel 工作表名最长 31 字符，脚本自动截断目录名。若多个目录名前 31 字符相同，可能导致工作表冲突，建议重命名输入目录。

5. **OnStream 类操作显示的 `w` 与 `z` 与实际不一致**
脚本顶部注明：For OnStream operations, the actual value of w is 1. The actual z varies with size. When it is greater than 256, use 256. 属正常现象。
