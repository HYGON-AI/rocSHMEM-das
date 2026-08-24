# rocSHMEM 单元测试驱动脚本使用文档

> 脚本路径：`scripts/unit_tests/driver.sh`
> 适用对象：rocSHMEM 单元测试二进制 `rocshmem_unit_tests`

## 1. 概述

`driver.sh` 是 rocSHMEM 单元测试的调度脚本，负责：

- 根据传入参数选择预定义测试集合或自定义 GTest 过滤器执行
- 自动探测节点 GPU 数量，仅在资源充足时运行多 rank 用例
- 通过 `mpirun` 启动多进程，单次执行超时时间为 20 分钟
- 将每个用例的输出追加写入带时间戳的日志文件

## 2. 命令行用法

```bash
./driver.sh <executable> <mode> [args...]
```

### 2.1 位置参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `<executable>` | 是 | 单元测试可执行文件路径（如 `rocshmem_unit_tests`） |
| `<mode>` | 是 | 运行模式，取值为 `all` 或 `custom` |

### 2.2 模式说明

| 模式 | 附加参数 | 说明 |
|------|----------|------|
| `all` | 无 | 执行预定义的标准测试集合 |
| `custom` | `<ranks> <filter>` | 以自定义 MPI rank 数与 GTest 过滤字符串运行 |

参数数量校验规则：

- 总参数少于 2：报错退出
- `all` 模式参数不等于 2：报错退出
- `custom` 模式参数不等于 4：报错退出
- `custom` 模式下 `ranks` 必须 > 1，否则报错退出

## 3. 运行模式

### 3.1 `all` 模式

执行预定义的标准测试集合，GTest 过滤器为：

```
-IPCImplSimpleCoarseTestFixture/*:IPCImplSimpleFineTestFixture/*:IPCImplTiledFineTestFixture/*:DegenerateTiledFine.*
:SdmaSimpleCoarse/*:SdmaSimpleFine/*:SdmaTiledFine/*
```

- 前导 `-` 表示**排除**匹配上述过滤器的用例（即运行这些用例之外的其余用例）
- 仅当探测到 GPU 数 ≥ 4 时，以 4 个 MPI rank 启动执行
- GPU 数 < 4 时不执行任何用例（脚本中 `run_mpirun 2 ...` 已被注释）

```bash
./driver.sh rocshmem_unit_tests all
```

### 3.2 `custom` 模式

由用户指定 MPI rank 数与 GTest 过滤字符串：

```bash
# 以 2 个 rank 运行指定过滤器
./driver.sh rocshmem_unit_tests custom 2 "IPCImplSimpleCoarseTestFixture/*"

# 以 4 个 rank 运行多个用例
./driver.sh rocshmem_unit_tests custom 4 "SdmaSimpleCoarse/*:SdmaSimpleFine/*"
```

> `ranks` 必须为大于 1 的正整数，否则报错并打印帮助信息。

## 4. GPU 数量探测

脚本按以下顺序探测节点 GPU 数量（`NUM_GPUS`）：

1. 若环境变量 `NUM_GPUS` 已设置，直接使用
2. 否则尝试 `hy-smi`，统计 `GPU` 行数
3. 若 `hy-smi` 不可用，尝试 `hy-smi --showserial`，统计 `GPU` 行数
4. 若上述探测均失败，回退为 `0`，最终兜底为 `8`

> 兜底值 `8` 仅在探测失败时生效，确保 CI 环境下 `all` 模式仍可触发 4 rank 执行。

## 5. 架构兼容性

若 `rocminfo` 输出中包含 `gfx1201`，脚本会：

- 打印 `Unit tests disabled in gfx1201` 与 `See AIROCSHMEM-393`
- 创建空日志文件后直接退出（退出码 0）

> 该跳过逻辑用于规避 `gfx1201` 上单元测试的已知问题，避免 CI 误报失败。

## 6. 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `NUM_GPUS` | 自动探测 | 节点 GPU 数量，手动设置可跳过探测 |
| `ROCSHMEM_DEBUG_LEVEL` | `WARN` | 由脚本固定注入到 `mpirun` 命令前，控制 rocSHMEM 日志级别 |

> `mpirun` 命令固定参数：`--timeout $mpi_timeout`，其中 `mpi_timeout=1200`（20 分钟）。

## 7. 日志输出

### 7.1 日志文件命名

每次脚本执行生成一个日志文件，命名格式：

```
unit_tests_<YYYY-MM-DD-HH:MM:SS>.log
```

示例：`unit_tests_2026-08-14-15-30-00.log`

### 7.2 日志内容

- 每个 `mpirun` 命令的标准输出与标准错误均追加写入同一日志文件
- 失败时脚本会将完整日志内容打印到 stderr，便于 CI 直接捕获

### 7.3 终端输出

- 执行前打印完整的 `mpirun` 命令字符串
- 失败时打印 `FAILED: <cmd_str>` 到 stderr
- 结束时打印 `Tests Completed` 与日志文件路径

## 8. 典型用法示例

### 8.1 基础用法

```bash
cd $BUILD_DIR
APP=tests/unit_tests/rocshmem_unit_tests

# 运行全部预定义用例（需 ≥ 4 GPU）
./scripts/unit_tests/driver.sh $APP all

# 自定义 rank 与过滤器
./scripts/unit_tests/driver.sh $APP custom 2 "IPCImplSimpleFineTestFixture/*"
```

### 8.2 调整 GPU 数量探测

```bash
# 强制按 8 GPU 处理
NUM_GPUS=8 ./scripts/unit_tests/driver.sh $APP all

# 单 GPU 环境下 all 模式不会触发任何执行
NUM_GPUS=1 ./scripts/unit_tests/driver.sh $APP all
```

### 8.3 显式指定可执行文件路径

```bash
./scripts/unit_tests/driver.sh /path/to/rocshmem_unit_tests all
./scripts/unit_tests/driver.sh /path/to/rocshmem_unit_tests custom 4 "SdmaTiledFine/*"
```

## 9. 退出码

| 退出码 | 含义 |
|--------|------|
| `0` | 全部用例通过 |
| `1` | 至少一个 `mpirun` 命令失败 |

> 退出码可通过 `$?` 获取，便于 CI 集成判定。失败时脚本不会立即终止，会继续执行后续用例，最终汇总返回。
