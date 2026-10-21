# rocSHMEM 功能测试驱动脚本使用文档

> 源码路径：`scripts/functional_tests/driver.sh`；安装后路径：`<安装前缀>/share/rocshmem/rocshmem_functional_driver.sh`
> 适用对象：rocSHMEM 功能测试套件 `rocshmem_functional_tests`

## 1. 概述

`driver.sh` 是 rocSHMEM 功能测试的核心调度脚本，负责：

- 根据传入参数选择测试套件或单个测试用例执行
- 自动探测 GPU 数量与 wavefront 大小，按需跳过资源不足的用例
- 支持两种运行时：MPI（多节点/多进程）与 SLR（Simple Local Runtime，单节点）
- 自动收集构建信息、系统/GPU/网络环境信息到日志目录
- 失败用例自动重试（阈值可配置）
- 可选生成性能热力图等可视化产物

安装测试组件后，驱动脚本与测试程序位于同一目录；直接运行示例见第 7 节。

## 2. 命令行用法

```bash
./driver.sh <executable> <test_suite | test_name | test_config> <log_dir> [hostfile] [options]
```

### 2.1 位置参数

| 参数 | 必填 | 说明 |
|------|------|------|
| `<executable>` | 是 | `rocshmem_functional_tests` 可执行文件路径 |
| `<test_suite>` | 是 | 测试套件名，如 `all`、`rma`、`put` |
| `<test_name>` | 是 | 单个测试用例名，如 `putnbi`、`amo_fadd` |
| `<test_config>` | 是 | 带参数的测试配置字符串，如 `"putnbi 2 8 1024 65536"` |
| `<log_dir>` | 是 | 日志输出目录（不存在会自动创建） |
| `[hostfile]` | 否 | MPI hostfile 路径，仅 MPI 模式可用，与 SLR 互斥 |

### 2.2 可选参数

| 参数 | 说明 |
|------|------|
| `--show-cases` | 打印所有可用测试用例名后退出 |
| `--artifact-dir DIR` | `heatmap`、`heatmaprma`、`heatmapcoll` 测试通过后，将可视化产物生成到 DIR；其他套件虽接受该参数，但不会生成产物 |
| `--artifact-dir=DIR` | 与上述空格形式等价 |

## 3. 测试套件

通过 `<test_suite>` 参数选择预定义的测试集合：

| 套件名 | 说明 |
|--------|------|
| `all` | 除 tile/host 外的全部功能测试（默认） |
| `gda` / `gda-mlx5` / `gda-bnxt` / `gda-ionic` | GDA 后端全套（跳过 IPC 专属用例） |
| `ro` / `all-ro` | RO 后端全套（跳过已知不兼容用例，前缀 `all-` 会被去除） |
| `rma` | 远程内存访问（Put/Get，含非阻塞与 wavegroup/wave 变体） |
| `put` / `get` | 仅 RMA Put 或 Get 子集 |
| `amo` | 原子内存操作（add/fadd/inc/cswap 等） |
| `sigops` | 信号量操作（putsignal/signalfetch 等） |
| `coll` | 集合通信（barrier/alltoall/broadcast/fcollect 等） |
| `stream` | Stream 上的内存与集合操作 |
| `other` | 初始化、pingpong、flood、team context、fence 等杂项 |
| `tiles` | Tile 级别的 Put/Get/Broadcast/Allgather |
| `heatmap` | 性能热力图采集（RMA + Coll，大消息体积） |
| `heatmaprma` / `heatmapcoll` | 热力图的 RMA 或 Coll 子集 |
| `*host` | 主机侧非 MPI IPC 测试（需构建启用 `USE_IPC=ON`） |
| `perf*` / `perf-mlx5*` | 性能测试模式，见下文 |

### 3.1 性能测试模式

语法：`perf <test_name> <ranks> [max_msg_size]`

```bash
# 全部性能测试，2 ranks，最大 1MB
./driver.sh $APP "perf all 2 1048576" logs

# 仅 RMA 性能，2 ranks，默认最大 8MB
./driver.sh $APP "perf rma 2" logs

# mlx5 后端 coll，2 ranks，最大 64KB
./driver.sh $APP "perf-mlx5 coll 2 65536" logs
```

- `max_msg_size` 缺省值为 `8388608`（8MB）
- workgroup 选项由 `ROCSHMEM_TEST_WGS` 控制（默认 `16`）
- thread 选项由 `ROCSHMEM_TEST_THDS` 控制（默认 `128 256`）
- 当 `wg * threads > 65536` 时自动跳过该组合
- 结果输出到 `<log_dir>/<test>_n<ranks>_<maxsize>B_best.log`，记录各消息体积下的最佳带宽与延迟
- `perf*` 模式会自动启用 `NOVERIF`，跳过结果正确性校验。

### 3.2 自定义测试配置

直接以字符串形式传入单个用例的参数：

```bash
# 5 字段：名称 ranks workgroups threads max_msg_size
./driver.sh $APP "putnbi 2 8 1024 65536" logs

# 4 字段：名称 ranks workgroups threads（无 max_msg_size）
./driver.sh $APP "amo_fadd 2 1 64" logs

# 仅名称：使用默认值 ranks=2 wg=1 thd=1 size=8
./driver.sh $APP "putnbi" logs
```

## 4. 运行时选择

### 4.1 MPI 模式（默认）

- 使用 `mpirun` 启动多进程
- 支持多节点（通过 hostfile）
- 默认参数：`-mca pml ucx -mca osc ucx --map-by numa`
- 可通过 `ROCSHMEM_TEST_MPI_PARAMS` 覆盖默认 launcher 参数；使用自定义参数测试时，需在该变量中包含 hostfile、所需 MPI 组件参数及远端环境变量

### 4.2 SLR 模式（单节点）

设置环境变量 `ROCSHMEM_TEST_SLR=1` 启用：

- 不使用 MPI，通过 `ROCSHMEM_SLR_NP` 控制 PE 数量
- 仅限单节点，**不能** 与 hostfile 同用
- 当可用 GPU 数 < 测试所需 rank 数时自动跳过

```bash
ROCSHMEM_TEST_SLR=1 ./driver.sh $APP all logs
```

## 5. 环境变量

### 5.1 运行时与资源

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ROCSHMEM_TEST_SLR` | `0` | 设为 `1` 启用 SLR 模式 |
| `ROCSHMEM_TEST_UUID` | — | 设为 `1` 强制使用 uniqueid bootstrap（host 测试自动设置） |
| `ROCSHMEM_MAX_NUM_CONTEXTS` | — | 最大 context 数 |
| `ROCSHMEM_MAX_NUM_HOST_CONTEXTS` | — | 主机侧最大 context 数 |
| `ROCSHMEM_HEAP_SIZE` | 6GB（heatmap=22GB） | 堆内存大小（字节） |
| `NUM_GPUS` | 自动探测 | GPU 数量，手动设置可跳过探测 |
| `IPC_HOST_NPES` | `4` | `host_amo_all_pes`/`host_amo_self` 的 PE 数 |

### 5.2 MPI 参数

| 变量 | 说明 |
|------|------|
| `ROCSHMEM_TEST_MPI_PARAMS` | 自定义参数，覆盖默认 mpirun 参数（不含 `-n`），如 `-x LD_LIBRARY_PATH -x ROCSHMEM_BACKEND=gda` |
| `OMPI_MCA_pml` | 默认 `ucx` |
| `OMPI_MCA_osc` | 默认 `ucx` |
| `HOSTFILE` | 由环境变量或第 4 个位置参数 `[hostfile]` 传入，仅默认 MPI 参数使用；自定义 MPI 参数时，需在 `ROCSHMEM_TEST_MPI_PARAMS` 中指定 `--hostfile` |

### 5.3 测试行为

| 变量 | 说明 |
|------|------|
| `ROCSHMEM_TEST_USE_DEFAULT_STREAM` | 设为 `1` 时传入 default stream 标志 |
| `ROCSHMEM_TEST_ARGS` | 附加到测试可执行文件后的额外参数 |
| `LOCALBUFTYPE` | 用户缓冲区类型：`heap`/`host`/`device`/`fine`/`uncached`/`managed` |
| `NOTIMEOUT` | 任意非空值禁用 5 分钟超时 |
| `NOVERIF` | 任意非空值跳过数据校验（heatmap/perf 自动启用） |
| `RETRY_THRESHOLD` | `5` | 触发自动重试的最大失败用例数，超过则跳过重试 |

### 5.4 性能测试参数

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `ROCSHMEM_TEST_WGS` | `16` | 性能测试 workgroup 选项（空格分隔） |
| `ROCSHMEM_TEST_THDS` | `128 256` | 性能测试 thread 选项（空格分隔） |

### 5.5 性能对比

| 变量 | 说明 |
|------|------|
| `PERF_BASELINE_DIR` | 基线构建目录（双构建对比模式） |
| `PERF_BRANCH_DIR` | 分支构建目录（双构建对比模式） |

二者同时设置时触发完整热力图对比；仅设置 `--artifact-dir` 时仅生成单次运行的 per-test 图。

## 6. 测试用例命名约定

测试用例名与 `tester.hpp` 中的 `TestType` 枚举一一对应，编号映射在脚本顶部 `TEST_NUMBERS` 关联数组中定义：

- `0`–`149`：社区原生用例（get/put/amo/coll/stream/host 等）
- `200`–`203`：自研用例（`*_dp` 双精度路径，从 200 起避免冲突）

使用 `--show-cases` 可列出全部可用名称：

```bash
./driver.sh placeholder --show-cases
```

## 7. 典型用法示例

**安装包直接运行**：将下例的安装前缀替换为构建时使用的 `INSTALL_PREFIX`（即 `CMAKE_INSTALL_PREFIX`）。驱动脚本与测试程序同在 `<安装前缀>/share/rocshmem`，日志写入可写的 `/tmp` 目录。

```bash
ROCSHMEM_INSTALL_PREFIX=/path/to/install-prefix
cd "${ROCSHMEM_INSTALL_PREFIX}/share/rocshmem"
./rocshmem_functional_driver.sh ./rocshmem_functional_tests rma /tmp/rocshmem-test-logs
```

以下示例以源码构建目录为例。先将 `BUILD_DIR` 替换为实际构建目录，并设置共用的测试程序和驱动脚本路径；使用安装包时，改用上面的安装路径。

```bash
BUILD_DIR=/path/to/rocshmem/build
cd "$BUILD_DIR"
APP=tests/functional_tests/rocshmem_functional_tests
DRIVER=../scripts/functional_tests/driver.sh
```

### 7.1 单节点 MPI 基础功能测试

不设置 `ROCSHMEM_TEST_SLR` 时，脚本默认通过 MPI 启动，无需手动调用 `mpirun`。

```bash
# 跑全部功能测试
$DRIVER $APP all logs

# 仅跑 RMA
$DRIVER $APP rma logs

# 跑单个用例
$DRIVER $APP putnbi logs

# 带参数的单个用例
$DRIVER $APP "putnbi 2 8 1024 65536" logs
```

### 7.2 多节点 MPI

```bash
# 通过 hostfile
$DRIVER $APP all logs ./hostfile

# 自定义 MPI 参数时，在变量内传入 hostfile
ROCSHMEM_TEST_MPI_PARAMS="--hostfile ./hostfile --mca pml ucx --mca osc ucx --map-by numa -x LD_LIBRARY_PATH -x PATH -x ROCSHMEM_BACKEND=gda" \
  $DRIVER $APP all logs
```

### 7.3 单节点 SLR

```bash
ROCSHMEM_TEST_SLR=1 $DRIVER $APP all logs
```

### 7.4 性能测试

```bash
# 全部性能测试，8 ranks
$DRIVER $APP "perf all 8" perf-logs

# 自定义 wg/thread 组合, 仅性能测试起作用
ROCSHMEM_TEST_WGS="8 16 32" ROCSHMEM_TEST_THDS="64 128 256" \
  $DRIVER $APP "perf coll 8 1048576" perf-logs
```

### 7.5 热力图采集（含产物）

```bash
# 单次构建，仅 per-test 图
$DRIVER $APP heatmap logs-heatmap --artifact-dir ./artifacts

# 双构建对比（branch vs develop）
PERF_BASELINE_DIR=$DEVELOP_BUILD \
PERF_BRANCH_DIR=$BRANCH_BUILD \
  $DRIVER $APP heatmap logs-heatmap --artifact-dir ./artifacts
# 产物：
#   ./artifacts/heatmap_summary.png
#   ./artifacts/heatmap_summary.txt
#   ./artifacts/heatmap_data.csv
#   ./artifacts/per_test/*.png
```

### 7.6 用户缓冲区类型测试

```bash
# 测试缓冲区（通过 test_config 间接生效，或直接修改套件，默认heap）
LOCALBUFTYPE=heap $DRIVER $APP "putnbi 2 32 128 512" logs
```

> 套件内 `TestRMAPut`/`TestRMAGet` 已内置对 `heap`/`host`/`device`/`fine`/`uncached`/`managed` 五种缓冲区类型的循环测试。

### 7.7 自定义组合测试

以下示例展示一次完整的 GDA 后端集合通信性能测试：以 mlx5 后端、16 ranks 运行 `coll` 套件，最大消息 8 MB，自定义 workgroup/thread 组合，并通过 `ROCSHMEM_TEST_MPI_PARAMS` 注入多节点 hostfile 与一系列后端调优参数。

```bash
HOSTFILE=/path/to/hostfile
LOG_DIR=/path/to/logs/coll-performance

MPI_PARAMS="\
  --allow-run-as-root \
  --hostfile ${HOSTFILE} \
  --mca coll_hcoll_enable 0 \
  -x ROCSHMEM_BACKEND=gda \
  -x LD_LIBRARY_PATH \
  -x PATH \
  -x ROCSHMEM_TEST_UUID=1 \
  -x ROCSHMEM_HEAP_SIZE=42949672960 \
  -x ROCSHMEM_GDA_NUM_QPS_DEFAULT_CTX=2 \
  -x ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX=2 \
  -x ROCSHMEM_ALIGN_BW_WITH_RCCL=1 \
  -x ROCSHMEM_IB_GID_INDEX=1 \
  -x ROCSHMEM_GDR_DISABLE_XDP=1"

ROCSHMEM_TEST_WGS="16 32" \
ROCSHMEM_TEST_THDS="256 512" \
ROCSHMEM_TEST_MPI_PARAMS="$MPI_PARAMS" \
  $DRIVER "$APP" \
    "perf-mlx5 coll 16 8388608" \
    "$LOG_DIR"
```

**参数说明**：

| 参数 | 作用 |
|------|------|
| `ROCSHMEM_TEST_WGS` / `ROCSHMEM_TEST_THDS` | 性能测试 workgroup 与 thread 组合（笛卡尔积） |
| `--allow-run-as-root` | 允许以 root 身份运行 mpirun |
| `--mca coll_hcoll_enable 0` | 关闭 hcoll 集合通信组件，避免与 rocSHMEM 集合操作冲突 |
| `-x ROCSHMEM_BACKEND=gda` | 指定 GDA 后端 |
| `-x LD_LIBRARY_PATH` / `-x PATH` | 将环境变量传播到所有节点 |
| `-x ROCSHMEM_TEST_UUID=123` | 强制使用 uniqueid bootstrap（多节点必需） |
| `--hostfile=...` | 指定多节点 hostfile |
| `-x ROCSHMEM_HEAP_SIZE=42949672960` | 堆内存大小（约 40 GB） |
| `-x ROCSHMEM_GDA_NUM_QPS_*` | GDA 后端 QP 数量调优（默认 ctx 与用户 ctx 各 2 个） |
| `-x ROCSHMEM_ALIGN_BW_WITH_RCCL=1` | 带宽结果对齐 RCCL 计算口径 |
| `-x ROCSHMEM_IB_GID_INDEX=1` | 指定 IB GID 索引（多端口网卡场景常见） |
| `-x ROCSHMEM_GDR_DISABLE_XDP=1` | 禁用 GDR 的 XDP 路径 |

## 8. 日志与输出

### 8.1 日志文件

每个用例生成一个日志文件，命名格式：

```
<log_dir>/<test_name>_n<ranks>_w<workgroups>_z<threads>[_<max_msg_size>B].log
```

重试日志后缀为 `.retry.log`。日志首行记录实际执行的命令，便于复现。

### 8.2 环境信息

`<log_dir>/env_info.log` 记录：

- rocSHMEM 构建信息（`rocshmem_info --env:all`）
- DTK 版本
- 系统/CPU/GPU 信息
- 网络接口与 IB/RoCE 状态

### 8.3 终端输出

- `Test:   <name>` — 用例开始
- `Retry:  <name>` — 重试用例
- `Skip:   <name> (reason)` — 跳过用例（GPU 不足、后端不兼容等）
- `FAILED: <name>` / `PASSED: <name>` — 结果（带颜色）

### 8.4 性能测试结果

`<log_dir>/<test>_n<ranks>_<maxsize>B_best.log` 汇总各消息体积下的：

- 最佳带宽（GB/s）及对应 `wg:thread` 配置
- AlgBw / BusBw（若测试输出包含）
- 最佳延迟（us）及对应 `wg:thread` 配置

## 9. 失败重试机制

1. 首次执行记录所有失败用例及其完整环境参数
2. 若失败数 ≤ `RETRY_THRESHOLD`（默认 5），自动重试一次
3. 重试时恢复原始环境变量（`ROCSHMEM_TEST_USE_DEFAULT_STREAM`、`ROCSHMEM_MAX_NUM_CONTEXTS`、`NOTIMEOUT`、`NOVERIF`）
4. 超过阈值则跳过重试，提示可能存在系统性问题
5. 最终摘要区分"重试后通过"与"重试后仍失败"

## 10. 后端兼容性说明

不同后端对用例的支持差异已在套件内通过条件判断处理：

| 后端 | 跳过的用例类别 | 原因 |
|------|----------------|------|
| `ro*` | get/g/amo_add/flood/putmem_signal/fence/hostteamsyncbarrier | 当前后端不支持这些用例 |
| `gda*` | `_g` / `flood_g` | 当前 GDA 后端不支持这些路径 |
| `gda-mlx5*` | — | 额外启用 `*_dp` 双精度路径用例 |
| 非 IPC 构建 | `tile_*`、`host_*` | 需 `USE_IPC=ON` |

## 11. Wavefront 大小自动探测

脚本根据 GPU 架构自动设置 `WAVE_SIZE`：

- `gfx936` / `gfx938`：`WAVE_SIZE=64`

该值用于 `tile_*` 系列用例的线程数计算（如 `tile_put_wave_contiguous` 使用 `$WAVE_SIZE`，`tile_put_wg_contiguous` 使用 `$((WAVE_SIZE * 16))`）。

## 12. 退出码

- `0`：全部用例通过（含重试后通过）
- 非 `0`：存在失败用例

退出码可通过 `$?` 获取，便于 CI 集成。
