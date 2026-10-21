# rocSHMEM 部署与测试指南

> 本文面向 Hygon HCU 平台上的 rocSHMEM 开发与部署。本文所述 rocSHMEM 是基于 rocSHMEM 社区实现进行 Hygon HCU 平台适配和功能增强的版本。示例中的版本、参数和配置来自特定验证环境，不是所有部署的默认要求；主机名、NIC 名称、PCI BDF 和路径须按目标集群替换。通信并发度、消息规模、对称内存需求、后端或硬件拓扑变化时，应重新验证正确性、稳定性和性能。

## 1. rocSHMEM 适合解决什么问题

rocSHMEM 提供类似 OpenSHMEM 的单边通信接口，并允许 HCU kernel 直接发起远端内存访问、原子操作和集合通信。它使用位于 HCU 显存中的对称堆，适合以下场景：

- 需要细粒度 HCU-HCU 通信，并希望将通信与计算重叠；
- MoE、图计算、模板计算、分布式 GEMM 等不规则或频繁通信负载；
- 希望避免每次通信都回到 CPU；
- 同一程序需要覆盖单机 HCU 互联和跨节点 RDMA。

部署时须核验通信粒度和同步范围；跨节点 GDA 部署还须核验 HCU-NIC 自动映射及网络配置，不能只替换通信 API。

## 2. 后端选择

| 后端 | 数据路径 | 适用范围 | 优点 | 注意事项 |
|---|---|---|---|---|
| IPC | HCU 通过节点内 IPC/load-store 通信 | 单节点 | 路径短、延迟低 | 不支持跨节点 |
| RO（Reverse Offload） | HCU 请求经 CPU 代理，由 MPI-RMA 完成 | 跨节点兼容路径 | 兼容不支持 GDA 的硬件 | CPU 代理进入关键路径；依赖支持 HCU 内存的 Open MPI/UCX |
| GDA（沿用实现中的后端名称） | HCU 直接构造 WQE、提交 NIC SQ 并轮询 CQ | 跨节点，配合 IPC 使用 | 避免 CPU 代理，支持高并发与通信计算重叠 | 必须正确配置 NIC、驱动、固件、拓扑和 QP |

按部署范围直接选择：

1. 纯单节点生产任务使用 IPC-only。
2. 配备 MLX5 或 SHCA RDMA NIC 的跨节点生产任务使用 GDA + IPC：节点内走 IPC，节点间走 GDA。
3. NIC 不支持 GDA 时才使用 RO + IPC。
4. `all_backends` 只用于需要同一安装包跨多种机型运行的环境；固定机型的生产包不得使用该配置。

注意：GDA + IPC 或 RO + IPC 构建中，原子操作会统一走网络后端，不一定走 IPC。原子密集型应用必须单独测量。

## 3. 环境与部署前检查

### 3.1 基础要求

- DTK 环境（支持 HIP），默认安装在 `/opt/dtk`；使用当前分支发布说明或回归测试覆盖的版本。
- Open MPI 和 UCX；RO 后端及仓库测试需使用支持 HCU 内存的构建。
- GDA 跨节点通信需使用受支持的 MLX5 或 SHCA RDMA NIC。
- 支持 `gfx936` 和 `gfx938` 架构；每个 PE 绑定一张 HCU，并保持 NUMA、HCU 与 NIC 亲和关系。

## 4. 编译与安装

### 4.1 推荐构建方式

在 rocSHMEM 源码根目录中，选择与目标部署类型对应的一组命令执行。推荐每种配置使用独立的 build 目录；复用同一目录切换后端时，必须先清空其中全部编译产物和 CMake 缓存，再重新配置和编译。

```bash
# 单节点 IPC
mkdir -p build-ipc && cd build-ipc
../scripts/build_configs/ipc_single
```

```bash
# GDA + IPC：MLX5 网卡
mkdir -p build-gda-mlx5 && cd build-gda-mlx5
../scripts/build_configs/gda_mlx5
```

```bash
# GDA + IPC：Hygon HCU 平台 SHCA 网卡
mkdir -p build-gda-shca && cd build-gda-shca
../scripts/build_configs/gda_shca
```

```bash
# RO + IPC
mkdir -p build-ro && cd build-ro
../scripts/build_configs/ro_ipc
```

```bash
# 多后端通用包
mkdir -p build-all && cd build-all
../scripts/build_configs/all_backends
```

构建脚本默认安装到 `/opt/rocshmem`，默认目标包含 `gfx936` 和 `gfx938`；实际目标以当前 DTK 和 CMake 配置输出为准。仅在需要覆盖默认安装路径或目标架构时设置 `INSTALL_PREFIX` 或 `GPU_TARGETS`。用 `/opt/dtk/bin/rocminfo` 的 agent `Name` 字段确认完整架构字符串；生产包使用 `Release` 构建，不遗留 ASAN、API trace 或 `HSA_XNACK=1` 等调试设置。

### 4.2 编译应用

rocSHMEM 设备代码需要 relocatable device code。生产应用优先使用 rocSHMEM 导出的 CMake targets，链接方式参考仓库中的 `examples/CMakeLists.txt`，由 CMake 传递拆分库及后端依赖。以下命令仅演示使用单体 `librocshmem.a` 手工编译基础应用：

```bash
export DTK_ROOT=/opt/dtk
export ROCSHMEM_INSTALL_DIR=/opt/rocshmem
export MPI_ROOT=/opt/mpi
export UCX_ROOT=/opt/ucx
export PATH=${DTK_ROOT}/bin:${UCX_ROOT}/bin:${MPI_ROOT}/bin:${PATH}
export LD_LIBRARY_PATH=${DTK_ROOT}/lib:${DTK_ROOT}/hip/lib:${UCX_ROOT}/lib:${MPI_ROOT}/lib:${LD_LIBRARY_PATH:-}

${DTK_ROOT}/bin/hipcc -c -fgpu-rdc -x hip app.cc \
  -I${ROCSHMEM_INSTALL_DIR}/include -I${MPI_ROOT}/include

${DTK_ROOT}/bin/hipcc -fgpu-rdc --hip-link app.o -o app \
  ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a \
  ${MPI_ROOT}/lib/libmpi.so \
  -L${DTK_ROOT}/lib -L${DTK_ROOT}/hip/lib \
  -lamdhip64 -lhsa-runtime64
```

GDA 应用手工链接时还需增加 `-lnuma`；启用其他可选功能时应补充对应依赖，因此不建议用手工命令替代 CMake。运行时的 `mpiexec` 必须来自编译时使用的 MPI 安装，避免系统中多个 MPI 混用。

## 5. 网络拓扑与 HCU-NIC 绑定

### 5.1 部署前确认拓扑

以下命令用于核验代码自动识别的 HCU-NIC 拓扑：

```bash
/opt/hyhal/bin/hy-smi --showtopo
/opt/hyhal/bin/hy-smi --showbus
lspci -D | grep -Ei 'display|vga|ethernet|infiniband'
ibv_devinfo
numactl --hardware
```

默认情况下，rocSHMEM 根据 OAM、PCIe 和 NUMA 拓扑自动选择就近 NIC，并在候选 NIC 之间进行负载分配。上述信息用于核对自动选择结果，不要求人工逐张 HCU 配置 NIC。

### 5.2 限制候选 NIC

自动选择结果正确时，不设置 NIC 过滤变量。集群中存在管理网、不可达端口或不参与通信的 NIC 时，优先使用 `ROCSHMEM_HCA_LIST` 限制候选范围；rocSHMEM 仍会在列表中自动完成 HCU-NIC 映射：

```bash
# 允许列表：生产环境只允许使用列出的 NIC
export ROCSHMEM_HCA_LIST=mlx5_0,mlx5_1,mlx5_2,mlx5_3

# 多 provider 构建中显式选择 provider
export ROCSHMEM_GDA_PROVIDER=mlx5
```

`ROCSHMEM_HCA_LIST` 在自动拓扑选择阶段过滤候选 NIC；`ROCSHMEM_ALLOWED_IBV_DEVICES` 在更早的 IBV 设备枚举阶段过滤设备，通常不需要设置。两者同时设置时只保留共同允许的设备，不要配置两份相同列表。

启动脚本按 local rank 设置 `HIP_VISIBLE_DEVICES`，保证一进程一 HCU。不建议为整个多 PE 作业统一设置单个 `ROCSHMEM_USE_IB_HCA`，否则所有 PE 会集中到同一 NIC。

### 5.3 可选的强制拓扑文件

仅当自动识别结果不符合实际拓扑，或应用需要固定映射以保证结果可复现时，才设置：

```bash
export ROCSHMEM_TOPO_FILE_FORCE=/etc/rocshmem/topo.config
```

`topo.config` 每行描述 HCU PCI 地址、NIC 名和 IBV 枚举序号。以下仅示意 8 HCU × 4 NIC 的两两共享映射；PCI 地址、NIC 名和序号必须按本机 `hy-smi --showbus`、`ibv_devinfo` 及实际枚举顺序核实后替换，序号不能直接由 NIC 名称后缀推断：

```text
0000:09:00.0 mlx5_0 0
0000:36:00.0 mlx5_0 0
0000:55:00.0 mlx5_1 1
0000:77:00.0 mlx5_1 1
0000:85:00.0 mlx5_2 2
0000:b5:00.0 mlx5_2 2
0000:d5:00.0 mlx5_3 3
0000:f5:00.0 mlx5_3 3
```

使用强制拓扑文件时，将其纳入机器配置管理，并在更换 HCU、NIC 或主板后同步更新。未设置 `ROCSHMEM_TOPO_FILE_FORCE` 时，rocSHMEM 使用自动选择结果。

### 5.4 RoCE 与多端口

- RoCE 网络应由集群侧统一配置 PFC/ECN、MTU、GID 和 traffic class；单机参数无法弥补无损网络配置错误。
- `ROCSHMEM_GDA_TRAFFIC_CLASS` 必须与交换网络策略一致，不能照搬其他集群的值。
- 使用经验证的多端口 bonded NIC 配置时，可设置 `ROCSHMEM_GDA_ALTERNATE_QP_PORTS=1`，让不同 context 的 QP 交替使用端口。

## 6. 对称堆、上下文和 QP 配置

### 6.1 对称堆

`ROCSHMEM_HEAP_SIZE` 是每个 PE 的 HCU 对称堆大小。按下式核算并保留 10%~20% 余量：

```text
heap >= 长生命周期对称对象
     + 并发收发/暂存缓冲区
     + collective scratch
     + 对齐与分配器开销
     + 10%~20% 安全余量
```

所有 PE 必须以相同顺序执行对称分配/释放。不要把普通 `hipMalloc` 指针当成对称地址使用；GDA 外部用户缓冲区必须通过 `rocshmem_buffer_register` 注册、使用完后通过 `rocshmem_buffer_unregister` 注销。生产代码不得超过构建版本规定的最大注册数量。

对称堆大小不是通用固定值，应根据实际对称对象和并发缓冲区核算。

### 6.2 Context

根据同时发起通信的 workgroup 数量、context 生命周期和资源预算确定 `ROCSHMEM_MAX_NUM_CONTEXTS`。编程时遵循：

- 低并发或串行控制路径使用 default context，减少资源和初始化成本。
- 多个 workgroup 并发通信时，创建 workgroup-private context，降低共享状态竞争。
- context 的创建与销毁应由对应 collective scope 的所有线程共同参与；不可让部分线程提前退出。
- 未创建大量用户 context 的低并发 RMA 程序保留默认值。

### 6.3 QP 与 SQ

根据目标 PE 数、并发通信流数和 NIC 资源预算确定 QP 数量。使用 `ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX` 和 `ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX`，分别设置 default context 和 user context 的每目标 PE QP 数。低并发应用可使用以下配置：

```bash
export ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX=1
export ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX=1
```

第 8 章的 `5/5`、`32/32` 均按上述顺序表示 default context 与 user context 的每目标 PE QP 数。

SQ 深度只在程序明确报告容量不足，或应用配置明确要求时调整。

## 7. 与部署相关的编程约束

### 7.1 生命周期与错误边界

典型顺序如下：

```cpp
rocshmem_init();

int me = rocshmem_my_pe();
int npes = rocshmem_n_pes();
void* symmetric = rocshmem_malloc(bytes);

// 初始化、通信 kernel、同步与校验

rocshmem_free(symmetric);
rocshmem_finalize();
```

- 初始化后立即校验 PE 数、rank-HCU 绑定和对称堆分配结果。
- 在计时区间外完成一次 warm-up，排除首次初始化、页表和连接建立开销。
- 每个 kernel launch 后检查 HIP 错误；调试阶段在关键边界同步，生产阶段移除不必要的 host synchronization。

### 7.2 选择合适的 API 粒度

- 单线程发起极少量操作：使用普通 API。
- 一个 wavefront 协同搬运连续数据：使用 `_wave`/wavefront collective 变体。
- 一个 workgroup 共同搬运较大数据：使用 `_wg` 变体，并保证工作组所有线程以一致参数参与。
- 二维/多维带 stride 数据（如 halo exchange）：使用 Tile API，避免拆分为大量手工小包。
- 小而零散的传输先在寄存器或 LDS 聚合，再发起较大连续传输。

gfx936/gfx938 的 wavefront 按 64 线程组织；参与 wavefront/workgroup collective API 的线程块尺寸必须是 64 的整数倍。

### 7.3 完成语义和内存顺序

- `fence` 用于保证同一目标 PE 上操作的顺序，不等价于等待远端完成。
- `quiet` 用于等待此前由相应 context 发起的操作完成。
- 数据加 signal 的生产者-消费者模式使用 put-with-signal，并由接收端 wait/test signal，不增加全局 barrier。
- 只在算法确实需要所有 PE 对齐时使用 `barrier_all`；局部依赖使用 team、signal 或点对点同步缩小范围。
- context 的完成语义相互独立：在哪个 context 发起，就在哪个 context 上完成。

### 7.4 通信与计算重叠

使用双缓冲流水线：准备块 N+1、传输块 N、计算块 N-1。通信 kernel 与计算 kernel 放入不同 HIP stream，计时区间内不得调用 `hipDeviceSynchronize()`；只使用 event 建立真实的数据依赖。

## 8. 验证与性能测试

`driver.sh` 的完整命令行接口、套件列表、脚本默认值、日志格式、重试机制和退出码，参见 [`README-driver.md`](../scripts/functional_tests/README-driver.md)。本章只维护 Hygon HCU 环境下经过验证的测试配置、执行顺序和验收口径。

### 8.1 验收步骤与通过标准

以下是每节点 8 HCU 验证环境的参考验收口径。PE 规模、运行时长和性能阈值应按目标部署及项目标准确定，并在报告中记录。

1. **环境验收**：所有节点的 rocSHMEM、DTK、驱动和 NIC 固件版本完全一致；`ENV:full` 日志显示一 PE 一 HCU，且自动选择的 NIC 符合实际拓扑。使用强制拓扑文件时，映射还必须与该文件一致。
2. **初始化验收**：2 PE、单节点 8 PE、双节点 16 PE 均连续启动和退出 100 次，无超时、崩溃或资源泄漏。
3. **正确性验收**：RMA、signal、atomic、collective 功能测试全部通过，覆盖 0、1、8、64、256、4096 字节及应用最大消息长度。
4. **稳定性验收**：使用生产配置满并发连续运行 24 小时，要求零数据错误、零死锁、零进程异常退出。
5. **性能验收**：首次上线将通过前四项验收后的结果归档为机型基线；后续版本的吞吐不得低于该基线的 95%，P99 延迟不得高于该基线的 110%。

仓库构建 examples 后，将 `ROCSHMEM_BUILD_DIR` 设为实际构建目录并执行：

```bash
export ROCSHMEM_BUILD_DIR="$(git rev-parse --show-toplevel)/build-ipc"

mpiexec --map-by numa --mca pml ucx --mca osc ucx \
  -np 2 "${ROCSHMEM_BUILD_DIR}/examples/rocshmem_getmem_test"

mpiexec --map-by numa -np 2 \
  "${ROCSHMEM_BUILD_DIR}/examples/rocshmem_put_signal_test"
```

随后通过 CTest 和 `scripts/functional_tests/driver.sh` 执行完整功能测试。单节点 IPC 与跨节点 GDA 必须分别通过，不能用 IPC 结果代替 GDA 验收。

### 8.2 性能测试专用配置

性能测试使用独立构建和独立环境变量，不得覆盖应用的生产配置。根据测试范围分别构建：节点内使用 `ipc_single`，节点间 MLX5 使用 `gda_mlx5`；两类都测时两组命令均执行，每组均从 rocSHMEM 源码根目录开始。

```bash
# 节点内 IPC
mkdir -p build-perf-ipc && cd build-perf-ipc
../scripts/build_configs/ipc_single
```

```bash
# 节点间 GDA/MLX5
mkdir -p build-perf-gda-mlx5 && cd build-perf-gda-mlx5
../scripts/build_configs/gda_mlx5
```

| 参数 | 性能测试值 | 用途 |
|---|---:|---|
| `ROCSHMEM_TEST_WGS` | RMA/COLL：`16 32`；AMO：`1 2 8` | 扫描每 PE 的 workgroup 数；AMO 不使用大 workgroup 数 |
| `ROCSHMEM_TEST_THDS` | `128 256 512` | 扫描每个 workgroup 的线程数 |
| `ROCSHMEM_TEST_UUID` | `1` | 固定 unique-ID bootstrap |
| `ROCSHMEM_HEAP_SIZE` | IPC：`42949672960`；GDA：`50737418240` | 为对应后端的大规模并发测试预留约 40/47 GiB 对称堆 |
| `ROCSHMEM_ALIGN_BW_WITH_RCCL` | `1` | 统一 rocSHMEM 与 RCCL 的带宽计算口径 |
| `ROCSHMEM_GDR_DISABLE_XDP` | `1` | 禁用 XDP 路径，固定对比数据路径 |
| `ROCSHMEM_IB_GID_INDEX`、`UCX_IB_GID_INDEX` | 示例为 `1` | 按业务网实际 GID index 设置，两个通信栈及所有 rank 保持一致 |
| `-noverif` | 仅单项性能测试使用 | 跳过结果校验；必须在正确性测试通过后使用 |

### 8.3 单用例性能测试

先执行不带 `-noverif` 的正确性测试。节点内 IPC，8 PE，32 workgroups，每组 512 线程，最大消息 8 MiB：

```bash
export ROCSHMEM_BUILD_DIR="$(git rev-parse --show-toplevel)/build-perf-ipc"
cd "${ROCSHMEM_BUILD_DIR}"

mpirun --allow-run-as-root --map-by numa --mca coll_hcoll_enable 0 \
  -np 8 -x ROCSHMEM_BACKEND=ipc -x LD_LIBRARY_PATH -x PATH \
  -x ROCSHMEM_TEST_UUID=1 -x ROCSHMEM_HEAP_SIZE=42949672960 \
  -x ROCSHMEM_MAX_NUM_CONTEXTS=32 \
  -x ROCSHMEM_ALIGN_BW_WITH_RCCL=1 -x ROCSHMEM_GDR_DISABLE_XDP=1 \
  tests/functional_tests/rocshmem_functional_tests \
  -a 19 -w 32 -z 512 -s 8388608
```

节点间 GDA，双节点各 8 PE，最大消息 8 MiB。先将 `node-a` 和 `node-b` 替换为实际可达主机名：

```bash
export ROCSHMEM_TEST_HOSTS='node-a:8,node-b:8'
export ROCSHMEM_BUILD_DIR="$(git rev-parse --show-toplevel)/build-perf-gda-mlx5"
cd "${ROCSHMEM_BUILD_DIR}"

mpirun --allow-run-as-root --mca coll_hcoll_enable 0 \
  -np 16 -H "${ROCSHMEM_TEST_HOSTS}" -x ROCSHMEM_BACKEND=gda \
  -x LD_LIBRARY_PATH -x PATH -x ROCSHMEM_TEST_UUID=1 \
  -x ROCSHMEM_HEAP_SIZE=50737418240 -x ROCSHMEM_ALIGN_BW_WITH_RCCL=1 \
  -x ROCSHMEM_IB_GID_INDEX=1 -x UCX_IB_GID_INDEX=1 \
  -x ROCSHMEM_GDR_DISABLE_XDP=1 \
  -x ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX=5 \
  -x ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX=5 \
  tests/functional_tests/rocshmem_functional_tests \
  -a 19 -w 32 -z 512 -s 8388608
```

上述用例在同一后端、拓扑、PE 数和参数下验证通过后，才可以在原命令末尾加入 `-noverif` 进行计时。`--allow-run-as-root` 仅适用于容器内的已授权测试；非容器环境不使用该选项。

### 8.4 `driver.sh` 批量性能测试

`driver.sh` 的 `perf*` 模式会自动启用 `NOVERIF`。执行下表性能套件前，必须先在相同后端、拓扑和 PE 数下执行对应的 `rma`、`amo` 或 `coll` 功能套件，并确认结果校验通过。性能套件按通信类型分别执行，不使用一组参数跑完所有用例：

| 后端 | 类别 | PE 数 | `ROCSHMEM_TEST_WGS` | `ROCSHMEM_TEST_THDS` | driver suite | 对称堆 | 输出目录 |
|---|---|---:|---|---|---|---:|---|
| IPC | RMA | 2 | `16 32` | `128 256 512` | `perf rma 2 8388608` | `42949672960` | `perf_ipc` |
| IPC | AMO | 2 | `1 2 8` | `128 256 512` | `perf amo 2` | `42949672960` | `perf_ipc` |
| IPC | COLL | 8 | `16 32` | `128 256 512` | `perf coll 8 v1073741824` | `42949672960` | `perf_ipc` |
| GDA/MLX5 | RMA | 2 | `16 32` | `128 256 512` | `perf-mlx5 rma 2 8388608` | `50737418240` | `perf_gda` |
| GDA/MLX5 | AMO | 2 | `1 2 8` | `128 256 512` | `perf-mlx5 amo 2` | `50737418240` | `perf_gda` |
| GDA/MLX5 | COLL | 16 | `16 32` | `128 256 512` | `perf-mlx5 coll 16 v1073741824` | `50737418240` | `perf_gda` |

AMO 数据长度固定为 8 字节。大 workgroup 数会增加 AMO 延迟，因此只扫描 `1 2 8`。先按 [`README-driver.md` 的完整示例](../scripts/functional_tests/README-driver.md) 设置 `ROCSHMEM_TEST_MPI_PARAMS`；自定义参数会覆盖脚本默认 MPI 选项和位置参数 hostfile。跨节点测试须在该变量中显式加入 `--hostfile`、所需 MPI 组件参数和远端环境变量。再按下列模板选择套件：

```bash
ROCSHMEM_TEST_WGS='<表中的 WG 列表>' \
ROCSHMEM_TEST_THDS='128 256 512' \
ROCSHMEM_TEST_MPI_PARAMS='<mpirun 参数及所有 -x 环境变量>' \
../scripts/functional_tests/driver.sh \
  tests/functional_tests/rocshmem_functional_tests \
  '<表中的 driver suite>' \
  '/home/result_rocshmem/<输出目录>'
```

GDA RMA/AMO 双 PE 测试的 `ROCSHMEM_TEST_MPI_PARAMS` 中加入强制多 NIC 合并，并使用 `32/32` QP。以下 NIC 分组只是已验证环境示例，必须根据目标节点的 HCU–NIC 拓扑替换：

```bash
GDA_RMA_AMO_MPI_PARAMS="${ROCSHMEM_TEST_MPI_PARAMS:-} \
 -x ROCSHMEM_GDA_NET_FORCE_MERGE=mlx5_2,mlx5_3,mlx5_4,mlx5_5;mlx5_6,mlx5_7,mlx5_8,mlx5_9 \
 -x ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX=32 \
 -x ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX=32"

ROCSHMEM_TEST_MPI_PARAMS="$GDA_RMA_AMO_MPI_PARAMS" \
  ../scripts/functional_tests/driver.sh \
  tests/functional_tests/rocshmem_functional_tests \
  'perf-mlx5 rma 2 8388608' \
  '/home/result_rocshmem/perf_gda'
```

`40/40` 仅用于明确的 RMA/AMO 饱和测试；GDA collective 使用 `5/5`。不得用 `perf*` 自动禁用校验的结果代替功能正确性验收。

### 8.5 性能测量规范

- 测试范围：节点内、跨节点一跳和全规模分别测量、分别报告；不叠加节点内互联带宽与 RDMA 带宽。
- 结果记录：注明消息大小、PE 数、每节点 HCU/NIC 数、后端、provider、QP/context/SQ、协议、时钟及软件版本；说明带宽、延迟的统计与计算口径，不以单次最好成绩代表结果。
- 横向对比：与 RCCL/MPI 比较时，统一数据量定义、同步边界、预热方式和进程绑定，并记录实际预热与采样次数。

将 `driver.sh` 的测试日志解析为 Excel 热力图报告时，参见 [`README-heatmap-parse-rocshmem-general.md`](../scripts/functional_tests/README-heatmap-parse-rocshmem-general.md)；解析脚本的参数、输入格式和工作表结构以该文档及脚本源码为准。

## 9. 故障排查

本表只列部署用户可通过环境、启动参数或构建配置处理的问题。若按表排查后仍可复现，应记录日志、版本和复现步骤，提交给维护人员定位。

| 阶段 | 典型现象 | 已确认原因 | 处理方式 |
|---|---|---|---|
| 环境准备 | 找不到 UCX；提示 MCA accelerator、PML 或 PLM 组件缺失 | UCX/Open MPI 安装不完整，或运行时加载了不匹配的 DTK、MPI/UCX 库 | 核对 DTK、UCX、Open MPI 的实际安装路径及 `PATH`、`LD_LIBRARY_PATH`；用当前 `mpirun`、`ompi_info` 确认组件，缺失时安装配套的 UCX/Open MPI |
| MPI 启动 | root 被拒绝；容器 SSH 失败 | Open MPI 安全限制，或 SSH 端口未传递 | 容器增加 `--allow-run-as-root`；非 22 端口配置 `plm_rsh_agent` |
| 跨节点启动 | 找不到 `libgalaxyhip.so.5`；UUID、QP 参数未生效 | 环境变量未传播 | 使用 `mpirun -x` 传播 `PATH`、`LD_LIBRARY_PATH`、UUID、后端、QP、context、heap 和 GID |
| RO 初始化 | 无 IB 卡时 HCOLL 报错 | MPI 仍加载 IB 组件 | 设置 `--mca coll_hcoll_enable 0 --mca btl ^openib`，并将 `UCX_NET_DEVICES` 指向业务网卡 |
| RO/IPC 初始化 | UCX assertion | UCX 1.22/1.23 与 Open MPI 5.0.3 组合不兼容 | 使用已验证的 UCX 1.15.0，或重新配套编译 UCX/Open MPI |
| GDA 初始化 | `modify_qp (RTR): Connection timed out (110)`；`ibv_create_ah` 超时 | GID index 与网络类型不匹配，或 NIC/端口不可达 | 核对实际端口的 GID 表、RoCE 模式和连通性；按查到的业务网 GID index 向所有 rank 传递 `ROCSHMEM_IB_GID_INDEX`，使用 UCX 时同步设置 `UCX_IB_GID_INDEX`；不要直接套用 0/1 |
| GDA 加载 | `no DV library could dlopen for IONIC, BNXT, or MLX5 GDA support` | SHCA 环境的 provider/DV 依赖缺失或版本不匹配 | 核对 SHCA 网卡对应的 provider 与 `shca-tools` 安装，使用与当前 DTK、驱动配套的版本后重新验证 |
| GDA 资源创建 | `ibv_create_cq_ex: Cannot allocate memory (12)` | 进程可锁定内存额度不足 | 用 `ulimit -l` 检查 memlock；容器部署时在创建容器前配置足够额度，必要时添加 `--cap-add=IPC_LOCK` 和 `--ulimit memlock=-1:-1` 后重建容器 |
| GDA 初始化 | QP 数大于 1 时卡住 | 各 rank 的 QP 参数不一致 | 通过 `mpirun -x ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX -x ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX` 将已设置的两个 QP 变量传给所有 rank，确保各 rank 配置一致 |
| GDA 启动 | 未指定后端时段错误 | 自动选择错误地进入 IPC 后端 | 显式设置并传播 `ROCSHMEM_BACKEND=gda` |
| 应用运行 | 多 PE 应用退出崩溃或通信超时 | rank-HCU 映射与启动参数不一致 | 检查启动器和应用的设备绑定逻辑；不要未经验证直接加入 `--map-by numa` |
| HCU 运行时 | `Failed to lock memory pool` | 已确认的 DTK runtime 组合中，SVM 内存池锁定失败 | 在确认应用不依赖 SVM 的前提下，可临时向各 rank 设置 `HSA_USE_SVM=0` 并重新验证；后续使用修复该问题的 runtime |
| MPI 依赖 | `ImportError: libhcoll.so.1: cannot open shared object file` | 镜像中的 MPI 依赖了缺失的 HCOLL 库 | 核对 MPI 与 HCOLL 的动态库依赖；安装匹配的 HCOLL，或重编不依赖该库的 MPI，并保持编译与运行使用同一套 MPI |
| 编译链接 | `post_wqe_amo` undefined hidden symbol | LTO 与符号可见性组合不兼容 | 设置 `CMAKE_CXX_VISIBILITY_PRESET=default`、`CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF` |
| 编译链接 | HIP 内置扩展接口未定义 | DTK、目标架构与扩展接口不匹配 | 使用 `/opt/dtk/bin/hipcc` 和 `gfx936;gfx938`，清理 CMake cache 后重编 |
| 编译后运行 | `hipGetSymbolAddress(... HIP_SYMBOL(print_lock))` 失败 | 编译产物的 gfx 目标架构与实际 HCU 不匹配 | 用 `/opt/dtk/bin/rocminfo` 核对目标 HCU 的 agent `Name`，以匹配的 `GPU_TARGETS` 清理构建目录并重编 |

### 日志开关与采集

rocSHMEM 日志分为运行时级别和编译期开关。先用运行时变量调整输出；只有需要 API 或内部 trace、而当前库未包含对应日志时，才启用编译期开关并重新构建。

| 用途 | 设置方法 |
|---|---|
| 默认错误日志 | `ROCSHMEM_DEBUG_LEVEL=ERROR`；这是默认级别 |
| 查看启动环境与构建信息 | 运行时设为 `ENV:full` 或 `VERSION`，例如 `ROCSHMEM_DEBUG_LEVEL=ENV:full ./app` |
| 查看 API/内部跟踪 | 运行时设为 `API` 或 `TRACE`；host trace 需构建时启用 `BUILD_DEBUG_TRACE_HOST=ON`，device trace 需启用 `BUILD_DEBUG_DEVICE=ON` 和 `BUILD_DEBUG_TRACE_DEVICE=ON` |
| 查看 device warn/info 与 GDA device 检查 | 构建时启用 `BUILD_DEBUG_DEVICE=ON` |
| 输出 GDA 统计计数 | 以 `-DPROFILE=ON` 构建，并运行时设置 `ROCSHMEM_DEBUG_LEVEL=INFO:stats` |

编译期开关需在 CMake 配置后重新构建 rocSHMEM；仅提高运行时级别不会补出未编入库的 trace。多节点运行时应将变量显式传给各 rank，例如 `mpirun -x ROCSHMEM_DEBUG_LEVEL=ENV:full ...`，并检查各 rank 的输出。诊断结束后恢复生产构建和默认 `ERROR` 级别，避免 trace 日志影响运行与性能。

## 10. 生产部署清单

- [ ] 构建类型为 `Release`，后端、provider、`GPU_TARGETS` 与目标机器一致。
- [ ] 一 PE 一 HCU，CPU/NUMA/HCU/NIC 绑定经过日志确认。
- [ ] 对称堆大小已根据应用的实际内存需求核算。
- [ ] 对称分配顺序一致，collective participation 无分支发散。
- [ ] QP、Context 和 SQ 已根据应用并发度和网络资源验证。
- [ ] GDA 跨节点部署的 GID index 与集群网络一致，允许列表中的所有 NIC 均验证该 GID 可达。
- [ ] 跨节点启动命令已通过 `-x` 传播 PATH、动态库路径、后端、QP、context、heap、GID 和 bootstrap 变量。
- [ ] 单机 IPC、跨节点 GDA/RO 的正确性和长稳测试均通过。
- [ ] 性能报告包含完整硬件、软件、网络和环境变量快照。
- [ ] 调试变量、ASAN、trace、`HSA_XNACK=1` 已从生产环境清除。
- [ ] 启动脚本进入版本管理；若使用强制拓扑文件，也纳入版本管理，否则保留自动映射的验收记录。

## 11. 部署与测试要点

Hygon HCU 上的 rocSHMEM 已验证部署基线遵循以下规则：

1. 单节点固定使用 IPC，跨节点固定使用 GDA + IPC；RO 仅作为不支持 GDA 时的兼容方案。
2. 一 PE 绑定一张 HCU；跨节点 GDA 默认自动选择 NIC，并按实际 PCIe/NUMA 拓扑核验映射，必要时才限制候选 NIC 或使用强制拓扑文件。
3. 对称堆、Context、QP 和 SQ 参数由应用负载及硬件资源决定，不使用其他应用的固定值。
4. GDA 使用多端口网络时核验端口轮换配置；RoCE traffic class 与集群无损网络配置保持一致。
5. 通信 kernel 优先使用 wavefront/workgroup collective API，数据与 signal 合并发送，局部同步不使用全局 barrier。
6. 生产环境使用 `Release` 构建和 `ROCSHMEM_DEBUG_LEVEL=ERROR`，不得遗留 ASAN、trace 或 `HSA_XNACK=1`。

应用配置应以实际通信并发度、消息规模、对称内存需求和目标拓扑为依据。后端、DTK 版本或网络拓扑发生变化时，先重新执行正确性和稳定性验证，再根据明确指标调整。

## 12. 常用环境变量速查

只列本文部署与测试场景中需要判断是否设置的变量；表中的“传递”指使用 MPI 跨节点启动时，显式设置的值是否需要通过 `mpirun -x` 等方式送到各 rank。示例数值不是通用默认值。

| 变量 | 作用与设置时机 | 跨节点传递 |
|---|---|---|
| `ROCSHMEM_BACKEND` | 指定 `ipc`、`gda` 或 `ro`；多后端构建或自动选择不符合目标部署时显式设置 | 显式设置时传给所有 rank |
| `HIP_VISIBLE_DEVICES` | 将每个 local rank 绑定到目标 HCU | 按 local rank 分别设置；不要用一条 `-x` 给所有 rank 传相同设备编号 |
| `ROCSHMEM_HEAP_SIZE` | 每 PE 对称堆大小（字节）；需要改变默认容量时按第 6.1 节的实际需求核算 | 显式设置时传给所有 rank |
| `ROCSHMEM_MAX_NUM_CONTEXTS` | 用户 context 上限；并发 context 需求超出默认值时设置 | 显式设置时传给所有 rank |
| `ROCSHMEM_GDA_NUM_QPS_PER_PE_DEFAULT_CTX` | GDA default context 的每目标 PE QP 数 | 显式设置时传给所有 rank |
| `ROCSHMEM_GDA_NUM_QPS_PER_PE_USR_CTX`、`ROCSHMEM_GDA_SQ_SIZE` | 用户 context 的每目标 PE QP 数、SQ 深度；仅在并发或容量需求明确时调整 | 显式设置时传给所有 rank |
| `ROCSHMEM_GDA_PROVIDER` | 多 provider 构建中选定网卡 provider；单 provider 构建通常不必设置 | 显式设置时传给所有 rank |
| `ROCSHMEM_HCA_LIST`、`ROCSHMEM_ALLOWED_IBV_DEVICES` | 分别在自动拓扑选择、IBV 枚举阶段限制候选 NIC；自动选择正确时不设置 | 显式设置时传给所有 rank，值按各节点 NIC 名核对 |
| `ROCSHMEM_USE_IB_HCA`、`ROCSHMEM_TOPO_FILE_FORCE` | 分别强制单个 NIC、按文件强制 HCU-NIC 映射；只在需要覆盖自动选择时使用 | 由受影响 rank 获取对应值；拓扑文件须在各节点可读 |
| `ROCSHMEM_IB_GID_INDEX`、`UCX_IB_GID_INDEX` | 分别指定 rocSHMEM GDA、UCX 使用的业务网 GID index；按实际端口 GID 表选择 | 使用对应通信栈且显式设置时传给相关 rank |
| `ROCSHMEM_GDA_TRAFFIC_CLASS`、`ROCSHMEM_GDA_ALTERNATE_QP_PORTS` | 分别匹配 RoCE 网络策略、控制多端口 QP 轮换；仅在网络配置要求时覆盖 | 显式设置时传给所有 GDA rank |
| `ROCSHMEM_DEBUG_LEVEL` | 临时设置 `ENV:full` 核对运行参数；生产环境使用默认的 `ERROR` 级别 | 需要采集各 rank 日志时传给所有 rank |
| `PATH`、`LD_LIBRARY_PATH` | 确保远端 rank 使用与本地一致的 DTK、MPI、UCX 及 rocSHMEM 工具和库 | 远端环境未统一配置时显式传递 |

`ROCSHMEM_TEST_WGS`、`ROCSHMEM_TEST_THDS`、`ROCSHMEM_TEST_MPI_PARAMS` 等由测试脚本在启动端读取，不等同于需要逐项 `-x` 的 rocSHMEM 运行时变量；具体用法见 [`README-driver.md`](../scripts/functional_tests/README-driver.md)。完整变量定义见 [`env_variables.rst`](env_variables.rst)，分支新增变量和实际默认值以当前源码为准。

## 参考资料

- 本仓库 `README.md`、`docs/build.rst`、`docs/compile_and_run.rst`、`docs/env_variables.rst`、`DEBUG.md` 与 examples。
