# GDA 代理集合通信

本文说明 GDA 自研集合通信的代码组织、代理路由和执行约束。
集合算法位于 `../collectives/`；本目录提供候选管理、路由规划、同步协议和代理传输辅助。

## 配置与术语

`ROCSHMEM_DISABLE_PXN` 控制是否启用 PXN：

| 取值 | 行为 |
| --- | --- |
| `1`（默认） | 保留原有非 PXN 路径，不要求通信两端使用同名 NIC。 |
| `0` | 跨 IPC domain 传输使用同名 NIC 代理；不允许失败后回退到不受限制的跨域路径。 |

- **PE**：参加通信的进程。
- **IPC domain**：同一物理节点内能够通过 IPC 直接互通的一组 PE。一个节点可以包含多个 IPC domain。
- **代理 PE**：通过自身选定的 NIC，代表域内其他成员进行网络传输。
- **WORLD**：包含全部 PE 的 team；部分 team 仅包含其中一部分成员。

`ipc_domain_first_pe` 是 IPC domain 首个成员的 world PE，不是物理节点编号。
PXN 复用 GDA 已完成的 GPU–NIC 选择，不重新选择网卡。

## 代码组织

| 位置 | 职责 |
| --- | --- |
| `../context_gda_tmpl_device.hpp` | 社区接口实现及自研入口接入点。 |
| `../context_gda_tmpl_device_das.hpp` | 集合实现头文件的包含入口，以及两种模式共用的 QP 辅助。 |
| `../collectives/` | 自研集合入口、路径分派、数据组织、工作区和同步阶段管理。 |
| `candidate.hpp` | 代理 NIC 候选的构建与交换。 |
| `route_plan_host.hpp` | Host 路由规划、候选调度和路由状态。 |
| `layout.hpp` | WORLD 和部分 team 的布局检查。 |
| `route_device.hpp` | 设备侧能力检查、代理选择和路由计划构建。 |
| `entry_device.hpp` | 统一代理尝试入口及失败后的回退限制。 |
| `sync_device.hpp` | 域内汇聚、跨域同步、域内释放及线程作用域适配。 |
| `transfer_device.hpp` | 代理 GET/PUT 的共用调用边界。 |

集合入口直接定义在对应的集合文件中，不再增加独立的 dispatch 转发层。
`_das` 表示自研入口；模式专属实现以 `_pxn` 或 `_nopxn` 标识。

## 初始化与执行流程

1. `GDABackend::select_nics()` 选择各 PE 实际使用的 NIC。
2. `setup_proxy_candidates()` 收集代理候选，建立 PE、NIC 名称及 IPC domain 映射。
3. `setup_pxn_sync_routes()` 为 PXN 建立固定同步路由。
4. 创建 team 时，`setup_team_proxy_routes()` 检查成员内可用的代理布局。
5. 集合入口调用 `internal_try_collective_pxn()`，执行适用的 PXN 算法并统一处理失败策略。
6. 集合算法准备数据路由计划，通过代理传输，并在完成后推进相应状态。

`SelectProxyCollectiveLayout()` 描述可用的路由能力，不直接代表具体集合算法。
WORLD 对齐能力是优化条件；部分 team 则需要满足成员范围和域内代理约束。

## 同步路由与数据路由

同步代理在初始化或 team 创建时确定，用于稳定的阶段握手。
数据代理按任务约束和操作序号选择，WORLD 与部分 team 共用选择规则：

1. 筛选满足源、目标及 team 成员约束的候选。
2. 先轮询 NIC 通道，再在该通道内轮询代理 PE。
3. 准备路由计划，供本次操作的各阶段复用。

PXN 要求通信两端的 NIC 名称匹配。部分 team 的代理必须属于该 team。
固定目标、块所有者等算法约束优先于轮询，避免为负载均衡增加不必要的中转。

数据轮询使用独立的 `proxy_data_route_sequence`，不使用同步 epoch 代替。
同步 epoch 区分协议阶段；路由序号决定候选轮次，两者由参与成员一致推进。
并发操作通过独立 team 隔离工作区、同步槽及路由状态。

初始化日志中的候选映射和固定同步代理不代表每次操作实际选择的数据路由。

## 集合算法

| 操作 | PXN 执行方式 |
| --- | --- |
| sync / barrier | 域内汇聚，由固定代理跨域同步，再在域内释放。 |
| broadcast | 源代理准备数据、代理间 PUT、目标域 IPC 分发；WORLD 与部分 team 共用协议。 |
| fcollect | 域内打包后广播至最终输出；根据工作区容量合并或按成员分批。 |
| alltoall | 完整 WORLD 映射使用目标定向传输；其他支持的布局使用行传输流程。 |
| alltoallv | 全体一致确认等长连续布局后复用 alltoall；否则交换计数信息并传输打包行。 |
| allreduce / reduce-scatter | 共用分段归约框架：域内归约、根域合并、结果返回；输出范围按操作区分。 |

集合算法保留各自的数据排列、分块和工作区协议，路由层仅为已有任务分配代理。
非 PXN 保留原有广播、收集、交换和归约实现，不进入新增的数据代理轮询。

## 支持边界

- broadcast、fcollect、alltoall、allreduce、reduce-scatter 及 team sync/barrier 支持具备合法代理路由的跨 IPC domain 部分 team。
- PXN alltoallv 当前仅支持 WORLD 成员布局；range sync 的 PXN 接入也仅覆盖 WORLD 范围。
- 部分 team 必须在参与的每个 IPC domain 内找到可用的同名 NIC 成员代理；
  当前不提供 team 外代理推进机制。
- 多 NIC/PE 配置必须具备明确的 NIC 到 QP 映射，不能仅凭网卡名称推断 QP 归属。
- 对齐快路径依赖完整连续的 IPC domain 和有效同步布局；
  候选路由存在不表示所有算法均满足执行条件。
- PXN 失败后的普通路径仅限允许的域内处理，不能进入不受限制的跨域传输。
