# ZXDH 专有改动审计

## 1. 范围

本文只审计 ZXDH provider 自身的行为，不比较 duSHMEM 与 rocSHMEM 的整体架构。

对比来源：

```text
ZXDH 最新专有改动：master..ibgda（2b000f7）
rocSHMEM 目标分支：deepep-backend
```

rocSHMEM 已有的通用 GDA、QueuePair、堆注册、QP 建连、context 和多 QP 框架直接复用，不重复搬运。

## 2. 审计结论

首版 rocSHMEM 适配只覆盖了可编译的 RMA/quiet 主路径，确实漏掉了 ZXDH 专有逻辑。本次已补充：

| ZXDH 专有项 | 状态 | rocSHMEM 落点 |
|---|---|---|
| 完整 ZXDH DV ABI | 已补充 | `src/gda/zxdh/zxdh_dv.h` |
| CQ/QP 创建、shadow 初始化、UAR 映射 | 已覆盖 | `src/gda/zxdh/backend_gda_zxdh.cpp` |
| QP 状态迁移 | 已覆盖 | `zxdh_modify_qp` + loopback QP 状态迁移 |
| expert/QP 选择 | 已覆盖 | 现有 `qps[qp_idx]` 多 QP 路径 |
| 普通 RDMA WRITE/READ | 已覆盖 | `zxdh_post_wqe_rma` |
| inline WRITE | 已补充 | `zxdh_build_inline_wqe` |
| expert `rma_p` 的 READ-before-WRITE | 已补充 | `zxdh_post_value_dp` |
| ZXDH 双 loopback RC QP | 已补充 | `zxdh_create_loopback_qps` |
| GET/AMO CST | 已补充 | blocking GET 在返回前 CST；NBI GET 在提交前及最终 quiet 后 CST；AMO 仅 `fetching=true` 使用 CST |
| dedicated put-signal（SET） | 已补充 | `GDAContext::putmem_signal*` |
| CQ polarity/WQE 对应校验 | 已补充 | `zxdh_poll_cq` |
| CQ 超时和错误码检查 | 已覆盖 | `zxdh_poll_cq` |
| AMO 临时 buffer 隔离 | 已补充 | per-QP `fetching_atomic` ring slots |
| AMO 硬件原子语义 | 受硬件能力限制 | QP 内软件 READ-modify-WRITE，多 QP 并发仍需验证 |
| WQE batching / warp coalescing | 未移植 | 性能优化，不影响主路径正确性 |
| CPU NIC handler | 未移植 | 当前 backend 只使用 GPU polling 路径 |
| 多 NIC / rail proxy | 未移植 | 当前 rocSHMEM backend 仍是单 NIC 选择模型 |

## 3. 本次补充的文件

```text
M src/gda/backend_gda.cpp
M src/gda/backend_gda.hpp
M src/gda/context_gda_device.cpp
M src/gda/queue_pair.cpp
M src/gda/queue_pair.hpp
M src/gda/zxdh/backend_gda_zxdh.cpp
M src/gda/zxdh/queue_pair_zxdh.cpp
A src/gda/zxdh/queue_pair_zxdh_multiqp.cpp
A src/gda/zxdh/segment_builder_zxdh.hpp
A src/gda/zxdh/segment_builder_zxdh.cpp
M src/gda/zxdh/zxdh_dv.h
```

查看补充内容：

```bash
git diff 62dec6f..1b77cc6
```

查看全部 ZXDH 适配：

```bash
git diff da3491f..1b77cc6 -- src/gda/zxdh src/gda/backend_gda.cpp \
  src/gda/backend_gda.hpp src/gda/queue_pair.cpp src/gda/queue_pair.hpp \
  src/gda/context_gda_device.cpp
```

## 4. 编译验证

在 s3 的 `yuji-test-vllm` 容器内验证：

| 配置 | 结果 |
|---|---|
| `USE_GDA=ON GDA_ZXDH=ON` | 编译通过，0 errors |
| `cmake --install` | 通过 |
| `USE_GDA=ON GDA_MLX5=ON` | 回归编译通过，0 errors |

s3 当前只暴露 mlx5 设备，因此仍需在 ZXDH 网卡机器验证：

1. 双 loopback RC QP 建连和 CST；
2. RDMA WRITE/READ、inline WRITE 和 expert-QP 路径；
3. 多 QP 并发下的 AMO/signal；
4. DeepEP normal 和 low-latency 模式；
5. SQ/CQ wrap、错误 CQE 和超时处理。
