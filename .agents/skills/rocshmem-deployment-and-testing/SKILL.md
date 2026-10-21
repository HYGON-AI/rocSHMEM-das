---
name: rocshmem-deployment-and-testing
description: 用于部署、配置、验证、性能测试和排查面向 Hygon HCU 适配的 rocSHMEM。适用于 DTK/HCU 环境检查、IPC/GDA/RO 后端选择、HCU-NIC 拓扑、运行配置、功能验证、RMA/AMO/COLL 性能测试、日志解析与 Excel 报告导出及已确认问题排查；不用于通用的社区 ROCm 硬件指导。
---

# rocSHMEM 部署与测试

本 Skill 用于面向 Hygon HCU 适配的 rocSHMEM。平台部署说明使用 **HCU** 和 DTK；引用源码接口、工具名称及 `README-driver.md` 的通用 GPU 表述时保留原称谓。`GPU_TARGETS`、`rocminfo` 和 `ROCM_PATH` 是构建系统或工具链保留的字面名称。

## 事实来源

在给出版本、配置、命令、性能测试、验收或故障排查建议前，先定位并阅读 [`docs/rocshmem_deployment_and_testing_guide.md`](../../../docs/rocshmem_deployment_and_testing_guide.md) 的相关章节。查询 `driver.sh` 的命令行接口、套件、默认值、日志、重试或退出码时，读取 [`scripts/functional_tests/README-driver.md`](../../../scripts/functional_tests/README-driver.md)。解析测试日志或导出 Excel 报告时，读取 [`scripts/functional_tests/README-heatmap-parse-rocshmem-general.md`](../../../scripts/functional_tests/README-heatmap-parse-rocshmem-general.md)，并以配套脚本源码为准。不要在本 Skill 中复制这些文档的参数表。

按以下顺序判定信息权威性：

1. 某个后端、参数或接口是否由当前版本实现，以当前源码和构建脚本为准。
2. `driver.sh` 的参数解析、默认行为和输出，以脚本源码为准，`README-driver.md` 作为其使用说明。
3. Hygon HCU 环境下经验证的推荐值、操作流程和验收口径，以部署与测试指南为准。
4. 当前节点的软硬件版本、路径、NIC、PCI BDF 和拓扑，以现场检查结果为准。

根据任务读取对应章节：

- 后端选择、环境、编译、拓扑、内存、Context 和 QP 的通用原则：阅读“后端选择”至“与部署相关的编程约束”。
- 正确性验证、性能测试执行和结果对比：阅读“验证与性能测试”。
- `driver.sh` 的命令语法、套件名、默认行为、日志和退出码：阅读 `README-driver.md`，必要时对照脚本源码。
- 测试日志解析、Excel 热力图导出：阅读 `README-heatmap-parse-rocshmem-general.md`，核对输入日志格式和解析脚本；不必为单纯导出报告重读部署配置章节。
- 部署或运行故障：阅读“故障排查”，优先返回用户可执行的检查和处理方式；源码缺陷交由维护人员定位。
- 发布或生产上线准备：阅读“生产部署清单”。
- 全文评审或文档更新：阅读完整指南，并与当前源码树核对。

## 执行约束

- 命令中的主机名、NIC、PCI BDF、安装路径等现场值须标明已核实，或写成明确的占位符。源码、指南与实际环境不一致时，说明差异。
- 生产配置与性能测试配置分开，不直接复用对称堆、Context 和 QP 参数。仅在相同后端、拓扑和 PE 数下通过正确性验证后，使用 `-noverif` 采集性能数据。
- 多节点 MPI 任务须将必要的运行时变量传递到远端 rank。每个 PE 绑定一张 HCU；跨节点 GDA 部署须核对自动选择的 HCU–NIC 映射，仅在映射不符或任务要求固定映射时使用强制拓扑文件。
- 给出可执行命令时，说明适用的后端、进程数和必要前提；环境变量与命令行选项的字面名称保持原样。
