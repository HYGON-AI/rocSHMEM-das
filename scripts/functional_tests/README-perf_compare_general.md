# rocSHMEM 通用性能对比工具

`perf_compare_general.py` 对比已有测试日志，输出 BASE/NEW 并排的 Excel 带宽报告，不构建或执行测试。以下命令从仓库根目录执行，目录和名称按现场替换。

## 工具选择

| 工具 | 用途 | 输出 |
|---|---|---|
| `perf_compare_general.py` | 两组 rocSHMEM，或 rocSHMEM 与 RCCL；按 Volume、IPC/GDA 和 W/Z 展示 | Excel |
| `perf_compare.py` | 一个 rocSHMEM 基线与多个版本；支持多轮统计 | 延迟曲线、汇总热力图、CSV、TXT |

统计脚本的独立用法见 [README-perf_compare.md](README-perf_compare.md#generate-the-heatmap-from-existing-logs-no-rebuild-no-re-run)。

## 依赖

```bash
python3 -m pip install xlsxwriter
```

## 命令与参数

```text
python3 scripts/functional_tests/perf_compare_general.py 
  --base INTRA INTER --new INTRA INTER [可选参数]
```

目录顺序始终为节点内、节点间。BASE 仅支持 rocSHMEM，NEW 默认也是 rocSHMEM；NEW 为 RCCL 时指定 `--vs rccl`。BASE 发现 RCCL 日志时直接报错，混合目录也不接受。

| 参数 | 必选 | 含义与默认值 |
|---|---|---|
| `--base INTRA INTER` | 是 | BASE 的 IPC、GDA 日志目录 |
| `--new INTRA INTER` | 是 | NEW 的节点内、节点间日志目录 |
| `--vs {rocshmem,rccl}` | 否 | NEW 类型，默认 rocshmem |
| `--ops {coll,rma,all}` | 否 | 默认 coll；rma 为单边通信；all 为两者。RCCL 仅允许 coll |
| `--labels BASE_NAME NEW_NAME` | 否 | 表格组名，默认 BASE/NEW；RCCL 模式为 BASE/RCCL |
| `--wgs N[,N...]` | 否 | 设备侧工作组数；不指定时不限制 |
| `--threads N[,N...]` | 否 | 设备侧每组线程数；不指定时不限制 |
| `--rccl-placement {out,in}` | 否 | RCCL 取 out-of-place 或 in-place，默认 out |
| `--min-volume BYTES` | 否 | 最小数据量，默认 256 字节 |
| `--max-volume BYTES` | 否 | 最大数据量，默认不限制 |
| `--device-type NAME` | 否 | 覆盖从日志识别的设备名称 |
| `-o OUTPUT` / `--output OUTPUT` | 否 | 输出文件；自动补 .xlsx；默认 rocshmem_vs_rocshmem.xlsx 或 rocshmem_vs_rccl.xlsx |
| `-h` / `--help` | 否 | 查看完整帮助 |

## 示例

### rocSHMEM 优化前后比较

```bash
python3 scripts/functional_tests/perf_compare_general.py \
  --base ./before/perf_ipc ./before/perf_gda \
  --new ./after/perf_ipc ./after/perf_gda \
  --labels 优化前 优化后 \
  --ops all --wgs 16,32 --threads 128,256 \
  -o comparison.xlsx
```

省略 `--ops` 时仅比较集合通信；单边通信使用 `--ops rma`。

### rocSHMEM 与 RCCL 比较

```bash
python3 scripts/functional_tests/perf_compare_general.py \
  --base ./rocshmem/perf_ipc ./rocshmem/perf_gda \
  --new ./rccl/intra ./rccl/inter --vs rccl \
  --labels rocSHMEM RCCL \
  --wgs 16,32 --threads 128,256 --rccl-placement out \
  -o rocshmem_vs_rccl.xlsx
```

## 输入日志

- 递归读取目录下的 `.log`。rocSHMEM 文件名采用 `<op>_n<PE>_w<W>_z<Z>_...log`，忽略 `_best.log` 和名称含 `.retry.` 的文件。
- rocSHMEM 使用八列：Volume、MsgSize、MsgCount、Latency、Bandwidth、AlgBwAlignRccl、BusBwAlignRccl、MsgRate。对齐带宽允许 `N/A`；本工具不读取旧六列日志。
- RCCL 按 `all_reduce_perf.log` 等用例文件名匹配；也兼容包含 `Test: ...` / `测试: ...` 的合并日志。
- rocSHMEM 按第一列 Volume 对齐；RCCL 按 size 列对齐。同一用例、后端和 Volume 有多条记录时，rocSHMEM 选择原始 Bandwidth 最大的整条记录，RCCL 选择 algbw 最大的整条记录。
- 每个目录应只包含同一 PE 规模和待比较配置的数据。当前工具不按文件名中的 PE 数单独分组，混合多个规模会合并取最大值。

## 工作表与计算口径

- 一个“主机侧”页签，包含 on-stream 用例；主机侧忽略 W/Z 筛选。
- 设备侧按 W/Z 笛卡尔积生成页签。例如 16,32 与 128,256 生成 w16z128、w16z256、w32z128、w32z256。
- W 或 Z 未指定时对应维度不限，合并选择最大带宽；两者都不指定时使用一个“设备侧”页签。
- 每页分别展示 IntraNode 和 InterNode。all 模式先集合通信、后单边通信；双方都没有数据的接口跳过。
- 只有 GDA 有数据也会展示，IPC 列留空。仅 BASE 或 NEW 有数据时保留单侧结果，变化率留空；分母为零也留空。
- rocSHMEM 变化率：`NEW.Bandwidth / BASE.Bandwidth - 1`。
- RCCL 变化率：`RCCL.algbw / BASE.AlgBwAlignRccl - 1`，使用对齐后的算法带宽，不能直接与原始 Bandwidth 相除。
- 正值表示 NEW 指标更高；自定义 labels 仅改变显示名称。使用相同数据比较，非零基线对应的变化率应为 0%。
