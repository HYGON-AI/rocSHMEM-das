# rocprofv2 Trace采集与日志转换使用文档
> 配套仓库内已托管转换脚本，无需手动新建脚本文件。
> ⚠️ 说明：本文流程适用于 rocprofv2 文本格式输出模式；rocprofv2 亦支持原生导出 perfetto 格式 `.pftrace`（无需转换脚本，但当前 DTK 环境暂不可用），见文末进阶方案。

## 1. 基础说明
### 1.1 采集范围
开启 `--sys-trace` 可捕获：
- `hip_api_trace.txt`：HIP Runtime 上层主机API调用
- `hsa_api_trace.txt`：底层HSA驱动主机API调用
- `hcc_ops_trace.txt`：GPU Kernel 设备执行事件

### 1.2 工具区分
- rocprofv2：DTK新版追踪工具（本文适用）
- 旧版 rocprof(v1)：日志格式不兼容，不能共用这套流程

## 2. 环境依赖
转换脚本依赖 `Python3 + pandas + rocprofv2`
```bash
pip3 install pandas
```

## 3. 步骤1：rocprofv2 采样采集Trace日志
### 3.1 完整采样命令（文本输出模式）
> 提示：DTK 安装完成后 `rocprofv2` 路径 `/opt/dtk/rocprofiler/bin/rocprofv2`，后续 `rocprofv2` 代指 `/opt/dtk/rocprofiler/bin/rocprofv2`

#### 单进程采样
```bash
# 创建日志输出目录
mkdir -p ./prof

# rocprofv2 采样简单示例
rocprofv2 \
    --sys-trace \
    -d ./prof \
    ./your_executable [程序参数]
```

#### mpirun 多rank采样
```bash
rm -rf ./prof  # 先清理历史采集数据，避免混杂
mpirun --allow-run-as-root -np 4 -x PATH -x LD_LIBRARY_PATH \
bash -c '
    # 兼容OpenMPI / PMI两套rank环境变量
    if [[ -n "$OMPI_COMM_WORLD_RANK" ]]; then
        RANK="$OMPI_COMM_WORLD_RANK"
    else
        RANK="${PMI_RANK:-0}"
    fi

    OUTDIR="./prof/rank_${RANK}"
    mkdir -p "$OUTDIR"
    echo "PROF START rank=$RANK outdir=$OUTDIR"

    /opt/dtk/rocprofiler/bin/rocprofv2 \
        --sys-trace \
        -o "$OUTDIR/trace" \
        ./tests/functional_tests/rocshmem_functional_tests \
        -a 76 -w 1 -z 128 -s 1048576 -noverif
    echo "PROF FINISH rank=$RANK"
'
```

### 3.2 参数说明
```
--sys-trace                开启全套追踪：HIP + HSA + Kernel
-d,--output-directory      指定trace日志输出根目录
-o,--output-name           指定输出文件前缀（如 -o ./prof/rank_0/trace 生成 trace_*_*.txt）
```

### 3.3 采集完成目录结构
多rank并行运行自动生成rank子目录：
```
prof/
├── rank_0/
│   ├── trace_xxxx_hcc_ops_trace.txt
│   ├── trace_xxxx_hip_api_trace.txt
│   └── trace_xxxx_hsa_api_trace.txt
├── rank_1/
│   ├── trace_xxxx_hcc_ops_trace.txt
│   ├── trace_xxxx_hip_api_trace.txt
│   └── trace_xxxx_hsa_api_trace.txt
...
```

## 4. 步骤2：使用仓库内置脚本转换Trace
> 转换脚本已上传至代码仓库，直接使用，无需复制创建。

### 4.1 添加执行权限（首次使用执行一次）
> 以下命令默认在仓库根目录（`rocshmem/`）下执行；若已 `cd scripts/functional_tests/`，去掉前缀直接用 `./rocprof_trace_convert.sh`。
```bash
chmod +x ./scripts/functional_tests/rocprof_trace_convert.sh
```

### 4.2 运行转换脚本
传入采样生成的trace根目录（本例为`./prof`）
```bash
bash ./scripts/functional_tests/rocprof_trace_convert.sh ./prof
```

## 5. 转换输出结果
脚本在**执行命令时的当前工作目录（cwd）**下生成 `./convert_out` 目录，输出三类文件：
> 提示：若想输出到别处，可先 `cd` 到目标父目录，再用绝对路径调用脚本和指定 trace 目录。
```
convert_out/
├── rank0_trace.json          # Perfetto可视化时序文件
├── rank0_trace_raw.csv       # 全量事件原始明细
├── rank0_trace_summary.csv   # 函数耗时汇总（调用次数、总耗时、平均耗时）
├── rank1_trace.json
├── rank1_trace_raw.csv
├── rank1_trace_summary.csv
...
```

## 6. Trace可视化
1. 浏览器打开：https://ui.perfetto.dev/
2. `Open trace file` 载入 `rankX_trace.json`
3. 泳道分类含义：
- `hip_api`：HIP上层Runtime调用
- `hsa_api`：底层HSA驱动调用
- `kernel`：GPU Kernel执行任务

## 7. 常见问题
1. **报错 ModuleNotFoundError: No module named 'pandas'**
```bash
pip3 install pandas
```

2. **部分rank缺少hsa_api_trace.txt**
脚本仅打印警告，正常解析剩余日志，不会整体中断。

## 8. 进阶方案：rocprofv2 原生输出 Perfetto 文件
> ⚠️ 当前 DTK 环境缺少关键依赖包，此方案暂不可用。待依赖补齐后可作为首选方案，省去文本解析步骤。

直接导出 `.pftrace`，**不需要转换脚本**：
```bash
mkdir -p ./prof_pftrace
rocprofv2 \
    --sys-trace \
    --plugin perfetto/ctf \
    -d ./prof_pftrace \
    ./your_executable
```
生成 `trace.pftrace`，可直接导入 ui.perfetto.dev 查看。

