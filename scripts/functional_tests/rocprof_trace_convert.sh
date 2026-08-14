#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

usage() {
    echo "Usage:"
    echo "  bash $0 <trace root directory>"
    echo
    echo "Example:"
    echo "  bash $0 ./prof"
    echo "Auto-parses hip_api_trace / hcc_ops_trace / hsa_api_trace"
    echo "Output: perfetto json, raw detail csv, summary statistics csv"
    exit 1
}

# Accept exactly 1 argument
if [ $# -ne 1 ]; then
    usage
fi
TRACE_ROOT="$1"
OUT_DIR="./convert_out"
mkdir -p "${OUT_DIR}"

if [ ! -d "${TRACE_ROOT}" ]; then
    echo "Error: directory does not exist -> ${TRACE_ROOT}"
    exit 1
fi

# Embedded Python parser code
PY_SCRIPT=$(cat <<'EOF'
import sys
import os
import json
import re
import csv
import pandas as pd

def extract_number(text):
    m = re.search(r"\((\d+)\)", text)
    return int(m.group(1)) if m else None

def extract_func_name(text):
    m = re.search(r"Function\((.+?)\)", text)
    return m.group(1) if m else None

def parse_hip_api_trace(filepath, rank):
    rows = []
    if not filepath or not os.path.isfile(filepath):
        return rows
    base_pid = 1000 + rank * 100
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            func_name = start_ns = end_ns = None
            for seg in parts:
                if seg.startswith("Function("):
                    func_name = extract_func_name(seg)
                elif seg.startswith("Start_Timestamp("):
                    start_ns = extract_number(seg)
                elif seg.startswith("End_Timestamp("):
                    end_ns = extract_number(seg)
            if func_name is None or start_ns is None or end_ns is None:
                continue
            start_us = start_ns / 1000.0
            dur_us = (end_ns - start_ns) / 1000.0
            rows.append({
                "rank": rank,
                "type": "hip_api",
                "name": func_name,
                "start_us": start_us,
                "dur_us": dur_us,
                "pid": base_pid,
                "tid": 1
            })
    return rows

def parse_hsa_api_trace(filepath, rank):
    rows = []
    if not filepath or not os.path.isfile(filepath):
        return rows
    base_pid = 1500 + rank * 100
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            func_name = start_ns = end_ns = None
            for seg in parts:
                if seg.startswith("Function("):
                    func_name = extract_func_name(seg)
                elif seg.startswith("Start_Timestamp("):
                    start_ns = extract_number(seg)
                elif seg.startswith("End_Timestamp("):
                    end_ns = extract_number(seg)
            if func_name is None or start_ns is None or end_ns is None:
                continue
            start_us = start_ns / 1000.0
            dur_us = (end_ns - start_ns) / 1000.0
            rows.append({
                "rank": rank,
                "type": "hsa_api",
                "name": func_name,
                "start_us": start_us,
                "dur_us": dur_us,
                "pid": base_pid,
                "tid": 1
            })
    return rows

def parse_hcc_ops_trace(filepath, rank):
    rows = []
    if not filepath or not os.path.isfile(filepath):
        return rows
    base_pid = 2000 + rank * 100
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            kernel_name = start_ns = end_ns = None
            gpu_id = 0
            for seg in parts:
                if seg.startswith("Kernel_Name("):
                    match = re.search(r"Kernel_Name\((.+?)\)", seg)
                    kernel_name = match.group(1) if match else None
                elif seg.startswith("Start_Timestamp("):
                    start_ns = extract_number(seg)
                elif seg.startswith("End_Timestamp("):
                    end_ns = extract_number(seg)
                elif seg.startswith("GPU_ID("):
                    val = extract_number(seg)
                    if val is not None:
                        gpu_id = val
            if kernel_name is None or start_ns is None or end_ns is None:
                continue
            start_us = start_ns / 1000.0
            dur_us = (end_ns - start_ns) / 1000.0
            pid = base_pid + gpu_id
            rows.append({
                "rank": rank,
                "type": "kernel",
                "name": kernel_name,
                "start_us": start_us,
                "dur_us": dur_us,
                "pid": pid,
                "tid": gpu_id
            })
    return rows

def write_json(rows, out_path):
    events = []
    for r in rows:
        events.append({
            "name": r["name"],
            "cat": r["type"],
            "ph": "X",
            "ts": r["start_us"],
            "dur": r["dur_us"],
            "pid": r["pid"],
            "tid": r["tid"]
        })
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(events, fp)

def write_raw_csv(all_rows, out_path):
    headers = ["rank","type","name","start_us","dur_us","pid","tid"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, headers)
        w.writeheader()
        w.writerows(all_rows)

def write_summary(all_rows, out_path):
    df = pd.DataFrame(all_rows)
    agg_df = df.groupby(["rank","type","name"])["dur_us"].agg(
        count="count",
        total_dur_us="sum",
        avg_dur_us="mean"
    ).reset_index()
    agg_df = agg_df.sort_values("total_dur_us", ascending=False)
    agg_df.to_csv(out_path, index=False, encoding="utf-8")

def main():
    if len(sys.argv) !=6:
        print("python args: hip_path ops_path hsa_path base_out rank")
        sys.exit(1)
    hip_path = sys.argv[1]
    ops_path = sys.argv[2]
    hsa_path = sys.argv[3]
    base_out = sys.argv[4]
    rank = int(sys.argv[5])

    r1 = parse_hip_api_trace(hip_path, rank)
    r2 = parse_hcc_ops_trace(ops_path, rank)
    r3 = parse_hsa_api_trace(hsa_path, rank)
    all_rows = r1 + r2 + r3
    print(f"HIP:{len(r1)}, Kernel:{len(r2)}, HSA:{len(r3)}, Total:{len(all_rows)}")

    json_path = base_out + "_trace.json"
    raw_csv_path = base_out + "_trace_raw.csv"
    summary_csv_path = base_out + "_trace_summary.csv"

    write_json(all_rows, json_path)
    write_raw_csv(all_rows, raw_csv_path)
    write_summary(all_rows, summary_csv_path)

if __name__ == "__main__":
    main()
EOF
)

# Iterate over rank directories
for rank_dir in "${TRACE_ROOT}"/rank_*; do
    [ -e "${rank_dir}" ] || continue
    rank_basename=$(basename "${rank_dir}")
    rank_id=${rank_basename#rank_}

    hip_files=("${rank_dir}/"*_hip_api_trace.txt)
    ops_files=("${rank_dir}/"*_hcc_ops_trace.txt)
    hsa_files=("${rank_dir}/"*_hsa_api_trace.txt)

    # Check required files
    if [ ! -f "${hip_files[0]}" ]; then
        echo "[SKIP] ${rank_basename} missing hip_api_trace.txt"
        continue
    fi
    if [ ! -f "${ops_files[0]}" ]; then
        echo "[SKIP] ${rank_basename} missing hcc_ops_trace.txt"
        continue
    fi
    if [ ! -f "${hsa_files[0]}" ]; then
        echo "[WARN] ${rank_basename} missing hsa_api_trace.txt"
        # If some ranks don't have hsa files, decide whether to continue
        hsa_path=""
    else
        hsa_path="${hsa_files[0]}"
    fi

    hip_path="${hip_files[0]}"
    ops_path="${ops_files[0]}"
    base_file="${OUT_DIR}/rank${rank_id}"

    echo "====================================="
    echo "Converting ${rank_basename}"
    python3 -c "${PY_SCRIPT}" "${hip_path}" "${ops_path}" "${hsa_path}" "${base_file}" "${rank_id}"

    echo "JSON(Perfetto): ${base_file}_trace.json"
    echo "Detail CSV:      ${base_file}_trace_raw.csv"
    echo "Summary CSV:     ${base_file}_trace_summary.csv"
done

echo -e "\n✅ All conversions complete, output directory: ${OUT_DIR}"
