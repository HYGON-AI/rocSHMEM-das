# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

import os
import sys
import re
#from matplotlib import pyplot as plt
import xlsxwriter
from datetime import datetime
from dataclasses import dataclass
from typing import List

## EDIT THESE VALUES
bkc_version = "BKC 00.25.11.02 (38.4 GT/s XGMI)"
ifwi_version = "IFWI 00.940.956978"
amdgpu_version = "amdgpu 6.14.14-2193512"
os_kernel = "CentOS Kernel 6.9.0-0_fbk10_brcmrdma13_141_g9b20106afb70"
nic_driver = "Broadcom driver=6.9.0-0_fbk10_brcmrdma13_141_g9 firmware=232.0.213.0/pkg 232.1.190.0"
###

rocshmem_version = ""
rocm_version = ""
hip_version = ""
minmsgsize = 8

if len(sys.argv) <= 1:
    print("No input directory provided. Aborting")
    sys.exit()

files_in_dir = {}
files = 0
mode = "--both"  # 默认 both
out_file = "rocshmem_test" # 默认输出文件
show_mode = "--v" # 默认按volume显示, --s按msgsize显示

for arg in sys.argv[1:]:
    if arg in ["--lat", "--bw", "--both"]:
        mode = arg
        continue
    elif arg in ["--s", "--v"]:
        show_mode = arg
        continue
    elif arg.startswith("-o="):
        out_file = arg.split("=")[1]
        continue
    file_names = []
    if not os.path.isdir (arg):
        continue
    for entry in os.listdir(arg):
        full_path = os.path.join(arg, entry)
        if os.path.isfile(full_path):
            file_names.append(entry)
    files_in_dir[arg] = file_names
    files = files + len(file_names)

if not files_in_dir:
    print("用法: python heatmap-parse-rocshmem-general.py [--lat/--bw/--both] [--s/--v] [-o=out_file] 目录1 目录2...\n" \
         "--lat: 只显示延迟,可选\n" \
         "--bw: 只显示带宽,可选\n" \
         "--both: 同时显示延迟和带宽(默认),可选\n" \
         "--s: 按消息大小显示(默认),可选\n" \
         "--v: 按Rank级消息总大小显示(volume=并发消息数*size),可选\n" \
         "-o=out_file: 指定输出Excel文件名(默认rocshmem_test),可选\n")
    sys.exit()

unique="rocSHMEM_MI300_Thor2_Heatmap"

## cols => no. of consecutive runs
## rows => no. of msg sizes in the sweep -- 35 = 1B-16GB
cols, rows = 1, 27
op_interval = 10

x = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824]
x_str = [8,16, 32, 64, 128, 256, 512, '1KB', '2KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB']

max_failed_chars = 500  # 失败时总打印字符数

@dataclass
class Measurement:
    volume: int
    msgsize: int
    msgcount: int
    avg_time: float
    avg_bw: float
    align_rccl_algbw: float
    align_rccl_busbw: float
    msg_rate: float

@dataclass
class Series:
    op: str
    ngpus: int
    nwgs: int
    nthreads: int
    data: List[Measurement]
    has_error: bool = False  # 新增：标记是否出错
    error_messages: List[str] = None  # 新增：存储错误信息
    
    def __post_init__(self):
        if self.error_messages is None:
            self.error_messages = []

workbook = xlsxwriter.Workbook(f"{out_file}.xlsx")
workbook.set_properties({'company':  'AMD'})

# 基础格式
cell_format = workbook.add_format({
    'align': 'center',
    'valign': 'vcenter',
    'bold': True,
    'text_wrap': True  # 让表头也自动换行！
})
num_format = workbook.add_format({
    'align': 'center',
    'valign': 'vcenter',
    'num_format': '0.00'
})
wrap_format = workbook.add_format({
    'align': 'center',
    'valign': 'vcenter',
    'num_format': '0.00',
    'text_wrap': True
})
# 黄色背景格式
yellow_format = workbook.add_format({
    'align': 'left',
    'valign': 'vcenter',
    'bold': True,
    'bg_color': '#FFFF00',
    'text_wrap': True
})
# 红色失败格式（新增）
failed_format = workbook.add_format({
    'align': 'left',
    'valign': 'vcenter',
    'bold': True,
    'font_color': 'red',
    'text_wrap': True
})

# 收集所有环境信息，最后统一写入
env_data_list = []
env_sheet_name = "EnvInfo"

for dir, file_names in files_in_dir.items():
    sheet_name = f"{dir}"[:31]
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.set_zoom(70)
    worksheet.set_column(0, 200, 14)  # 足够宽
    all_data = []
    
    # 收集环境信息（暂不写入）
    env_log_path = os.path.join(dir, "env_info.log")
    if os.path.isfile(env_log_path):
        env_info_content = []
        with open(env_log_path, 'r', encoding='utf-8') as f:
            for line in f:
                env_info_content.append(line.strip('\n'))
        env_data_list.append((dir, env_info_content))

    for file in file_names:
        mpirun_cmd = ""
        filename=f"{dir}/{file}"

        match = re.search(r'^(.+?)_n(\d+)_w(\d+)_z(\d+)', file)
        if not match:
            print(f"⏭️ Skip: {file}")
            continue

        op      = match.group(1)
        ngpu    = int(match.group(2))
        nwgs    = int(match.group(3))
        nthreads= int(match.group(4))

        this_series = Series(op, ngpu, nwgs, nthreads, [])

        # ===================== 先扫描是否有 error =====================
        has_error = False
        error_messages = []
        total_chars = 0  # 累计字符数
        
        try:
            with open(filename, 'r') as f_check:
                check_line = f_check.readline()
                while check_line:
                    lower_line = check_line.lower()
                    # 检测常见错误关键词
                    if 'error' in lower_line or 'failed' in lower_line or 'segfault' in lower_line:
                        has_error = True
                        # 将当前触发行与后续内容拼接，然后截取最多max_failed_chars个字符
                        remaining_content = check_line + f_check.read()
                        error_info = remaining_content[:max_failed_chars]
                        error_messages.append(error_info)
                        total_chars = len(error_info)
                        break
                    check_line = f_check.readline()
        except Exception as e:
            has_error = True
            error_messages.append(f"文件读取错误: {str(e)[:50]}")


        this_series.has_error = has_error
        this_series.error_messages = error_messages
        
        if has_error:
            print(f"❌ 文件包含错误，标记为 Failed: {filename}")
            all_data.append(this_series)
            continue
        # ====================================================================

        with open(f"{filename}", 'r') as file1:
            for line in file1:
                line = line.strip()
                if not line:
                    continue
                if '#' in line or '[' in line or 'mpirun' in line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                valid = True
                for p in parts:
                    if not re.match(r'^-?\d+(\.\d+)?$', p):
                        valid = False
                        break
                if not valid:
                    continue

                if len(parts) == 5:
                    volume1 = 0
                    size1 = int(parts[0])
                    msgcount1 = int(parts[1])
                    avg_time1 = float(parts[2])
                    avg_bw1 = float(parts[3])
                    msg_rate1 = float(parts[4])
                    align_rccl_algbw1 = 0
                    align_rccl_busbw1 = 0
                elif len(parts) == 6:
                    volume1 = int(parts[0])
                    size1 = int(parts[1])
                    msgcount1 = int(parts[2])
                    avg_time1 = float(parts[3])
                    avg_bw1 = float(parts[4])
                    msg_rate1 = float(parts[5])
                    align_rccl_algbw1 = 0
                    align_rccl_busbw1 = 0
                elif len(parts) == 8:
                    volume1 = int(parts[0])
                    size1 = int(parts[1])
                    msgcount1 = int(parts[2])
                    avg_time1 = float(parts[3])
                    avg_bw1 = float(parts[4])
                    align_rccl_algbw1 = float(parts[5])
                    align_rccl_busbw1 = float(parts[6])
                    msg_rate1 = float(parts[7])

                if volume1 < minmsgsize:
                    continue

                datapoint = Measurement(volume1, size1, msgcount1, avg_time1, avg_bw1, align_rccl_algbw1, align_rccl_busbw1, msg_rate1)
                this_series.data.append(datapoint)
        all_data.append(this_series)

    # sort all_data such that data of the same operation appear together
    # and num_gpus for the same operation are increasing
    all_data.sort(key = lambda item: (item.op, item.ngpus))

    prev_op = ""
    op_count = 1
    show_mode_info = f" Show Mode: {'Message Size' if show_mode=='--s' else 'Volume Size'}"
    worksheet.merge_range(1, 2, 1, 10, f"Data directory: {dir}", yellow_format)
    worksheet.write(1, 12, f"Lat: us\nBw: GB/s", yellow_format)
    worksheet.merge_range(1, 14, 1, 15, f"{show_mode_info}", yellow_format)
    worksheet.merge_range(3, 2, 3, 15, f"Note: For OnStream operations, the actual value of w is 1. The actual z varies with size. When it is greater than 256, use 256.", yellow_format)
    # 在目录sheet第5行第3列添加跳转到对应环境信息工作表的链接
    worksheet.write_url(4, 2, f"internal:'{env_sheet_name}'!A1", string="📎 环境信息")

    dataset_count = 0
    pre_pad_top = 0
    dataset_count_one_op = 0
    pad_left = 2
    pad_top = 0
    for data_series in all_data:
        if prev_op != data_series.op:
            # ===================== 修改并扩大前一个op的行 =====================
            if dataset_count_one_op > op_interval:
                worksheet.merge_range(pre_pad_top, pad_left - 1, pre_pad_top + dataset_count_one_op, pad_left - 1, f"{prev_op}", cell_format)
                pad_top = 1 + pre_pad_top + dataset_count_one_op
            else:
                pad_top = 1 + pre_pad_top + op_interval

            prev_op = data_series.op
            pre_pad_top = pad_top
            pad_left = 2
            dataset_count = 0
            dataset_count_one_op = 0
            op_count = op_count + 1

            worksheet.merge_range(pad_top, pad_left - 1, pad_top + op_interval, pad_left - 1, f"{data_series.op}", cell_format)
            worksheet.write(pad_top, pad_left, f"num-gpus", cell_format)
            worksheet.write(pad_top, pad_left+1, f"num-wgs", cell_format)
            worksheet.write(pad_top, pad_left+2, f"num-threads", cell_format)

            for i in range(rows):
                if mode == "--lat":
                    header = f"{x_str[i]}\nLat"
                elif mode == "--bw":
                    header = f"{x_str[i]}\nBw/AlgBw/BusBw"
                else:
                    header = f"{x_str[i]}\nBw/AlgBw/BusBw/Lat"
                worksheet.write(pad_top, i+pad_left+3, header, cell_format)

        top_start = pad_top + 1 + dataset_count
        worksheet.write(top_start, pad_left, f"{data_series.ngpus}", cell_format)
        worksheet.write(top_start, pad_left+1, f"{data_series.nwgs}", cell_format)
        worksheet.write(top_start, pad_left+2, f"{data_series.nthreads}", cell_format)

        # ===================== 出错显示 Failed + 关键错误日志（合并单元格）=====================
        if data_series.has_error:
            # 获取第一条错误信息
            error_info = ""
            if data_series.error_messages:
                error_info = data_series.error_messages[0]
                # 限制长度在max_failed_chars字符以内
                if len(error_info) > max_failed_chars:
                    error_info_len = max_failed_chars-3
                    error_info = error_info[:error_info_len] + "..."
            
            # 合并从 pad_left+3 到 pad_left+3+rows-1 的所有列，写入错误信息
            error_text = f"Failed：{error_info}" if error_info else "Failed"
            worksheet.merge_range(top_start, pad_left+3, top_start, pad_left+3+rows-1, error_text, failed_format)
            dataset_count += 1
            dataset_count_one_op += 1
            continue
        # ======================================================================

        for i in range(rows):
            msg_size = x[i]
            lat = None
            bw = None
            align_rccl_algbw = None
            align_rccl_busbw = None
            for pt in data_series.data:
                if (show_mode == "--s" and pt.msgsize == msg_size) or (show_mode == "--v" and pt.volume == msg_size):
                    lat = pt.avg_time
                    bw = pt.avg_bw
                    align_rccl_algbw = pt.align_rccl_algbw
                    align_rccl_busbw = pt.align_rccl_busbw
                    break

            # ====================== 最终显示格式 ======================
            if mode == "--lat":
                val = f"{lat:.2f}" if lat is not None else ""
                worksheet.write(top_start, i+pad_left+3, val, wrap_format)
            elif mode == "--bw":
                if bw is not None:
                    val = f"{bw:.2f}\n{align_rccl_algbw:.2f}\n{align_rccl_busbw:.2f}"
                else:
                    val = ""
                worksheet.write(top_start, i+pad_left+3, val, wrap_format)
            else:
                if lat is not None and bw is not None:
                    val = f"{bw:.2f}\n{align_rccl_algbw:.2f}\n{align_rccl_busbw:.2f}\n{lat:.2f}"
                elif lat is not None:
                    val = f"Lat:{lat:.2f}"
                elif bw is not None:
                    val = f"Bw:{bw:.2f}"
                else:
                    val = ""
                worksheet.write(top_start, i+pad_left+3, val, wrap_format)
            # ============================================================

        dataset_count += 1
        dataset_count_one_op += 1

# 最后创建 EnvInfo 工作表（放在最后）
if env_data_list:
    env_sheet_name = "EnvInfo"
    env_sheet = workbook.add_worksheet(env_sheet_name)
    global_env_row = 0
    
    for dir, env_content in env_data_list:
        if global_env_row > 0:
            global_env_row += 5
        env_sheet.merge_range(global_env_row, 0, global_env_row, 13, f"********************************************** {dir} **********************************************", yellow_format)
        global_env_row += 1
        for line in env_content:
            env_sheet.write_string(global_env_row, 0, line)
            global_env_row += 1

workbook.close()
print(f"\n✅ 生成成功：{out_file}.xlsx")
print(f"✅ 输出模式：{'Message Size' if show_mode=='--s' else 'Volume Size'}, {'Latency' if mode=='--lat' else 'Bandwidth' if mode=='--bw' else 'Latency and Bandwidth'}")
