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
minmsgsize = 16

if len(sys.argv) <= 1:
    print("No input directory provided. Aborting")
    sys.exit()

files_in_dir = {}
files = 0
mode = "--Both"  # 默认 Both
out_file = "rocshmem_test" # 默认输出文件

for arg in sys.argv[1:]:
    if arg in ["--Lat", "--Bw", "--Both"]:
        mode = arg
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
    print("用法: python heatmap-parse-rocshmem-general.py [--Lat/--Bw/--Both] [-o=out_file] 目录1 目录2...")
    sys.exit()

unique="rocSHMEM_MI300_Thor2_Heatmap"

## cols => no. of consecutive runs
## rows => no. of msg sizes in the sweep -- 35 = 1B-16GB
cols, rows = 1, 27

x = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728, 268435456, 536870912, 1073741824]
x_str = [16, 32, 64, 128, 256, 512, '1KB', '2KB', '4KB', '8KB', '16KB', '32KB', '64KB', '128KB', '256KB', '512KB', '1MB', '2MB', '4MB', '8MB', '16MB', '32MB', '64MB', '128MB', '256MB', '512MB', '1GB']


@dataclass
class Measurement:
    volume: int
    msgsize: int
    msgcount: int
    avg_time: float
    avg_bw: float
    msg_rate: float

@dataclass
class Series:
    op: str
    ngpus: int
    nwgs: int
    nthreads: int
    data: List[Measurement]

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
    'align': 'center',
    'valign': 'vcenter',
    'bold': True,
    'bg_color': '#FFFF00',
    'text_wrap': True
})

now = datetime.now()
date_str = now.strftime("%Y-%m-%d")

for dir, file_names in files_in_dir.items():
    sheet_name = f"{dir}-{date_str}"[:31]
    worksheet = workbook.add_worksheet(sheet_name)
    worksheet.set_zoom(70)
    worksheet.set_column(0, 200, 14)  # 足够宽
    all_data = []

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

        with open(f"{filename}", 'r') as file1:
            for line in file1:
                line = line.strip()
                if not line:
                    continue
                if '#' in line or '[' in line or 'mpirun' in line:
                    continue

                parts = line.split()
                if len(parts) < 6:
                    continue

                valid = True
                for p in parts:
                    if not re.match(r'^-?\d+(\.\d+)?$', p):
                        valid = False
                        break
                if not valid:
                    continue

                volume1 = int(parts[0])
                size1 = int(parts[1])
                msgcount1 = int(parts[2])
                avg_time1 = float(parts[3])
                avg_bw1 = float(parts[4])
                msg_rate1 = float(parts[5])

                if volume1 < minmsgsize:
                    continue

                datapoint = Measurement(volume1, size1, msgcount1, avg_time1, avg_bw1, msg_rate1)
                this_series.data.append(datapoint)
        all_data.append(this_series)

    # sort all_data such that data of the same operation appear together
    # and num_gpus for the same operation are increasing
    all_data.sort(key = lambda item: (item.op, item.ngpus))

    prev_op = ""
    op_count = 1
    worksheet.write(1, 3, f"Data directory: {dir}", yellow_format)
    worksheet.write(1, 4, f"Lat:us\nBw:GB/s", yellow_format)
    dataset_count = 0
    for data_series in all_data:
        if prev_op != data_series.op:
            prev_op = data_series.op
            pad_top = 1 + 10*(op_count)
            pad_left = 2
            dataset_count = 0
            op_count = op_count + 1

            worksheet.write(pad_top, pad_left+1, f"{data_series.op}", cell_format)
            worksheet.write(pad_top+1, pad_left, f"num-gpus", cell_format)
            worksheet.write(pad_top+1, pad_left+1, f"num-wgs", cell_format)
            worksheet.write(pad_top+1, pad_left+2, f"num-threads", cell_format)

            for i in range(rows):
                if mode == "--Lat":
                    header = f"{x_str[i]}\nLat"
                elif mode == "--Bw":
                    header = f"{x_str[i]}\nBw"
                else:
                    header = f"{x_str[i]}\nLat/Bw"
                worksheet.write(pad_top+1, i+pad_left+3, header, cell_format)

        top_start = pad_top+2 + dataset_count
        worksheet.write(top_start, pad_left, f"{data_series.ngpus}", cell_format)
        worksheet.write(top_start, pad_left+1, f"{data_series.nwgs}", cell_format)
        worksheet.write(top_start, pad_left+2, f"{data_series.nthreads}", cell_format)

        for i in range(rows):
            msg_size = x[i]
            lat = None
            bw = None
            for pt in data_series.data:
                if pt.volume == msg_size:
                    lat = pt.avg_time
                    bw = pt.avg_bw
                    break

            # ====================== 最终显示格式 ======================
            if mode == "--Lat":
                val = f"{lat:.2f}" if lat is not None else ""
                worksheet.write(top_start, i+pad_left+3, val, wrap_format)
            elif mode == "--Bw":
                val = f"{bw:.2f}" if bw is not None else ""
                worksheet.write(top_start, i+pad_left+3, val, wrap_format)
            else:
                if lat is not None and bw is not None:
                    val = f"{lat:.2f}\n{bw:.2f}"
                elif lat is not None:
                    val = f"Lat:{lat:.2f}"
                elif bw is not None:
                    val = f"Bw:{bw:.2f}"
                else:
                    val = ""
                worksheet.write(top_start, i+pad_left+3, val, wrap_format)
            # ============================================================

        dataset_count = dataset_count + 1

workbook.close()
print(f"\n✅ 生成成功：{out_file}.xlsx")
print(f"✅ 模式：{mode}")
