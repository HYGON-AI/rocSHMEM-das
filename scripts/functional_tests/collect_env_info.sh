#!/bin/bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

EnvCommand() {
  printf '\n$'; printf ' %q' "$@"; printf '\n'
  local input_code=""
  if [[ "$1" == python3 && "${2:-}" == - ]]; then
    input_code=$(cat)
    printf '[stdin Python source]\n%s\n[end stdin Python source]\n' "$input_code"
  fi
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "[unavailable] command not found: $1"
    return 0
  fi
  local status=0
  if command -v timeout >/dev/null 2>&1; then
    if [[ "$1" == python3 && "${2:-}" == - ]]; then
      LC_ALL=C timeout 15s "$@" <<< "$input_code" 2>&1 || status=$?
    else
      LC_ALL=C timeout 15s "$@" 2>&1 || status=$?
    fi
  else
    echo "[unavailable] timeout missing; command skipped to avoid blocking tests"
    return 0
  fi
  [[ $status -eq 0 ]] || echo "[incomplete] exit status $status (124 means timeout)"
  return 0
}

# Keep command headers and failures even when reducing verbose successful output.
EnvCommandFiltered() {
  local pattern=$1
  shift
  EnvCommand "$@" | awk -v pattern="$pattern" '
    /^\$/ { print; print "[output filter] " pattern; next }
    /^\[unavailable\]/ || /^\[incomplete\]/ || $0 ~ pattern { print }
  '
}

EnvFiles() {
  local file
  for file in "$@"; do
    [[ -e "$file" ]] || continue
    printf '\n%s: ' "$file"
    cat "$file" 2>&1 || echo "[unavailable] cannot read"
  done
}

EnvPciDevice() {
  local device
  device=$(readlink -f "$1") || return 0
  echo "PCI path: $device"
  if [[ -n "${env_pci_seen[$device]:-}" ]]; then
    echo "PCI details: already recorded above for ${device##*/}"
    return 0
  fi
  env_pci_seen[$device]=1
  EnvFiles "$device/numa_node" "$device/local_cpulist" \
    "$device/current_link_speed" "$device/current_link_width" \
    "$device/max_link_speed" "$device/max_link_width"
  if [[ -L "$device/iommu_group" ]]; then
    echo "IOMMU group: $(readlink -f "$device/iommu_group")"
    EnvFiles "$device/iommu_group/type"
  else
    echo "IOMMU group: unavailable (does not establish IOMMU state)"
  fi
  EnvCommand lspci -D -s "${device##*/}" -vv

}

PrintfNetWorkInfo() {
  local nic_path nic ib_path port gid gid_value rdma_driver
  local has_dcb=0
  if command -v dcb >/dev/null 2>&1; then
    has_dcb=1
  fi
  if command -v ip >/dev/null 2>&1; then
    EnvCommand ip -details address show
  elif command -v ifconfig >/dev/null 2>&1; then
    EnvCommand ifconfig -a
  else
    EnvCommand ip -details address show
  fi
  EnvFiles /proc/net/route
  for nic_path in /sys/class/net/*; do
    [[ -e "$nic_path" ]] || continue
    nic=${nic_path##*/}
    [[ "$nic" == lo ]] && continue
    printf '\n--- Network interface: %s ---\n' "$nic"
    EnvFiles "$nic_path/operstate" "$nic_path/carrier" "$nic_path/mtu" \
      "$nic_path/speed" "$nic_path/duplex" "$nic_path/address"
    if [[ -e "$nic_path/device" ]]; then
      EnvPciDevice "$nic_path/device"
      EnvCommand ethtool -i "$nic"
      EnvCommand ethtool "$nic"
      EnvCommand ethtool -a "$nic"
      if [[ -d "$nic_path/device/infiniband" ]]; then
        if command -v mlnx_qos >/dev/null 2>&1 &&
           [[ $(readlink "$nic_path/device/driver") == */mlx5_core ]]; then
          EnvCommand mlnx_qos -i "$nic"
        elif [[ "$has_dcb" == 0 ]]; then
          EnvCommand dcb pfc show dev "$nic"
        fi
        if [[ "$has_dcb" == 1 ]]; then
          EnvCommand dcb pfc show dev "$nic"
        fi
      fi
    fi
  done
  EnvFiles /proc/net/bonding/*
  for ib_path in /sys/class/infiniband/*; do
    [[ -e "$ib_path" ]] || continue
    printf '\n--- RDMA device: %s ---\n' "${ib_path##*/}"
    EnvFiles "$ib_path/fw_ver" "$ib_path/node_guid" "$ib_path/node_type" "$ib_path/hw_rev" "$ib_path/board_id"
    rdma_driver=$(basename "$(readlink -f "$ib_path/device/driver")")
    echo "RDMA driver: $rdma_driver"
    EnvFiles "/sys/module/$rdma_driver/version"
    EnvPciDevice "$ib_path/device"
    for port in "$ib_path"/ports/*; do
      [[ -d "$port" ]] || continue
      EnvFiles "$port/state" "$port/phys_state" "$port/rate" \
        "$port/link_layer" "$port/active_mtu"
      for gid in "$port"/gids/*; do
        [[ -e "$gid" ]] || continue
        IFS= read -r gid_value < "$gid" 2>/dev/null || { echo "[unavailable] cannot read $gid"; continue; }
        # Empty GID slots can number in the hundreds; retain only assigned slots.
        [[ -n "${gid_value//[0:]/}" ]] || continue
        EnvFiles "$gid" "$port/gid_attrs/types/${gid##*/}" \
          "$port/gid_attrs/ndevs/${gid##*/}"
      done
    done
  done
  EnvCommand ibv_devinfo
}

PrintHardwareEnvInfo() {
  local card
  local -A env_pci_seen=()
  echo "=== System / CPU / memory / firmware ==="
  EnvCommand hostname
  EnvCommand uname -a
  if [[ -f /.dockerenv || -f /run/.containerenv ]]; then
    echo "Execution environment: container"
  else
    echo "Execution environment: unknown"
  fi
  printf 'DTK path: %s\n' "${ROCM_PATH:-unknown}"
  EnvFiles /etc/os-release /proc/cmdline \
    /sys/class/dmi/id/bios_vendor /sys/class/dmi/id/bios_version \
    /sys/class/dmi/id/bios_date /sys/class/dmi/id/product_name
  echo "Memory summary (kB):"
  awk '/^(MemTotal|MemAvailable|SwapTotal|SwapFree|HugePages_Total|HugePages_Free|Hugepagesize):/' /proc/meminfo 2>&1
  EnvCommand lscpu
  EnvCommand lscpu -p=CPU,NODE,SOCKET,CORE
  EnvCommand numactl --hardware
  echo "=== DTK / GPU ==="
  EnvFiles /opt/dtk/.dtk_version
  EnvCommand hipcc --version
  EnvCommand rocminfo
  local smi=hy-smi
  if ! command -v "$smi" >/dev/null 2>&1 && [[ -x /opt/hyhal/bin/hy-smi ]]; then
    smi=/opt/hyhal/bin/hy-smi
  fi
  EnvCommand "$smi"
  EnvCommand "$smi" --showid --showproductname --showdriverversion --showvbios
  EnvCommand "$smi" --showclocks --showbus
  EnvCommand "$smi" --showtopo
  if [[ -c /dev/mkfd ]]; then
    EnvCommand python3 - oam-mkfd <<'PY_ENV_OAM'
import ctypes
import os
import platform
import sys
from pathlib import Path

# The same read-only register query used by basic-perf-probe.
class RegArgs(ctypes.Structure):
    _fields_ = [
        ("gpu_id", ctypes.c_uint32),
        ("read", ctypes.c_bool),
        ("pm_pg_lock", ctypes.c_bool),
        ("use_bank", ctypes.c_bool),
        ("use_ring", ctypes.c_bool),
    ] + [(name, ctypes.c_uint32) for name in (
        "se_bank", "sh_bank", "instance_bank", "me", "pipe", "queue",
        "vmid", "reg", "value",
    )]


def query(devices):
    """Run only in the supervised child; retain successes on partial failure."""
    import fcntl

    if (sys.platform != "linux" or platform.machine().lower() not in
            ("x86_64", "amd64") or ctypes.sizeof(RegArgs) != 44):
        raise ValueError("当前平台未适配 mkfd OAM ioctl")
    request = (3 << 30) | (44 << 16) | (ord("M") << 8) | 0x18
    records = []
    fd = os.open("/dev/mkfd", os.O_RDWR | os.O_CLOEXEC)
    try:
        for device in devices:
            record = dict(device, status="ERROR", oam_id=None)
            try:
                args = RegArgs()
                args.gpu_id = device["gpu_id"]
                args.read = True
                args.reg = 0x5A08C
                buf = bytearray(bytes(args))
                fcntl.ioctl(fd, request, buf, True)
                record.update(status="OK", oam_id=RegArgs.from_buffer_copy(buf).value)
            except OSError as exc:
                record.update(error=str(exc), errno=exc.errno)
            records.append(record)
    finally:
        os.close(fd)
    return records

devices = []
for node in Path('/sys/class/kfd/kfd/topology/nodes').glob('*'):
    try:
        gpu_id = int((node / 'gpu_id').read_text())
        if not 0 < gpu_id <= 0xffffffff:
            continue
        props = dict(line.split(None, 1) for line in
                     (node / 'properties').read_text().splitlines()
                     if len(line.split(None, 1)) == 2)
        location = int(props['location_id'])
        domain = int(props.get('domain', '0'))
        bdf = f'{domain:04x}:{location >> 8 & 255:02x}:{location >> 3 & 31:02x}.{location & 7}'
        devices.append(dict(gpu_id=gpu_id, bdf=bdf))
    except (OSError, ValueError, KeyError):
        continue
devices = [d for d in devices if sum(
    d['gpu_id'] == other['gpu_id'] or d['bdf'] == other['bdf']
    for other in devices) == 1]
if not devices:
    sys.exit('No unambiguous KFD GPU mapping')
try:
    for record in query(devices):
        if record['status'] == 'OK':
            print('OAM ID', record['bdf'], record['oam_id'])
        else:
            print('[incomplete] OAM ID', record['bdf'], record['error'])
except (OSError, ValueError, ImportError) as exc:
    sys.exit(str(exc))
PY_ENV_OAM
  fi
  for card in /sys/class/drm/card[0-9]*; do
    [[ "${card##*/}" =~ ^card[0-9]+$ && -e "$card/device" ]] || continue
    echo "--- GPU ${card##*/} ---"
    EnvPciDevice "$card/device"
  done
  echo "=== PCIe tree / ACS evidence / IOMMU / peer-memory modules ==="
  EnvCommand lspci -D -tv
  EnvCommandFiltered '^[[:xdigit:]]{4}:|Access Control Services|ACSCap:|ACSCtl:|access denied' lspci -D -vvv
  EnvCommandFiltered '^(Module|amdgpu|hycu|hydcu|hygon|ib_|rdma_|mlx|bnxt|ionic|shca|hns|nvidia_peermem)' lsmod
  EnvFiles /sys/module/amdgpu/parameters/xdp_size \
    /sys/module/hycu/parameters/xdp_size
  echo "Missing XDP parameters or IOMMU groups mean unknown, not disabled."
  echo "=== Network / RDMA / PFC / GID / bonding ==="
  PrintfNetWorkInfo
  echo "=== MPI / UCX ==="
  EnvCommand mpirun --version
  EnvCommand ucx_info -v
}

# Embedded standard-library renderer keeps installed drivers self-contained.
WriteEnvHtml() {
  if ! command -v python3 >/dev/null 2>&1; then
    echo "WARNING: python3 unavailable; HTML skipped, raw environment log retained" >&2
    return 0
  fi
  python3 - "$1" "$2" <<'PY_ENV_HTML'
import html
import re
import shlex
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
raw = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', source.read_text(encoding='utf-8', errors='replace'))
esc = lambda value: html.escape(str(value))
UNKNOWN = '未取得'
# Parse the recorded snapshot only: regenerating HTML never queries a different node.
sections, commands, files = [], [], {}
section, command, output = '采集记录', None, []
def flush():
    if command is not None:
        commands.append((section, command, '\n'.join(output)))
active_file = None
for line in raw.splitlines():
    match = re.match(r'^={3,}\s+(.+?)\s+={3,}$', line)
    if match and (line.startswith('=== ') or 'Build Info and Env Vars' in line):
        flush()
        section, command, output = match.group(1), None, []
        active_file = None
        continue
    if line.startswith(('Execution environment:', 'DTK path:', '--- GPU ', '--- Network interface: ', '--- RDMA device: ')):
        flush()
        command, output, active_file = None, [], None
        continue
    if line.startswith('$ '):
        flush()
        command, output = line[2:], []
        active_file = None
        continue
    output.append(line)
    match = re.match(r'^(/(?:sys|proc|etc)/.+?):[ \t]+(.*)$', line)
    if match:
        if command is not None:
            output.pop()
            flush()
            command, output = None, []
        active_file = match.group(1)
        files[active_file] = match.group(2).strip()
    elif active_file and (active_file.startswith('/proc/net/bonding/') or active_file in ('/etc/os-release', '/proc/net/route')) and line and not line.startswith(('--- ', 'PCI path:', 'IOMMU group:', '===', 'Memory summary')):
        files[active_file] += '\n' + line
    else:
        active_file = None
flush()

def query(name, option=None):
    found = []
    for _, cmd, body in commands:
        try:
            parts = shlex.split(cmd)
        except ValueError:
            continue
        if parts and Path(parts[0]).name == name and (option is None or option in parts):
            found.append(body)
    return '\n'.join(found)

def field(text, pattern):
    match = re.search(pattern, text, re.M | re.I)
    return match.group(1).strip() if match else UNKNOWN

def table(headers, rows):
    rows = list(rows)
    if not rows:
        return '<p class="muted">未取得可展示记录，详见原始输出。</p>'
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+esc(x)+'</th>' for x in headers) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+esc(x if x not in (None, '') else UNKNOWN)+'</td>' for x in row)+'</tr>' for row in rows) + '</tbody></table></div>'

def evidence(predicate, wrap=True):
    items = [(cmd, body) for sec, cmd, body in commands if predicate(sec, cmd)]
    if not items:
        return ''
    labels={'uname':'内核信息','lscpu':'CPU / NUMA 信息','hipcc':'hipcc 编译器版本','rocminfo':'HSA 运行时与设备能力','lspci':'PCI 设备详情','hy-smi':'GPU 信息','hostname':'节点名称','numactl':'NUMA 信息'}
    labels.update({'lspci -D -vvv':'ACS 能力与控制位查询','lspci -D -tv':'PCIe 拓扑树','python3 - oam-mkfd':'OAM ID 查询'})
    displayed_items = []
    for cmd, body in items:
        label = labels.get(cmd, labels.get(Path(shlex.split(cmd)[0]).name, '设备与配置查询'))
        if body.startswith('[output filter] '):
            pattern, separator, body = body[len('[output filter] '):].partition('\n')
            cmd += ' 2>&1 | grep -E ' + shlex.quote(pattern)
            labels[cmd] = label
        if body.startswith('[stdin Python source]\n'):
            code, separator, result = body[len('[stdin Python source]\n'):].partition('\n[end stdin Python source]\n')
            if separator:
                labels[cmd + " <<'PY_ENV_OAM'\n" + code + "\nPY_ENV_OAM"] = labels.get(cmd, '设备与配置查询')
                cmd += " <<'PY_ENV_OAM'\n" + code + "\nPY_ENV_OAM"
                body = result
        displayed_items.append((cmd, body))
    items = displayed_items
    content = ''.join('<div class="query"><p class="muted">'+esc(labels.get(cmd,labels.get(Path(shlex.split(cmd)[0]).name,'设备与配置查询')))+'</p><button type="button" class="copy">复制查询命令</button><pre>'+esc(cmd)+'</pre><details><summary>原始输出</summary><pre>'+esc(body)+'</pre></details></div>' for cmd, body in items)
    return '<details><summary>查询命令与原始输出</summary>'+content+'</details>' if wrap else content

panels, nav = [], []
def panel(key, label, body, child=False):
    if key in ('overview', 'basic'):
        nav.append('<strong>' + ('报告概览' if key == 'overview' else '环境与设备') + '</strong>')
    style = ' class="nav-child"' if child else ''
    nav.append('<a'+style+' href="#'+key+'">'+label+'</a>')
    heading = 'h3' if child else 'h2'
    if child:
        panels[-1] = panels[-1][:-len('</section>')] + '<div class="report-child" id="'+key+'"><h3>'+label+'</h3>'+body+'</div></section>'
        return
    panels.append('<section id="'+key+'"><'+heading+'>'+label+'</'+heading+'>'+body+'</section>')

host = query('hostname').strip().splitlines()
host = host[0] if host and not host[0].startswith('[') else UNKNOWN
stamp = field(raw, r'^Environment info: (.+)$')
issues = []
context = '环境采集'
for line in raw.splitlines():
    if line.startswith('$ '):
        context = line[2:]
    elif re.match(r'^/(?:sys|proc|etc)/.+?:[ \t]',line):
        context = line.rsplit(': ',1)[0]
    if line.startswith(('[unavailable]', '[incomplete]')):
        if context.split()[:3] != ['dcb', 'app', 'show']:
            issues.append((context,line))
panel('overview', '采集概览', '')
panel('issues', '需要关注的事项', '')
cpu = query('lscpu')
os_name = field(raw, r'^PRETTY_NAME="?([^"\n]+)')
# /etc/os-release is logged with its first line after the file label.
if os_name == UNKNOWN:
    os_name = field(raw, r'PRETTY_NAME="([^"\n]+)')
def compact_version(text, pattern):
    value = field(text, pattern)
    return value if value != UNKNOWN else UNKNOWN
hip = query('hipcc')
compiler = '; '.join(value for value in (
    ('DCC ' + field(hip,r'^dcc version:\s*(\S+)')) if field(hip,r'^dcc version:\s*(\S+)') != UNKNOWN else '',
    ('Clang ' + field(hip,r'^clang version\s+(\S+)')) if field(hip,r'^clang version\s+(\S+)') != UNKNOWN else '') if value) or UNKNOWN
cpu_mapping = query('lscpu','-p=CPU,NODE,SOCKET,CORE')
cores=set()
for line in cpu_mapping.splitlines():
    parts=line.split(',')
    if len(parts)==4 and all(part.isdigit() for part in parts):
        cores.add((parts[2],parts[3]))
physical = str(len(cores)) if cores else UNKNOWN
# Older snapshots lack the per-CPU mapping. Avoid inventing a physical count.
topology = 'Socket: {0}; 物理核心: {1}; 逻辑 CPU: {2}; NUMA: {3}'.format(
    field(cpu,r'^Socket\(s\):\s*(.+)$'),physical,
    field(cpu,r'^CPU\(s\):\s*(.+)$'),field(cpu,r'^NUMA node\(s\):\s*(.+)$'))
roc=query('rocminfo')
architectures=sorted(set(re.findall(r'^\s*Name:\s*(gfx[0-9a-z]+)\s*$',roc,re.M)))
gpu_agents=len(re.findall(r'^\s*Device Type:\s*(?:GPU|HCU)\s*$',roc,re.M))
roc_summary='GPU agent {0} 个; {1}; HSA Runtime {2}'.format(gpu_agents if gpu_agents else UNKNOWN,', '.join(architectures) or UNKNOWN,field(roc,r'^Runtime Version:\s*(\S+)'))
root=field(raw,r'^DTK path:\s*(.+)$')
if root in (UNKNOWN,'unknown'):
    root=field(raw,r'^ROCM_PATH=(.+)$')
if root==UNKNOWN:
    install=field(hip,r'^InstalledDir:\s*(.+)$')
    if install.endswith('/llvm/bin'):
        root=install[:-len('/llvm/bin')]
    elif install.endswith('/bin'):
        root=install[:-len('/bin')]
cmdline=files.get('/proc/cmdline','')
iommu_params=' '.join(word for word in cmdline.split() if re.match(r'(?:iommu|amd_iommu|intel_iommu|iommu\.[\w]+)=',word))
iommu_summary=iommu_params or ('未发现 IOMMU 启动参数' if cmdline else UNKNOWN)
domains=[value for path,value in files.items() if path.endswith('/iommu_group/type')]
domain_summary=('已读取 {0} 个设备: {1}'.format(len(domains),', '.join(sorted(set(domains))))) if domains else '未取得设备默认域类型'
acs_output = next((body for _, cmd, body in commands if cmd == 'lspci -D -vvv'), '')
acs_blocks = re.split(r'(?m)(?=^[0-9a-fA-F]{4}:|^[ \t]+Capabilities:[^\n]*Access Control Services)', acs_output)
acs_rows=[]
for body in acs_blocks:
    control=field(body,r'^\s*ACSCtl:\s*(.+)$')
    if control!=UNKNOWN:
        acs_rows.append((field(body,r'^([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])'),control))
acs_enabled=any(re.search(r'\w+\+',control) for _,control in acs_rows)
acs_summary=('部分控制位开启' if acs_enabled else '已读取的控制位均未开启') if acs_rows else '未取得可读取的 ACS 控制信息'
xdp=next(iter(files.get('/sys/module/hycu/parameters/xdp_size',UNKNOWN).splitlines()),UNKNOWN)
xdp_state=('ON' if int(xdp)>0 else 'OFF') if xdp.isdigit() else UNKNOWN
vbios=sorted(set(re.findall(r'(?:HCU|GPU)\[\d+\]\s*:\s*VBIOS version:\s*(\S+)',query('hy-smi','--showvbios'))))
environment=field(raw,r'^Execution environment:\s*(.+)$')
environment={'container':'容器','unknown':UNKNOWN}.get(environment,environment)
base_rows=[('执行环境',environment),('操作系统',os_name),('内核',next(iter(query('uname').splitlines()),UNKNOWN)),
           ('BIOS version',files.get('/sys/class/dmi/id/bios_version',UNKNOWN)),
           ('CPU',field(cpu,r'^Model name:\s*(.+)$')),('CPU 拓扑',topology),
           ('hipcc 编译器版本',compiler),('rocminfo',roc_summary),
           ('GPU驱动版本',field(query('hy-smi','--showdriverversion'),r'^Driver Version:\s*(.+)$')),
           ('GPU VBIOS','; '.join(vbios) or UNKNOWN),('DTK 路径',root),
           ('IOMMU 启动参数',iommu_summary),('IOMMU 设备默认域',domain_summary),
           ('XDP',xdp_state),('XDP size',xdp)]
basic_queries=evidence(lambda sec,cmd:'System / CPU' in sec or Path(shlex.split(cmd)[0]).name in ('hipcc','rocminfo') or cmd=='lspci -D -vvv', wrap=False)
file_queries=''.join('<div class="query"><p class="muted">'+esc(label)+'</p><button type="button" class="copy">复制查询命令</button><pre>'+esc(command)+'</pre></div>' for label,command in (
    ('BIOS 版本','cat /sys/class/dmi/id/bios_version'),('操作系统','cat /etc/os-release'),
    ('IOMMU 启动参数','cat /proc/cmdline'),('DTK 路径查询','printenv ROCM_PATH'),
    ('XDP 配置','cat /sys/module/hycu/parameters/xdp_size'),
    ('IOMMU 默认域','for f in /sys/kernel/iommu_groups/*/type; do printf "%s: " "$f"; cat "$f"; done')))
acs_items=[]
for name in ('SrcValid','TransBlk','ReqRedir','CmpltRedir','UpstreamFwd','EgressCtrl','DirectTrans'):
    cap_values,ctl_values=set(),set()
    for body in acs_blocks:
        cap=field(body,r'^\s*ACSCap:\s*(.+)$')
        ctl=field(body,r'^\s*ACSCtl:\s*(.+)$')
        if cap==UNKNOWN and ctl==UNKNOWN:continue
        cap_values.add(field(cap,r'\b'+name+r'([+-])'))
        ctl_values.add(field(ctl,r'\b'+name+r'([+-])'))
    def acs_status(values,support=False):
        if not values or values=={UNKNOWN}:return '未确认'
        if UNKNOWN in values:return '部分未确认'
        if support:return '全部设备支持' if values=={'+'} else '全部设备不支持' if values=={'-'} else '部分设备支持'
        return '全部开启（+）' if values=={'+'} else '全部未开启（-）' if values=={'-'} else '部分开启（+），部分未开启（-）'
    acs_items.append((name,acs_status(cap_values,True),acs_status(ctl_values)))
basic_table=table(['项目',host],base_rows[:-2])
acs_row='<tr><td>ACS 能力与控制</td><td>'+esc(acs_summary)+'</td></tr>'
acs_row+='<tr><td colspan="2"><details style="margin:0 0 0 24px"><summary>ACS 分项状态</summary>'+table(['ACS 项','能力支持','开启状态'],acs_items)+'<p class="muted">“能力支持”表示硬件是否具备该项功能，“开启状态”表示该项功能当前是否启用。</p></details></td></tr>'
acs_row+=''.join('<tr><td>'+esc(label)+'</td><td>'+esc(value)+'</td></tr>' for label,value in base_rows[-2:])
basic_table=basic_table.replace('</tbody></table>',acs_row+'</tbody></table>',1)
panel('basic','基础环境信息',basic_table+'<details><summary>基础环境查询命令</summary><h3>'+esc(host)+'</h3>'+basic_queries+file_queries+'</details>')

# Device blocks retain the association between sysfs path, PCI evidence and NIC.
blocks = re.split(r'^--- (GPU |Network interface: |RDMA device: )([^\n]+?) ---\s*$', raw, flags=re.M)
devices = {'GPU ': [], 'Network interface: ': [], 'RDMA device: ': []}
for i in range(1, len(blocks), 3):
    kind, name, text = blocks[i:i+3]
    path = field(text, r'^PCI path: (.+)$')
    bdf = field(path, r'([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])$')
    def attr(suffix):
        return field(text, r'^/[^\n:]+/' + re.escape(suffix) + r':\s*([^\n]+)$')
    devices[kind].append({'name': name, 'text': text, 'path': path, 'bdf': bdf,
                         'numa': attr('numa_node'), 'speed': attr('current_link_speed'),
                         'width': attr('current_link_width'), 'max_speed': attr('max_link_speed'),
                         'max_width': attr('max_link_width')})
# Read attributes globally as repeated PCI blocks may reference earlier evidence.
def pci_value(device, suffix):
    if device['path'] != UNKNOWN:
        return files.get(device['path']+'/'+suffix, UNKNOWN)
    return UNKNOWN

gpus = [d for d in devices['GPU '] if not re.search(r'VGA compatible controller:.*(?:ASPEED|Matrox)', d['text'], re.I)]
nics = devices['RDMA device: ']
clocks = query('hy-smi', '--showclocks')
clock_data = {}
for line in clocks.splitlines():
    match = re.search(r'(?:GPU|HCU)\[(\d+)\]\s*:\s*(fclk|mclk|sclk|socclk)\s+clock level:\s*\d+\s*\(([^)]*)\)', line, re.I)
    if match:
        index, clock, frequency = match.groups()
        clock_data.setdefault(index, {})[clock.lower()] = frequency
smi_fields = {}
for option in ('--showid','--showproductname','--showvbios','--showbus'):
    for line in query('hy-smi',option).splitlines():
        match = re.match(r'(?:HCU|GPU)\[(\d+)\]\s*:\s*([^:]+):\s*(.*)',line)
        if match:
            index,key,value=match.groups()
            smi_fields.setdefault(index,{})[key.strip().lower()]=value.strip()
driver_version = field(query('hy-smi','--showdriverversion'),r'^Driver Version:\s*(.+)$')
version_rows=[]
for index,values in sorted(smi_fields.items(),key=lambda item:int(item[0])):
    bus=field(values.get('pci bus',''),r'^([0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7])')
    version_rows.append((index,bus,values.get('card series',UNKNOWN),values.get('device id',UNKNOWN),driver_version,values.get('vbios version',UNKNOWN)))
mkfd_oam = {}
for line in query('python3', 'oam-mkfd').splitlines():
    match = re.fullmatch(r'OAM ID ([0-9a-f:]+\.[0-7]) (\d+)', line)
    if match:
        mkfd_oam[match[1]] = match[2]
gpu_locations = {}
for row in version_rows:
    index, bus = row[:2]
    oam_id = field(smi_fields[index].get('pci bus', ''), r'\bOAM ID:\s*(\d+)\b')
    if oam_id == UNKNOWN:
        oam_id = field(smi_fields[index].get('oam id', ''), r'^(\d+)$')
    if oam_id == UNKNOWN:
        oam_id = mkfd_oam.get(bus.lower(), UNKNOWN)
    gpu_locations[index] = (bus, oam_id)

clock_rows = [
    (index, gpu_locations.get(index, (UNKNOWN, UNKNOWN))[0])
    + tuple(re.sub(r'\s*MHz$', '', values.get(clock, UNKNOWN), flags=re.I)
            for clock in ('fclk', 'mclk', 'sclk', 'socclk'))
    for index, values in sorted(clock_data.items(), key=lambda item: int(item[0]))
]
panel('gpu', 'GPU 信息', '')
panel('gpu-version', '设备与版本', table(['GPU 管理编号', 'PCIe 地址', 'OAM ID', '型号', 'Device ID', '驱动版本', 'VBIOS'],
      [row[:2] + (gpu_locations[row[0]][1],) + row[2:] for row in version_rows]) + evidence(lambda sec, cmd: 'DTK / GPU' in sec and (
          cmd.startswith('lspci ') or cmd.endswith('oam-mkfd')
          or Path(cmd).name=='hy-smi'
          or any(option in shlex.split(cmd) for option in ('--showid','--showproductname','--showdriverversion','--showvbios')))), child=True)
panel('gpu-clock', '频率', ''
      + table(['GPU', 'PCIe 地址', 'fclk MHz', 'mclk MHz', 'sclk MHz', 'socclk MHz'], clock_rows)
      + evidence(lambda sec, cmd: cmd.endswith(('--showclocks', '--showbus'))), child=True)
gpu_indices = {row[1].lower(): row[0] for row in version_rows if row[1] != UNKNOWN}
panel('gpu-pcie', 'PCIe 链路', table(['GPU','PCI 地址','NUMA','当前速率','当前宽度','最大速率','最大宽度'], [(gpu_indices.get(d['bdf'].lower(), UNKNOWN),d['bdf'],pci_value(d,'numa_node'),pci_value(d,'current_link_speed'),pci_value(d,'current_link_width'),pci_value(d,'max_link_speed'),pci_value(d,'max_link_width')) for d in gpus]), child=True)
network_rows=[]
for d in devices['Network interface: ']:
    text = d['text']
    network_rows.append((d['name'], d['bdf'], pci_value(d,'numa_node'), field(text,r'^driver:\s*(.+)$'), field(text,r'^version:\s*(.+)$'), field(text,r'^firmware-version:\s*(.+)$'),field(text,r'/operstate:\s*(\S+)'),field(text,r'/mtu:\s*(\S+)')))
hca_rows=[]
for d in nics:
    interface=next((row for row in network_rows if row[1]==d['bdf'] and d['bdf']!=UNKNOWN),None)
    driver=field(d['text'],r'^RDMA driver:\s*(.+)$')
    if driver==UNKNOWN and interface:driver=interface[3]
    module_version=files.get('/sys/module/'+driver+'/version',UNKNOWN)
    if module_version==UNKNOWN and interface:module_version=interface[4]
    pci_text='\n'.join(body for _,cmd,body in commands if cmd.startswith('lspci ') and d['bdf'] in cmd)
    model=field(pci_text,r'\b(MT\d+)\b')
    hca_rows.append((d['name'],d['bdf'],model,pci_value(d,'numa_node'),driver,module_version,field(d['text'],r'/fw_ver:\s*(.+)$')))
physical_interfaces=[row for row in network_rows if row[1]!=UNKNOWN]
virtual_interfaces=[row for row in network_rows if row[1]==UNKNOWN]
def recipe(label,command):
    return '<div class="query"><p class="muted">'+esc(label)+'</p><button type="button" class="copy">复制查询命令</button><pre>'+esc(command)+'</pre></div>'
def batch_query(names,variable,template):
    return 'for '+variable+' in '+' '.join(shlex.quote(name) for name in names)+'; do\n    printf \'\\n%s\\n\' "$'+variable+'"\n    '+template+'\ndone'
interfaces=[row[0] for row in physical_interfaces if row[1] in {d['bdf'] for d in nics}]
modules=sorted({row[4] for row in hca_rows if row[4]!=UNKNOWN})
nic_queries=''
if interfaces:nic_queries+=recipe('接口驱动与固件版本',batch_query(interfaces,'iface','ethtool -i "$iface"'))
if modules:nic_queries+=recipe('已加载驱动版本',batch_query(modules,'module','cat "/sys/module/$module/version"'))
if nics:nic_queries+=recipe('RDMA 固件版本',batch_query([d['name'] for d in nics],'hca','cat "/sys/class/infiniband/$hca/fw_ver"'))
panel('nic','网卡信息','')
nic_queries+=evidence(lambda sec,cmd:'Network / RDMA' in sec and ('ethtool -i' in cmd or 'ibv_devinfo' in cmd or cmd.startswith('lspci ')), wrap=False)
panel('nic-version','设备与版本',table(['HCA','PCI 地址','型号','NUMA','驱动模块','模块版本','网卡固件版本'],hca_rows)+'<details><summary>查询命令与原始输出</summary>'+nic_queries+'</details>', child=True)
port_rows=[]
gid_rows=[]
for path, value in files.items():
    match=re.match(r'/sys/class/infiniband/([^/]+)/ports/(\d+)/state$',path)
    if match:
        hca,port=match.groups(); parent=path.rsplit('/',1)[0]
        port_rows.append((hca,port,value,files.get(parent+'/phys_state',UNKNOWN),files.get(parent+'/rate',UNKNOWN),files.get(parent+'/link_layer',UNKNOWN)))
    match=re.match(r'/sys/class/infiniband/([^/]+)/ports/(\d+)/gids/(\d+)$',path)
    if match:
        hca,port,index=match.groups();parent=path.split('/gids/')[0]
        gid_type = files.get(parent+'/gid_attrs/types/'+index, UNKNOWN)
        link_layer = files.get(parent+'/link_layer', '').strip().lower()
        if gid_type == 'IB/RoCE v1':
            gid_type = {'ethernet': 'RoCE v1', 'infiniband': 'IB'}.get(link_layer, gid_type)
        gid_rows.append((hca,port,index,value,gid_type,files.get(parent+'/gid_attrs/ndevs/'+index,UNKNOWN)))
address_details = evidence(lambda sec, cmd: cmd in ('ip -details address show', 'ifconfig -a'))
panel('ports','网卡状态','<h3>RDMA 端口</h3>'+table(['HCA','端口','逻辑状态','物理状态','速率','链路类型'],port_rows)+'<h3>物理网卡接口</h3>'+table(['接口','PCI 地址','状态','MTU'],[(r[0],r[1],r[6],r[7]) for r in physical_interfaces])+'<details><summary>虚拟接口及未取得 PCI 关联的接口</summary>'+table(['接口','状态','MTU'],[(r[0],r[6],r[7]) for r in virtual_interfaces])+'</details>' + address_details, child=True)
def parse_pfc(tool, body):
    if '[unavailable]' in body or '[incomplete]' in body:
        return None
    if tool == 'mlnx_qos':
        match = re.search(r'^PFC configuration:[ \t]*\n[ \t]*priority[ \t]+([^\n]+)\n[ \t]*enabled[ \t]+([^\n]+)', body, re.M | re.I)
        if match and match[1].split() == list('01234567'):
            values = match[2].split()
            if len(values) == 8 and set(values) <= {'0','1'}:
                return tuple(i for i, value in enumerate(values) if value == '1')
    else:
        match = re.search(r'^[ \t]*prio-pfc[ \t]+([^\n]+)', body, re.M)
        tokens = match[1].split() if match else []
        if len(tokens) == 8 and all(re.fullmatch(r'[0-7]:(on|off)', token) for token in tokens):
            values = dict(token.split(':') for token in tokens)
            if len(values) == 8:
                return tuple(i for i in range(8) if values[str(i)] == 'on')
    return None

pfc_results = {}
for _, cmd, body in commands:
    args = cmd.split()
    if args[:2] == ['mlnx_qos','-i'] or args[:4] == ['dcb','pfc','show','dev']:
        if len(args) in (3,5):
            pfc_results.setdefault(args[-1], {})[args[0]] = parse_pfc(args[0],body)
pfc_rows = []
for hca, port, _, _, _, link in port_rows:
    logical = sorted({row[5] for row in gid_rows if row[:2] == (hca,port) and row[5] != UNKNOWN})
    device = next((d for d in nics if d['name'] == hca), None)
    physical = [row[0] for row in physical_interfaces if device and row[1] == device['bdf']]
    candidates = logical or (physical if sum(row[0] == hca for row in port_rows) == 1 and len(physical) == 1 else [])
    queried, statuses = [], []
    for iface in candidates:
        targets = [iface] if iface in pfc_results else physical if len(physical) == 1 else []
        bond = files.get('/proc/net/bonding/'+iface, '')
        if bond:
            targets = re.findall(r'^Slave Interface: (.+)$', bond, re.M)
        for query_iface in targets:
            queried.append(query_iface)
            readings = {value for value in pfc_results.get(query_iface,{}).values() if value is not None}
            if len(readings) > 1:
                statuses.append(query_iface+'：查询结果不一致')
            elif readings:
                enabled = next(iter(readings))
                statuses.append(query_iface+'：'+('开启优先级 '+', '.join(map(str,enabled)) if enabled else '全部关闭（0～7 均 off）'))
            else:
                statuses.append(query_iface+'：未取得完整优先级配置')
    state = '不适用（InfiniBand 链路）' if link.lower() == 'infiniband' else '\n'.join(statuses) or '未取得可查询接口'
    pfc_rows.append((hca+'/'+port, ', '.join(logical or candidates) or UNKNOWN, ', '.join(queried) or UNKNOWN, state))
panel('pfc','PFC 配置',table(['HCA/端口','逻辑接口','查询接口','PFC 状态/优先级'],pfc_rows)+evidence(lambda sec,cmd: cmd.startswith(('mlnx_qos -i ', 'dcb pfc show dev '))), child=True)
panel('nic-pcie','网卡 PCIe 链路',table(['HCA','PCI 地址','NUMA','当前速率','当前宽度','最大速率','最大宽度'],[(d['name'],d['bdf'],pci_value(d,'numa_node'),pci_value(d,'current_link_speed'),pci_value(d,'current_link_width'),pci_value(d,'max_link_speed'),pci_value(d,'max_link_width')) for d in nics]), child=True)
panel('gid','GID 信息',table(['HCA','端口','索引','GID','类型','关联网卡'],gid_rows), child=True)
bond_rows=[(path,value) for path,value in files.items() if path.startswith('/proc/net/bonding/')]
panel('bond','bond 状态',table(['配置文件','记录'],bond_rows) + '<p class="muted">完整 bond 参数见原始记录。</p>', child=True)

issue_groups = {}
for command, reason in issues:
    if command.startswith('dcb pfc show dev '):
        iface = command.split()[-1]
        covered = pfc_results.get(iface, {}).get('mlnx_qos') is not None
        item = ('PFC 优先级开关', '已由 mlnx_qos 取得' if covered else '未取得完整优先级配置')
        label = 'dcb pfc show dev'
    else:
        label = command
        item = {
            'ip -details address show': ('接口地址与详细属性', '接口状态、MTU 等见 sysfs；详细地址需核对 ifconfig 或原始记录'),
            'ip route show': ('IPv4 路由', '已采集 /proc/net/route' if '/proc/net/route' in files else '未取得 IPv4 路由'),
            'rdma link show': ('RDMA 端口状态与接口关联', '状态见 RDMA sysfs；GID 关联接口见 GID 表'),
        }.get(command, ('采集诊断', '详见原始记录'))
    key = (label, item[0], item[1], reason)
    issue_groups[key] = issue_groups.get(key, 0) + 1
issue_rows = [(*key, count) for key, count in issue_groups.items()]
panels[1] = (
    '<section id="issues"><h2>需要关注的事项</h2>'
    + (table(['查询命令', '查询内容', '影响或替代来源', '采集提示', '次数'], issue_rows)
       if issue_rows else '<p>未发现当前检查范围内的问题</p>')
    + '</section>'
)

overview_note = ('<p>有 {0} 条查询未完成或不可用，见<a href="#issues">需要关注的事项</a>。</p>'.format(len(issues))
                 if issues else '<p>未发现当前检查范围内的问题</p>')
panels[0] = (
    '<section id="overview"><h2>采集概览</h2>'
    '<p><span class="badge">环境信息快照</span> 共记录 {0} 次查询。</p>'.format(len(commands))
    + overview_note
    + table(['节点', 'GPU', 'NIC 设备'], [(host, len(gpus), len(nics))])
    + '</section>'
)


# Match basic-perf-probe: compare the first three components below /sys/devices.
def switch_key(device):
    prefix = '/sys/devices/'
    path = device['path']
    if path.startswith(prefix):
        parts = path[len(prefix):].split('/')
        if len(parts) >= 3 and all(part not in ('', '.', '..') for part in parts):
            return '/'.join(parts[:3])
    return None

nic_switches = {}
for nic in nics:
    key = switch_key(nic)
    if key:
        nic_switches.setdefault(key, []).append(nic)

groups, direct, by_oam = {}, {}, {}
for gpu in gpus:
    bdf = gpu['bdf'].lower()
    key = switch_key(gpu)
    if key in nic_switches:
        groups.setdefault(key, []).append((gpu, False))
        direct[bdf] = key
    index = gpu_indices.get(bdf)
    _, oam_id = gpu_locations.get(index, (UNKNOWN, UNKNOWN))
    if oam_id.isdigit():
        by_oam.setdefault(int(oam_id), []).append(gpu)

for first in sorted({oam & ~1 for oam in by_oam}):
    left, right = by_oam.get(first, []), by_oam.get(first + 1, [])
    if len(left) != 1 or len(right) != 1:
        continue
    a, b = left[0], right[0]
    ka, kb = direct.get(a['bdf'].lower()), direct.get(b['bdf'].lower())
    if bool(ka) != bool(kb):
        groups[ka or kb].append((b if ka else a, True))

def gpu_card(gpu, inferred=False):
    bdf = gpu['bdf']
    index = gpu_indices.get(bdf.lower())
    label = 'GPU ' + index if index is not None else 'GPU（' + gpu['name'] + '）'
    _, oam_id = gpu_locations.get(index, (UNKNOWN, UNKNOWN))
    css = ' inferred' if inferred else ''
    return ('<div class="device'+css+'"><strong>'+esc(label)+'</strong><small>BDF '
            +esc(bdf)+' · OAM '+esc(oam_id)+'</small></div>')

topo = []
for number, (key, members) in enumerate(sorted(groups.items()), 1):
    ordered = sorted(members, key=lambda item: int(gpu_indices.get(item[0]['bdf'].lower(), '999999')))
    topo.append('<div class="topo topo-row"><svg class="topo-links" aria-hidden="true"></svg>'
                '<div class="gpu-stack">'+''.join(gpu_card(gpu, inferred) for gpu,inferred in ordered)
                +'</div><div class="bridge switch-node" title="'+esc(key)+'">PCIe Switch '+str(number)
                +'<small>'+esc(key.rsplit('/',1)[-1])+'</small></div><div class="nic-stack">'
                +''.join('<div class="device nic"><strong>'+esc(nic['name'])+'</strong><small>BDF '
                         +esc(nic['bdf'])+'</small></div>' for nic in nic_switches[key])+'</div></div>')
grouped = {gpu['bdf'] for members in groups.values() for gpu, _ in members}
unassigned = [gpu_card(gpu) for gpu in gpus if gpu['bdf'] not in grouped]
unassigned += ['<div class="device nic"><strong>'+esc(nic['name'])+'</strong><small>BDF '
               +esc(nic['bdf'])+'</small></div>' for nic in nics if switch_key(nic) not in groups]
if unassigned:
    topo.append('<h3>未归组设备</h3><div class="topo-unassigned">'+''.join(unassigned)+'</div>')
panel('topology', 'GPU—NIC 拓扑',
      '<p class="muted">实线：GPU 与 NIC 的 PCIe 路径前三级相同。虚线：按相邻偶奇 OAM ID 配对辅助补全。</p>'
      + (''.join(topo) or '<p>未取得 GPU 或 NIC 拓扑数据。</p>')
      + evidence(lambda sec,cmd: cmd=='lspci -D -tv' or cmd.endswith('--showtopo')))
panel('raw','原始输出','<details><summary>展开完整采集记录</summary><button type="button" class="copy">复制完整采集记录</button><pre>'+esc(raw)+'</pre></details>')

page = """<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>rocSHMEM 环境信息报告</title><style>
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#f4f7fa;color:#203349;font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif}header{background:#18324c;color:#fff;padding:28px max(24px,calc((100vw - 1480px)/2))}h1{margin:0;font-size:28px}header p{margin:10px 0 0;color:#d3e0ea}.layout{max-width:1480px;margin:24px auto;padding:0 24px;display:grid;grid-template-columns:230px minmax(0,1fr);gap:24px}main{min-width:0}nav,section{background:#fff;border:1px solid #d8e2eb;border-radius:10px}nav{position:sticky;top:18px;align-self:start;max-height:calc(100vh - 36px);overflow:auto;padding:14px}nav strong{display:block;padding:8px;color:#5b6d80;font-size:13px}a{color:#165f9c}nav a{display:block;padding:9px 10px;border-left:3px solid transparent;text-decoration:none;border-radius:5px}.report-child{scroll-margin-top:20px}.report-child+.report-child{border-top:1px solid #e5ebf0;margin-top:20px}nav a.nav-child{margin-left:20px;padding-left:12px}nav a:hover,nav a:focus{background:#e7f1fa;border-left-color:#165f9c}section{padding:24px;margin-bottom:20px;scroll-margin-top:20px}h2{margin:0 0 18px;font-size:23px}h3{font-size:18px;margin:22px 0 12px}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;text-align:left}th,td{padding:12px 14px;border-bottom:1px solid #e5ebf0;vertical-align:top;white-space:pre-line;overflow-wrap:anywhere}th{background:#edf3f7;font-weight:650}td:first-child{min-width:100px}tr:last-child td{border-bottom:0}.muted,small{color:#5b6d80}.badge{display:inline-block;padding:4px 10px;border-radius:5px;background:#edf3f7}details{margin-top:16px}summary{cursor:pointer;background:#f3f6fa;padding:12px 14px;border-radius:7px;font-weight:600;overflow-wrap:anywhere}.query{margin:16px 0 20px 16px;padding-left:16px;border-left:2px solid #dce5ef}.query pre{margin-top:0}.copy{border:1px solid #d8e2eb;background:white;color:#203349;border-radius:6px;padding:9px 13px;cursor:pointer}#basic td:first-child{width:220px;white-space:nowrap}#basic table{table-layout:fixed}pre{margin:10px 0;background:#1b2f43;color:#eff5fa;border-radius:7px;padding:16px;font:12px/1.6 Consolas,monospace;white-space:pre;overflow:auto;max-height:520px}.topo{display:grid;grid-template-columns:minmax(140px,1fr) 200px minmax(140px,1fr);gap:30px;align-items:center;background:#f8fbfd;border:1px solid #e3ebf1;border-radius:8px;padding:20px;margin:16px 0}.device{background:white;border:1px solid #9fb8cb;border-radius:7px;padding:14px;margin:8px 0;overflow-wrap:anywhere}.device small{display:block}.nic{border-color:#9acbb3}.bridge{padding:18px 12px;background:#dcebf6;border:1px solid #9fb8cb;border-radius:8px;text-align:center;overflow-wrap:anywhere}
@media(max-width:900px){.layout{display:block;padding:0 12px}nav{position:static;max-height:none;margin-bottom:20px}.topo{grid-template-columns:1fr;gap:8px}section{padding:16px}header{padding:24px}h1{font-size:24px}}@media print{nav{display:none}.layout{display:block}pre{max-height:none;white-space:pre-wrap}details:not([open]){display:none}}

.topo-row{position:relative}.topo-links{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.topo-links path{fill:none;stroke:#92acc0;stroke-width:2;stroke-linejoin:round}.topo-links path.inferred{stroke:#c19136;stroke-dasharray:5 4}.topo-row .device.inferred{border-style:dashed;border-color:#c19136}.switch-node small{display:block}.topo-unassigned{display:flex;gap:12px;flex-wrap:wrap}
nav a[aria-current="location"]{background:#e7f1fa;border-left-color:#165f9c;color:#165f9c;font-weight:650}
</style></head><body><header><h1>rocSHMEM 环境信息报告</h1><p>节点：__HOST__　·　采集时间：__STAMP__　·　仅环境采集</p></header><div class="layout"><nav aria-label="报告目录">__NAV__</nav><main>__PANELS__</main></div><script>
const reportNav=document.querySelector('nav');
const chapterLinks=Array.from(reportNav.querySelectorAll('a[href^="#"]'));
const chapters=chapterLinks.map(link=>({link,target:document.getElementById(link.hash.slice(1))})).filter(item=>item.target);
let currentChapter=null;
let chapterFrame=0;
function updateChapter(){
 chapterFrame=0;
 let selected=chapters[0];
 for(const chapter of chapters){
  if(chapter.target.getBoundingClientRect().top<=100)selected=chapter;
 }
 if(!selected||selected===currentChapter)return;
 if(currentChapter)currentChapter.link.removeAttribute('aria-current');
 selected.link.setAttribute('aria-current','location');
 currentChapter=selected;
 const linkBox=selected.link.getBoundingClientRect();
 const navBox=reportNav.getBoundingClientRect();
 if(linkBox.top<navBox.top||linkBox.bottom>navBox.bottom){
  reportNav.scrollTop+=linkBox.top-navBox.top-reportNav.clientHeight/2;
 }
}
function scheduleChapter(){
 if(!chapterFrame)chapterFrame=requestAnimationFrame(updateChapter);
}
window.addEventListener('scroll',scheduleChapter,{passive:true});
window.addEventListener('resize',scheduleChapter);
window.addEventListener('hashchange',scheduleChapter);
document.addEventListener('toggle',scheduleChapter,true);
window.addEventListener('load',scheduleChapter);
scheduleChapter();
document.addEventListener('click',async function(event){
 const button=event.target.closest('button.copy');if(!button)return;
 const block=button.nextElementSibling;const text=block.textContent;
 try{await navigator.clipboard.writeText(text);button.textContent='已复制';}
 catch(error){const range=document.createRange();range.selectNodeContents(block);const selection=window.getSelection();selection.removeAllRanges();selection.addRange(range);button.textContent='请按 Ctrl+C 复制';}
});
var topologyRows=Array.from(document.querySelectorAll('.topo-row'));
function drawTopology(row){
  var svg=row.querySelector('.topo-links'),sw=row.querySelector('.switch-node');
  if(!svg||!sw)return;
  var origin=svg.getBoundingClientRect(),box=sw.getBoundingClientRect();
  if(!origin.width||!origin.height)return;
  svg.setAttribute('viewBox','0 0 '+origin.width+' '+origin.height);
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  var vertical=window.matchMedia('(max-width:900px)').matches;
  ['gpu','nic'].forEach(function(side){
    var stack=row.querySelector('.'+side+'-stack');
    if(!stack)return;
    var stackBox=stack.getBoundingClientRect();
    Array.from(stack.querySelectorAll('.device')).forEach(function(card){
      var b=card.getBoundingClientRect(),gpu=side==='gpu',d;
      if(vertical){
        var startX=(gpu?b.right:b.left)-origin.left,startY=(b.top+b.bottom)/2-origin.top;
        var busX=gpu?origin.width-7:7,joinY=(gpu?(stackBox.bottom+box.top)/2:(box.bottom+stackBox.top)/2)-origin.top;
        var endX=(box.left+box.right)/2-origin.left,endY=(gpu?box.top:box.bottom)-origin.top;
        d='M '+startX+' '+startY+' H '+busX+' V '+joinY+' H '+endX+' V '+endY;
      }else{
        var startX=(gpu?b.right:b.left)-origin.left,startY=(b.top+b.bottom)/2-origin.top;
        var endX=(gpu?box.left:box.right)-origin.left,endY=(box.top+box.bottom)/2-origin.top;
        var busX=(startX+endX)/2;
        d='M '+startX+' '+startY+' H '+busX+' V '+endY+' H '+endX;
      }
      var path=document.createElementNS('http://www.w3.org/2000/svg','path');
      path.setAttribute('d',d);
      path.setAttribute('class',card.classList.contains('inferred')?'inferred':'direct');
      svg.appendChild(path);
    });
  });
}
var topologyFrame=false;
function updateTopology(){topologyFrame=false;topologyRows.forEach(drawTopology);}
function scheduleTopology(){if(!topologyFrame){topologyFrame=true;requestAnimationFrame(updateTopology);}}
if(window.ResizeObserver){
  var topologyObserver=new ResizeObserver(scheduleTopology);
  topologyRows.forEach(function(row){topologyObserver.observe(row);});
}
window.addEventListener('resize',scheduleTopology);
window.addEventListener('beforeprint',updateTopology);
window.addEventListener('afterprint',scheduleTopology);
if(document.fonts)document.fonts.ready.then(scheduleTopology);
scheduleTopology();
</script></body></html>"""
values = {'HOST':esc(host),'STAMP':esc(stamp),'NAV':''.join(nav),'PANELS':''.join(panels)}
page = re.sub(r'__(HOST|STAMP|NAV|PANELS)__', lambda match:values[match.group(1)], page)
target.write_text(page,encoding='utf-8')

PY_ENV_HTML
  local status=$?
  if [[ $status -eq 0 ]]; then
    echo "Environment HTML saved to: $2"
  else
    echo "WARNING: HTML generation failed ($status); raw log retained: $1" >&2
  fi
  return 0
}

PrintEnvInfo() {
  local report_host
  report_host=$(hostname 2>/dev/null) || report_host=${HOSTNAME:-unknown}
  report_host=${report_host//[^a-zA-Z0-9._-]/_}
  local env_log="$LOG_DIR/env_info_${report_host:-unknown}.log"
  if ! mkdir -p -- "$LOG_DIR"; then
    printf 'Error: Failed to create log directory %s\n' "$LOG_DIR" >&2
    return 1
  fi

  if ! {
    printf 'Environment info: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
    if [[ -x "$ROCSHMEM_INFO" ]]; then
      printf '\n=== Build Info and Env Vars ===\n'
      EnvCommand "$ROCSHMEM_INFO" --env:all
    fi

    printf '\n=== Selected runtime environment ===\n'
    env | LC_ALL=C sort | grep -E '^(ROCSHMEM_|ROCM_PATH=|HIP_VISIBLE_DEVICES=|ROCR_VISIBLE_DEVICES=|GPU_DEVICE_ORDINAL=|UCX_|OMPI_MCA_|LD_LIBRARY_PATH=)' || true
    PrintHardwareEnvInfo
    printf '\n=== Collection complete ===\n'
    printf 'Environment info collection completed: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')"
  } > "$env_log" 2>&1; then
    printf 'Error: Failed to collect environment info in %s\n' "$env_log" >&2
    return 1
  fi

  printf 'Environment info saved to: %s\n' "$env_log"
  if [[ "${ENV_INFO_LOG_ONLY:-0}" != 1 ]]; then
    WriteEnvHtml "$env_log" "${env_log%.log}.html"
  fi
}

CollectRemoteEnvInfo() {
  local host=$1 response=$2
  if ! timeout 180s ssh -T -o BatchMode=yes -o ConnectTimeout=5 \
    -o ServerAliveInterval=5 -o ServerAliveCountMax=2 -- "$host" '
      dir=$(mktemp -d) || exit 1
      trap '\''rm -rf -- "$dir"'\'' EXIT
      bash -lc '\''
        if ! command -v hipcc >/dev/null || ! command -v rocminfo >/dev/null; then
          if [[ -r /opt/dtk/env.sh ]]; then
            source /opt/dtk/env.sh </dev/null ||
              echo "WARNING: Failed to load /opt/dtk/env.sh" >&2
          fi
        fi
        export ENV_INFO_LOG_ONLY=1
        exec bash -s -- "$1"
      '\'' bash "$dir" >&2 || exit 1
      set -- "$dir"/env_info_*.log
      [ "$#" -eq 1 ] && [ -f "$1" ] || exit 1
      printf "%s\n" "${1##*/}"
      cat -- "$1"
    ' < "${BASH_SOURCE[0]}" > "$response"; then
    rm -f -- "$response"
    return 1
  fi
}

SaveRemoteEnvInfo() {
  local host=$1 response=$2 name
  {
    if ! IFS= read -r name || [[ ! $name =~ ^env_info_[a-zA-Z0-9._-]+\.log$ ]]; then
      rm -f -- "$response"
      return 1
    fi
    if [[ -n ${collected[$name]:-} ]]; then
      rm -f -- "$response"
      printf 'Skipping %s: %s already collected\n' "$host" "$name"
      return 0
    fi
    if ! cat > "$response.log" || ! mv -fT -- "$response.log" "$LOG_DIR/$name"; then
      rm -f -- "$response" "$response.log"
      return 1
    fi
  } < "$response"
  rm -f -- "$response"
  collected[$name]=1
  printf 'Environment info saved to: %s\n' "$LOG_DIR/$name"
  WriteEnvHtml "$LOG_DIR/$name" "$LOG_DIR/${name%.log}.html"
}

usage="Usage: $0 <log_dir> [--hosts host1,host2]"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
  echo "$usage"
  exit 0
fi
if [[ -z "${1:-}" ]] || { [[ $# -ne 1 ]] &&
  [[ $# -ne 3 || $2 != --hosts || ! $3 =~ ^[a-zA-Z0-9_.@:-]+(,[a-zA-Z0-9_.@:-]+)*$ ]]; }; then
  echo "$usage" >&2
  exit 2
fi

LOG_DIR=$1
ROCSHMEM_INFO=${ROCSHMEM_INFO:-$(command -v rocshmem_info || true)}
PrintEnvInfo || exit $?
if [[ $# -eq 3 ]]; then
  declare -A collected
  local_host=$(hostname 2>/dev/null) || local_host=${HOSTNAME:-unknown}
  local_host=${local_host//[^a-zA-Z0-9._-]/_}
  collected[env_info_${local_host:-unknown}.log]=1
  IFS=, read -ra hosts <<< "$3"
  remote_work=$(mktemp -d -- "$LOG_DIR/.env-remote.XXXXXXXX") || exit 1
  trap 'rm -rf -- "$remote_work"' EXIT
  declare -A requested
  remote_hosts=()
  remote_pids=()
  for host in "${hosts[@]}"; do
    [[ $host == "$local_host" || $host == localhost ]] && continue
    [[ -n ${requested[$host]:-} ]] && continue
    requested[$host]=1
    index=${#remote_hosts[@]}
    remote_hosts+=("$host")
    CollectRemoteEnvInfo "$host" "$remote_work/$index" &
    remote_pids+=("$!")
    # Limit simultaneous SSH sessions to four.
    if (( index >= 3 )); then
      wait "${remote_pids[index-3]}" || true
    fi
  done
  for index in "${!remote_hosts[@]}"; do
    if wait "${remote_pids[index]}" && [[ -f "$remote_work/$index" ]]; then
      SaveRemoteEnvInfo "${remote_hosts[index]}" "$remote_work/$index" ||
        printf 'WARNING: Failed to save environment information for %s\n' "${remote_hosts[index]}" >&2
    else
      printf 'WARNING: Environment collection failed for %s\n' "${remote_hosts[index]}" >&2
    fi
  done
fi
