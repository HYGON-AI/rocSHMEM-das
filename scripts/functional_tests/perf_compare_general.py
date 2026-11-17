#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: MIT

"""Generate colored Excel comparisons for BASE vs NEW rocSHMEM or BASE vs RCCL.

rocSHMEM SIZE is the Volume column (the first numeric column).  RCCL SIZE is
the rccl-tests size column.  For repeated rocSHMEM runs, the row with the
largest raw Bandwidth is selected for each operation/backend/SIZE.
"""

import argparse
import re
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import xlsxwriter


RCCL_TO_ROCSHMEM = {
    "all_gather_perf": "fcollect",
    "all_reduce_perf": "teamreduction",
    "alltoall_perf": "alltoall",
    "alltoallv_perf": "alltoallv",
    "broadcast_perf": "teambroadcast",
    "reduce_scatter_perf": "teamreducescatter",
}

OP_ORDER = [
    "alltoall", "alltoallv", "fcollect", "teambroadcast",
    "teamreducescatter", "teamreduction",
]

# RMA operations are compared only in rocSHMEM-vs-rocSHMEM mode.
RMA_DEVICE_ORDER = [
    "put", "wgput", "waveput", "putnbi", "wgputnbi", "waveputnbi",
    "get", "wgget", "waveget", "getnbi", "wggetnbi", "wavegetnbi",
    "defaultctxput", "teamctxput", "defaultctxputnbi", "teamctxputnbi",
    "defaultctxget", "teamctxget", "defaultctxgetnbi", "teamctxgetnbi",
]
RMA_HOST_ORDER = ["putmem_on_stream", "getmem_on_stream"]


DISPLAY_NAMES = {
    "fcollect": "allgather",
    "teambroadcast": "broadcast",
    "teamreducescatter": "reducescatter",
    "teamreduction": "allreduce",
}

# Normalize host-side on-stream case names to the RCCL comparison keys.
HOST_TO_COMPARISON = {
    "alltoallmem_on_stream": "alltoall",
    "alltoallv_on_stream": "alltoallv",
    "broadcastmem_on_stream": "teambroadcast",
    "fcollect_on_stream": "fcollect",
    "reduce_on_stream": "teamreduction",
    "reduce_scatter_on_stream": "teamreducescatter",
}


HOST_OPERATIONS = {**HOST_TO_COMPARISON, **{op: op for op in RMA_HOST_ORDER}}
ROCSHMEM_DEVICE_OPS = set(OP_ORDER + RMA_DEVICE_ORDER)


DEVICE_RE = re.compile(r"\bdevice\s+\d+\s+\[[^\]]+\]\s+(.+?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class RocshmemPoint:
    bandwidth: float
    algbw: Optional[float]
    busbw: Optional[float]
    source: str


@dataclass(frozen=True)
class RcclPoint:
    algbw: float
    busbw: float
    source: str


def numeric_tokens(line: str):
    parts = line.split()
    if not parts:
        return None
    for index, token in enumerate(parts):
        # RMA has no RCCL-equivalent aligned bandwidth; retain its raw bandwidth.
        if len(parts) == 8 and index in (5, 6) and token.lower() in ("na", "n/a"):
            continue
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", token):
            return None
    return parts


def detect_device_types(roots: Iterable[Path]):
    """Return every distinct device type printed after the PCI address."""
    device_types = set()
    for root in roots:
        for path in root.rglob("*.log"):
            try:
                with path.open("r", encoding="utf-8", errors="replace") as stream:
                    for line in stream:
                        match = DEVICE_RE.search(line)
                        if match:
                            device_type = " ".join(match.group(1).split())
                            if device_type:
                                device_types.add(device_type)
            except OSError as exc:
                print(f"WARN: cannot scan device type from {path}: {exc}")
    return sorted(device_types, key=str.casefold)


def parse_rocshmem(root: Path, node: str, selected_w: Optional[int] = None,
                   selected_z: Optional[int] = None) -> Dict[Tuple[str, str, int], RocshmemPoint]:
    result: Dict[Tuple[str, str, int], RocshmemPoint] = {}
    for path in root.rglob("*.log"):
        if path.name.endswith("_best.log") or ".retry." in path.name:
            continue
        match = re.match(r"(.+?)_n\d+_w(\d+)_z(\d+)_", path.name, re.IGNORECASE)
        if not match:
            continue
        op = match.group(1).lower()
        file_w = int(match.group(2))
        file_z = int(match.group(3))
        # Host-side on-stream operations tune W/Z internally.
        if op not in HOST_OPERATIONS:
            if selected_w is not None and file_w != selected_w:
                continue
            if selected_z is not None and file_z != selected_z:
                continue
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"WARN: cannot read {path}: {exc}")
            continue
        for line in lines:
            parts = numeric_tokens(line)
            if parts is None or len(parts) != 8:
                continue
            volume = int(parts[0])
            point = RocshmemPoint(
                float(parts[4]),
                None if parts[5].lower() in ("na", "n/a") else float(parts[5]),
                None if parts[6].lower() in ("na", "n/a") else float(parts[6]), str(path))
            key = (node, op, volume)
            old = result.get(key)
            if old is None or point.bandwidth > old.bandwidth:
                result[key] = point
    return result


def rccl_op_from_filename(path: Path) -> Optional[str]:
    """Map a per-case log name such as all_reduce_perf.log to rocSHMEM."""
    return RCCL_TO_ROCSHMEM.get(path.stem.lower())


def parse_rccl(root: Path, node: str, place: str) -> Dict[Tuple[str, str, int], RcclPoint]:
    result: Dict[Tuple[str, str, int], RcclPoint] = {}
    test_re = re.compile(r"^(?:测试|Test)\s*[:：]\s*([\w.-]+)", re.IGNORECASE)
    for path in root.rglob("*.log"):
        # New format: one log per case and the file stem is the RCCL test name.
        # The header fallback keeps old combined suite logs readable.
        current_op = rccl_op_from_filename(path)
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            print(f"WARN: cannot read {path}: {exc}")
            continue
        for line in lines:
            match = test_re.match(line.strip())
            if match:
                current_op = RCCL_TO_ROCSHMEM.get(match.group(1).lower())
                continue
            if current_op is None:
                continue
            parts = line.split()
            # Most rccl-tests rows contain "type redop root". alltoallv has no
            # redop token, so its time/algbw/busbw columns start one position
            # earlier and the complete row contains 12 instead of 13 tokens.
            if len(parts) < 12 or not re.fullmatch(r"\d+", parts[0]):
                continue
            try:
                size = int(parts[0])
                out_offset = 4 if len(parts) == 12 else 5
                offset = out_offset if place == "out" else out_offset + 4
                point = RcclPoint(float(parts[offset + 1]), float(parts[offset + 2]), str(path))
            except (ValueError, IndexError):
                continue
            if size <= 0:
                continue
            key = (node, current_op, size)
            old = result.get(key)
            if old is None or point.algbw > old.algbw:
                result[key] = point
        if current_op is None:
            print(f"WARN: skip RCCL log with an unmapped case filename: {path}")
    return result


def sizes_for_op(op: str, roc, rccl, minimum: int, maximum: Optional[int]) -> Iterable[int]:
    sizes = {key[2] for key in roc if key[1] == op}
    sizes.update(key[2] for key in rccl if key[1] == op)
    return sorted(size for size in sizes if size >= minimum and (maximum is None or size <= maximum))


def write_number(ws, row, col, value, fmt):
    if value is None:
        ws.write_blank(row, col, None, fmt)
    else:
        ws.write_number(row, col, value, fmt)


def build_workbook(output, device_type, roc, rccl, minimum, maximum, device_groups=None, label_a="BASE", label_b="RCCL"):
    host = {(node, HOST_TO_COMPARISON[op], size): point
            for (node, op, size), point in roc.items() if op in HOST_TO_COMPARISON}
    if device_groups is None:
        device_groups = {"设备侧": {key: point for key, point in roc.items() if key[1] in OP_ORDER}}
    groups = {"主机侧": host, **device_groups}
    return build_roc_comparison(output, device_type, groups, {name: rccl for name in groups},
                                label_a, label_b, minimum, maximum, rccl_mode=True)


def find_default_directory(keyword: str, current_dir: Path) -> Path:
    candidates = sorted(
        (path for path in current_dir.iterdir()
         if path.is_dir() and keyword.casefold() in path.name.casefold()),
        key=lambda path: path.name.casefold(),
    )
    if not candidates:
        raise SystemExit(
            f"No directory containing '{keyword}' was found under current directory: {current_dir}"
        )
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise SystemExit(
            f"Multiple directories containing '{keyword}' were found: {names}. "
            "Please specify the corresponding directory argument explicitly."
        )
    return candidates[0]


def positive_int_list(value):
    try:
        values = list(dict.fromkeys(int(part.strip()) for part in value.split(",")))
    except ValueError:
        raise argparse.ArgumentTypeError("Expected comma-separated positive integers, e.g. 16,32")
    if not values or any(number <= 0 for number in values):
        raise argparse.ArgumentTypeError("W/Z values must be positive integers")
    return values


def device_sheet_name(w, z):
    if w is None and z is None:
        return "设备侧"
    name = f"w{w if w is not None else 'All'}z{z if z is not None else 'All'}"
    if len(name) > 31:
        raise ValueError(f"W/Z worksheet name exceeds Excel's 31-character limit: {name}")
    return name


def load_comparison_groups(intra, inter, ws, zs, ops="coll"):
    host_ops = {}
    device_ops = set()
    if ops in ("coll", "all"):
        host_ops.update(HOST_TO_COMPARISON)
        device_ops.update(OP_ORDER)
    if ops in ("rma", "all"):
        host_ops.update({op: op for op in RMA_HOST_ORDER})
        device_ops.update(RMA_DEVICE_ORDER)
    groups = {"主机侧": {}}
    for w, z in product(ws or [None], zs or [None]):
        points = parse_rocshmem(intra, "intranode", w, z)
        points.update(parse_rocshmem(inter, "internode", w, z))
        groups["主机侧"].update({(node, host_ops[op], size): point
            for (node, op, size), point in points.items() if op in host_ops})
        groups[device_sheet_name(w, z)] = {key: point for key, point in points.items() if key[1] in device_ops}
    return groups


def build_roc_comparison(output, device_type, groups_a, groups_b, label_a, label_b, minimum, maximum, rccl_mode=False):
    output.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    width = 6 if rccl_mode else 7
    with xlsxwriter.Workbook(str(output)) as book:
        border = {"border": 1, "border_color": "#AEBBC5", "font_color": "#263238", "valign": "vcenter"}
        def header(bg, fg="#263238"):
            return book.add_format({**border, "bold": True, "align": "center", "text_wrap": True, "bg_color": bg, "font_color": fg})
        title, intra, inter = header("#FFF4CC"), header("#2E86AB", "#FFFFFF"), header("#168C80", "#FFFFFF")
        base_head, new_head, col_head = header("#DCEFF4"), header("#FFE3A3"), header("#E8F1F5")
        base_num = book.add_format({**border, "align": "right", "num_format": "0.00", "bg_color": "#EDF6DC"})
        new_num = book.add_format({**border, "align": "right", "num_format": "0.00", "bg_color": "#FFF3D6"})
        percent = book.add_format({**border, "align": "right", "num_format": "0.00%", "bg_color": "#E8F1F5"})
        size_fmt = book.add_format({**border, "align": "right", "num_format": "#,##0", "bg_color": "#F4F7F8"})
        text = book.add_format(border)
        book.set_properties({"title": f"{label_a} vs {label_b}", "company": "Hygon"})
        for name, a in groups_a.items():
            b = groups_b.get(name, {})
            sheet = book.add_worksheet(name)
            sheet.hide_gridlines(2)
            sheet.freeze_panes(3, 2)
            sheet.set_zoom(85)
            sheet.set_column(0, 0, 24)
            sheet.set_column(1, 1, 16)
            sheet.set_column(2, 1 + width * 2, 18)
            sheet.merge_range(0, 0, 1, 1, device_type, title)
            sheet.write(2, 0, "Type", col_head)
            sheet.write(2, 1, "Volume (bytes)", col_head)
            for offset, node, fmt in ((2, "IntraNode", intra), (2 + width, "InterNode", inter)):
                sheet.merge_range(0, offset, 0, offset + width - 1, node, fmt)
                sheet.merge_range(1, offset, 1, offset + 2, label_a, base_head)
                sheet.merge_range(1, offset + 3, 1, offset + width - 2, label_b, new_head)
                sheet.write(1, offset + width - 1, "RCCL/BASE - 1" if rccl_mode else "NEW/BASE - 1", col_head)
                metrics = ["Bandwidth", "AlgBwAlignRccl", "BusBwAlignRccl"]
                headers = metrics + (["algbw", "busbw"] if rccl_mode else metrics) + ["Bandwidth change"]
                for i, value in enumerate(headers):
                    sheet.write(2, offset + i, value, base_head if i < 3 else new_head if i < width - 1 else col_head)
                sheet.write_comment(2, offset + width - 1, "RCCL.algbw / BASE.AlgBwAlignRccl - 1" if rccl_mode else "NEW.Bandwidth / BASE.Bandwidth - 1")
            for row, height in ((0, 26), (1, 24), (2, 42)):
                sheet.set_row(row, height)
            row = 3
            for op in (OP_ORDER if rccl_mode else OP_ORDER + RMA_DEVICE_ORDER + RMA_HOST_ORDER):
                start = row
                for size in sizes_for_op(op, a, b, minimum, maximum):
                    sheet.write_number(row, 1, size, size_fmt)
                    for offset, node in ((2, "intranode"), (2 + width, "internode")):
                        pa, pb = a.get((node, op, size)), b.get((node, op, size))
                        for i, metric in enumerate(("bandwidth", "algbw", "busbw")):
                            write_number(sheet, row, offset + i, getattr(pa, metric) if pa else None, base_num)
                        for i, metric in enumerate(("algbw", "busbw") if rccl_mode else ("bandwidth", "algbw", "busbw")):
                            write_number(sheet, row, offset + 3 + i, getattr(pb, metric) if pb else None, new_num)
                        denominator = (pa.algbw if rccl_mode else pa.bandwidth) if pa else None
                        numerator = (pb.algbw if rccl_mode else pb.bandwidth) if pb else None
                        delta = numerator / denominator - 1 if numerator is not None and denominator not in (None, 0) else None
                        write_number(sheet, row, offset + width - 1, delta, percent)
                    row += 1
                if row - start == 1:
                    sheet.write(start, 0, DISPLAY_NAMES.get(op, op), text)
                elif row > start:
                    sheet.merge_range(start, 0, row - 1, 0, DISPLAY_NAMES.get(op, op), text)
            sheet.autofilter(2, 0, max(2, row - 1), 1 + width * 2)
            sheet.set_landscape()
            sheet.fit_to_pages(1, 0)
            sheet.set_margins(0.25, 0.25, 0.5, 0.5)
            total += row - 3
    return total


def run_roc_comparison(args):
    for path in (args.baseline_intra_dir, args.baseline_inter_dir):
        if not path.is_dir():
            raise SystemExit(f"B-group directory not found: {path}")
    a = load_comparison_groups(args.rocshmem_intra_dir, args.rocshmem_inter_dir, args.rocshmem_w, args.rocshmem_z, args.ops)
    b = load_comparison_groups(args.baseline_intra_dir, args.baseline_inter_dir, args.rocshmem_w, args.rocshmem_z, args.ops)
    for label, groups in ((args.label_a, a), (args.label_b, b)):
        if not any(groups.values()):
            raise SystemExit(f"No {args.ops} rocSHMEM measurement rows parsed for {label}")
    devices = detect_device_types((args.rocshmem_intra_dir, args.rocshmem_inter_dir, args.baseline_intra_dir, args.baseline_inter_dir))
    rows = build_roc_comparison(args.output, args.device_type or ', '.join(devices) or 'Unknown device', a, b, args.label_a, args.label_b, args.min_size, args.max_size)
    print(f"Operations: {args.ops}")
    print(f"Comparison: {args.label_a} (A) vs {args.label_b} (B); bandwidth change = B/A - 1")
    print(f"Sheets: {', '.join(a)}; comparison rows: {rows}")
    print(f"Output: {args.output.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compare BASE with NEW. BASE must be rocSHMEM; RCCL is allowed only as NEW. Directory pairs are INTRA then INTER.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s --base INTRA INTER --new INTRA INTER [--vs {rocshmem,rccl}] [--ops {coll,rma,all}] [--labels BASE_NAME NEW_NAME]\n       [--wgs N[,N...]] [--threads N[,N...]] [--rccl-placement {out,in}]\n       [--min-volume BYTES] [--max-volume BYTES] [--device-type NAME] [-o OUTPUT] [-h]",
        epilog=(
            "Examples:\n"
            "  %(prog)s --base old/IPC old/GDA --new new/IPC new/GDA\n"
            "  %(prog)s --base roc/IPC roc/GDA --new rccl/intra rccl/inter --vs rccl\n"
            "\n"
            "Notes:\n"
            "  - --base and --new are required; bracketed options are optional.\n"
            "  - BASE supports rocSHMEM only; RCCL or mixed inputs are rejected.\n"
            "  - --ops coll is the default; rma selects one-sided operations; all selects both.\n"
            "  - all lists collectives first; RCCL comparisons support coll only.\n"
            "  - Legacy directory options remain accepted but are hidden from help."
        ))
    inputs = parser.add_argument_group("Required inputs (--base and --new; directory order: INTRA INTER)")
    tuning = parser.add_argument_group("Device-side tuning (host-side ignores these filters)")
    report = parser.add_argument_group("Report options")
    legacy = parser.add_argument_group("Legacy individual directories (compatible with earlier commands)")
    report.add_argument("--ops", type=str.lower, choices=("coll", "rma", "all"), default="coll", help="Operations: coll=collectives (default), rma=one-sided, all=both; RCCL supports coll only")
    report.add_argument("--vs", dest="compare_with", choices=("rccl", "rocshmem"), help="NEW data type: rocshmem (default) or rccl; BASE supports rocSHMEM only")
    inputs.add_argument("--base", dest="a", nargs=2, type=Path, metavar=("INTRA", "INTER"), help="BASE directories: intra-node then inter-node; rocSHMEM only; RCCL input is rejected")
    inputs.add_argument("--new", dest="b", nargs=2, type=Path, metavar=("INTRA", "INTER"), help="NEW directories: intra-node then inter-node; parsed using --vs (default: rocshmem)")
    legacy.add_argument("--rccl", "-r", dest="r", nargs=2, type=Path, metavar=("INTRA", "INTER"), help=argparse.SUPPRESS)
    report.add_argument("--labels", nargs=2, metavar=("BASE_NAME", "NEW_NAME"), help="Optional BASE/NEW display names (RCCL mode: BASE/RCCL)")
    legacy.add_argument("--baseline-intra-dir", type=Path, help=argparse.SUPPRESS)
    legacy.add_argument("--baseline-inter-dir", type=Path, help=argparse.SUPPRESS)
    report.add_argument("--label-a", default=None, help=argparse.SUPPRESS)
    report.add_argument("--label-b", default=None, help=argparse.SUPPRESS)
    legacy.add_argument("--rocshmem-intra-dir", type=Path, help=argparse.SUPPRESS)
    legacy.add_argument("--rocshmem-inter-dir", type=Path, help=argparse.SUPPRESS)
    legacy.add_argument("--rccl-intra-dir", type=Path,
                        help=argparse.SUPPRESS)
    legacy.add_argument("--rccl-inter-dir", type=Path,
                        help=argparse.SUPPRESS)
    tuning.add_argument("--wgs", dest="rocshmem_w", metavar="N[,N...]", type=positive_int_list, help="Device-side workgroup counts (filename w), e.g. 16,32; one sheet per W/Z combination; ignored by host-side cases")
    tuning.add_argument("--threads", dest="rocshmem_z", metavar="N[,N...]", type=positive_int_list, help="Device-side threads per workgroup (filename z), e.g. 128,256; ignored by host-side cases; omitted values are unrestricted")
    report.add_argument("-o", "--output", type=Path, default=None, help="Output file name; .xlsx is added automatically when omitted")
    report.add_argument("--device-type", help="Override the automatically detected device type")
    report.add_argument("--rccl-placement", dest="rccl_place", choices=("out", "in"), default="out", help="RCCL result columns (default: out-of-place)")
    report.add_argument("--min-volume", dest="min_size", metavar="BYTES", type=int, default=256, help="Minimum volume in bytes (default: 256)")
    report.add_argument("--max-volume", dest="max_size", metavar="BYTES", type=int, help="Maximum volume in bytes (default: unlimited)")
    # Hidden aliases preserve old commands without cluttering usage.
    parser.add_argument("--with", "--compare-with", dest="compare_with", choices=("rccl", "rocshmem"), default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--rocshmem-a", "-a", dest="a", nargs=2, type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--rocshmem-b", "-b", dest="b", nargs=2, type=Path, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--w", "--rocshmem-w", dest="rocshmem_w", type=positive_int_list, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--z", "--rocshmem-z", dest="rocshmem_z", type=positive_int_list, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--rccl-place", dest="rccl_place", choices=("out", "in"), default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--min-size", dest="min_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument("--max-size", dest="max_size", type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    # Resolve the NEW role only after choosing its parser.
    has_rccl = args.r is not None or args.rccl_intra_dir is not None or args.rccl_inter_dir is not None
    args.compare_with = args.compare_with or ("rccl" if has_rccl else "rocshmem")
    for option, fields in (("a", ("rocshmem_intra_dir", "rocshmem_inter_dir")),
                           ("b", ("rccl_intra_dir", "rccl_inter_dir") if args.compare_with == "rccl" else ("baseline_intra_dir", "baseline_inter_dir")),
                           ("r", ("rccl_intra_dir", "rccl_inter_dir"))):
        pair = getattr(args, option)
        if pair is not None:
            for field, value in zip(fields, pair):
                old = getattr(args, field)
                if old is not None and old != value:
                    parser.error(f"{dict(a='--base', b='--new', r='--rccl')[option]} conflicts with --{field.replace('_', '-')}")
                setattr(args, field, value)
    if args.rocshmem_intra_dir is None or args.rocshmem_inter_dir is None:
        parser.error("--base INTRA INTER is required (or both legacy rocSHMEM directory options)")
    if args.compare_with == "rccl" and args.ops != "coll":
        parser.error("--vs rccl supports --ops coll only; rma/all require rocSHMEM vs rocSHMEM")
    if args.labels:
        if args.label_a is not None or args.label_b is not None:
            parser.error("Use --labels or --label-a/--label-b, not both")
        args.label_a, args.label_b = args.labels
    args.label_a = args.label_a or "BASE"
    args.label_b = args.label_b or ("RCCL" if args.compare_with == "rccl" else "NEW")
    required = ("baseline_intra_dir", "baseline_inter_dir") if args.compare_with == "rocshmem" else ("rccl_intra_dir", "rccl_inter_dir")
    for field in required:
        if getattr(args, field) is None:
            parser.error(f"--new INTRA INTER is required (NEW type: {args.compare_with}); legacy individual directory options are also accepted")
    if args.compare_with == "rccl" and (args.baseline_intra_dir or args.baseline_inter_dir):
        parser.error("Baseline directories require --vs rocshmem")
    if args.compare_with == "rocshmem" and (args.rccl_intra_dir or args.rccl_inter_dir):
        parser.error("RCCL directories are only valid with --vs rccl")
    if args.min_size < 0 or (args.max_size is not None and args.max_size < args.min_size):
        parser.error("Require 0 <= min-volume <= max-volume")
    args.output = args.output or Path(f"rocshmem_vs_{args.compare_with}")
    return args


def reject_rccl_base(roots):
    """Reject RCCL inputs by the rccl-tests banner, with legacy filename/header fallbacks."""
    seen = set()
    for root in roots:
        for path in root.rglob("*.log"):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found = rccl_op_from_filename(path) is not None
            if not found:
                try:
                    with path.open("r", encoding="utf-8", errors="replace") as stream:
                        saw_out_place = saw_in_place = False
                        for line in stream:
                            stripped = line.strip()
                            # Prefer the rccl-tests version banner over table layout heuristics.
                            if re.match(r"^#?\s*rccl-tests\b", stripped, re.IGNORECASE):
                                found = True
                                break
                            saw_out_place |= bool(re.search(r"\bout[- ]of[- ]place\b", stripped, re.IGNORECASE))
                            saw_in_place |= bool(re.search(r"\bin[- ]place\b", stripped, re.IGNORECASE))
                            if saw_out_place and saw_in_place:
                                found = True
                                break
                            test = re.match(r"^(?:测试|Test)\s*[:：]\s*([\w.-]+)", stripped, re.IGNORECASE)
                            if test and test.group(1).lower().endswith("_perf"):
                                found = True
                                break
                            tokens = set(stripped.lower().lstrip("#").split())
                            if {"size", "count", "type", "algbw", "busbw"}.issubset(tokens):
                                found = True
                                break
                except OSError as exc:
                    raise SystemExit(f"Cannot inspect BASE log {path}: {exc}")
            if found:
                raise SystemExit(f"Error: --base accepts rocSHMEM only; detected RCCL log: {path}. "
                                 "Move RCCL inputs to --new INTRA INTER and use --vs rccl.")


def main():
    args = parse_args()
    if args.output.suffix.lower() != ".xlsx":
        args.output = args.output.with_name(args.output.name + ".xlsx")
    current_dir = Path.cwd()
    args.rocshmem_intra_dir = args.rocshmem_intra_dir or find_default_directory("IPC", current_dir)
    args.rocshmem_inter_dir = args.rocshmem_inter_dir or find_default_directory("GDA", current_dir)
    if not args.rocshmem_intra_dir.is_dir():
        raise SystemExit(f"Intra-node rocSHMEM directory not found: {args.rocshmem_intra_dir}")
    if not args.rocshmem_inter_dir.is_dir():
        raise SystemExit(f"Inter-node rocSHMEM directory not found: {args.rocshmem_inter_dir}")
    reject_rccl_base((args.rocshmem_intra_dir, args.rocshmem_inter_dir))
    if args.compare_with == "rocshmem":
        run_roc_comparison(args)
        return
    if not args.rccl_intra_dir.is_dir():
        raise SystemExit(f"Intra-node RCCL directory not found: {args.rccl_intra_dir}")
    if not args.rccl_inter_dir.is_dir():
        raise SystemExit(f"Inter-node RCCL directory not found: {args.rccl_inter_dir}")
    device_groups = {}
    roc = {}
    for w, z in product(args.rocshmem_w or [None], args.rocshmem_z or [None]):
        points = parse_rocshmem(args.rocshmem_intra_dir, "intranode", w, z)
        points.update(parse_rocshmem(args.rocshmem_inter_dir, "internode", w, z))
        roc.update(points)
        name = device_sheet_name(w, z)
        device_groups[name] = {key: point for key, point in points.items() if key[1] in OP_ORDER}
    rccl = parse_rccl(args.rccl_intra_dir, "intranode", args.rccl_place)
    rccl.update(parse_rccl(args.rccl_inter_dir, "internode", args.rccl_place))
    if not roc:
        raise SystemExit("No rocSHMEM measurement rows were parsed")
    if not rccl:
        raise SystemExit("No mapped RCCL measurement rows were parsed")
    device_types = detect_device_types((
        args.rocshmem_intra_dir, args.rocshmem_inter_dir,
        args.rccl_intra_dir, args.rccl_inter_dir,
    ))
    device_type = args.device_type or ", ".join(device_types) or "Unknown device"
    rows = build_workbook(args.output, device_type, roc, rccl, args.min_size, args.max_size, device_groups=device_groups, label_a=args.label_a, label_b=args.label_b)
    print(f"rocSHMEM points: {len(roc)}")
    print(f"RCCL points: {len(rccl)}")
    print(f"device types: {', '.join(device_types) if device_types else '(not found)'}")
    print(f"rocSHMEM device-side W/Z filter (host-side on-stream unfiltered): w={args.rocshmem_w or '*'}, z={args.rocshmem_z or '*'}")
    print(f"device-side sheets: {', '.join(device_groups)}")
    print(f"comparison rows: {rows}")
    print(f"output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
