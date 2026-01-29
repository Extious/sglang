#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import shutil
import signal
import socket
import sys
import textwrap
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence


SECTION_GPU = "__SGTOP_GPU__"
SECTION_PROC = "__SGTOP_PROC__"

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


@dataclass(frozen=True)
class NodeInfo:
    name: str
    state: str | None = None
    gres_gpu_type: str | None = None
    gres_gpu_count: int | None = None
    partitions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProcessInfo:
    gpu_uuid: str
    pid: int
    name: str
    used_memory_mib: float | None


@dataclass(frozen=True)
class GpuInfo:
    index: int
    uuid: str
    name: str
    memory_total_mib: float | None
    memory_used_mib: float | None
    util_gpu_pct: float | None
    util_mem_pct: float | None
    temperature_c: float | None
    power_draw_w: float | None
    power_limit_w: float | None
    processes: tuple[ProcessInfo, ...] = ()


@dataclass(frozen=True)
class NodeResult:
    node: NodeInfo
    gpus: tuple[GpuInfo, ...]
    error: str | None = None
    duration_s: float | None = None


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def visible_len(text: str) -> int:
    return len(strip_ansi(text))


def pad(text: str, width: int, *, align: str = "left") -> str:
    current = visible_len(text)
    if current >= width:
        return text
    padding = " " * (width - current)
    if align == "right":
        return padding + text
    return text + padding


def truncate(text: str, width: int) -> str:
    if visible_len(text) <= width:
        return text
    if width <= 1:
        return "…"
    clean = strip_ansi(text)
    return clean[: max(0, width - 1)] + "…"


class Ansi:
    reset = "\x1b[0m"
    bold = "\x1b[1m"
    dim = "\x1b[2m"

    red = "\x1b[31m"
    green = "\x1b[32m"
    yellow = "\x1b[33m"
    blue = "\x1b[34m"
    magenta = "\x1b[35m"
    cyan = "\x1b[36m"
    gray = "\x1b[90m"


def colorize(text: str, color: str, *, enabled: bool) -> str:
    if not enabled:
        return text
    return f"{color}{text}{Ansi.reset}"


def severity_color(pct: float | None) -> str:
    if pct is None:
        return Ansi.gray
    if pct >= 90:
        return Ansi.red
    if pct >= 70:
        return Ansi.yellow
    return Ansi.green


def format_bar(pct: float | None, width: int, *, ascii_only: bool) -> str:
    if pct is None:
        return "[" + ("?" * width) + "]"
    pct = max(0.0, min(100.0, pct))
    filled = int(round((pct / 100.0) * width))
    filled = max(0, min(width, filled))
    if ascii_only:
        fill_char = "="
        empty_char = "."
    else:
        fill_char = "█"
        empty_char = "░"
    return "[" + (fill_char * filled) + (empty_char * (width - filled)) + "]"


def safe_float(value: str) -> float | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def safe_int(value: str) -> int | None:
    value = value.strip()
    if not value or value.upper() == "N/A":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_scontrol_nodes(output: str) -> list[NodeInfo]:
    nodes: list[NodeInfo] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = dict(field.split("=", 1) for field in line.split() if "=" in field)
        node_name = parts.get("NodeName")
        if not node_name:
            continue
        state = parts.get("State")

        gres = parts.get("Gres") or ""
        gpu_type, gpu_count = parse_slurm_gres_gpu(gres)
        partitions = parse_slurm_partitions(parts.get("Partitions") or "")
        nodes.append(
            NodeInfo(
                name=node_name,
                state=state,
                gres_gpu_type=gpu_type,
                gres_gpu_count=gpu_count,
                partitions=partitions,
            )
        )
    return nodes


def parse_slurm_gres_gpu(gres: str) -> tuple[str | None, int | None]:
    # Examples:
    # - gpu:rtx4090:2,gpu_mem:24564
    # - gpu:8
    # - gpu:a100:4
    for entry in gres.split(","):
        entry = entry.strip()
        if not entry.startswith("gpu:"):
            continue
        fields = entry.split(":")
        if len(fields) == 2:
            return None, safe_int(fields[1])
        if len(fields) >= 3:
            return fields[1] or None, safe_int(fields[2])
    return None, None


def parse_slurm_partitions(value: str) -> tuple[str, ...]:
    value = value.strip()
    if not value or value == "(null)":
        return ()
    parts = [part.strip() for part in value.split(",") if part.strip()]
    return tuple(parts)


def srun_extra_has_partition(extra: Sequence[str]) -> bool:
    for arg in extra:
        if arg in ("-p", "--partition"):
            return True
        if arg.startswith("-p") and len(arg) > 2:
            return True
        if arg.startswith("--partition="):
            return True
    return False


def srun_extra_has_immediate(extra: Sequence[str]) -> bool:
    for arg in extra:
        if arg in ("-I", "--immediate"):
            return True
        if arg.startswith("-I") and len(arg) > 2:
            return True
        if arg.startswith("--immediate="):
            return True
    return False


def srun_extra_has_job_name(extra: Sequence[str]) -> bool:
    for arg in extra:
        if arg in ("-J", "--job-name"):
            return True
        if arg.startswith("-J") and len(arg) > 2:
            return True
        if arg.startswith("--job-name="):
            return True
    return False


def srun_extra_has_time_limit(extra: Sequence[str]) -> bool:
    for arg in extra:
        if arg in ("-t", "--time"):
            return True
        if arg.startswith("-t") and len(arg) > 2:
            return True
        if arg.startswith("--time="):
            return True
    return False


def build_nvidia_query_script(*, include_processes: bool) -> str:
    gpu_query_fields = ",".join(
        [
            "index",
            "uuid",
            "name",
            "memory.total",
            "memory.used",
            "utilization.gpu",
            "utilization.memory",
            "temperature.gpu",
            "power.draw",
            "power.limit",
        ]
    )
    proc_query_fields = ",".join(
        [
            "gpu_uuid",
            "pid",
            "process_name",
            "used_memory",
        ]
    )

    lines = [
        "set -euo pipefail",
        "if ! command -v nvidia-smi >/dev/null 2>&1; then",
        "  echo 'nvidia-smi not found' >&2",
        "  exit 127",
        "fi",
        f"echo {SECTION_GPU}",
        f"nvidia-smi --query-gpu={gpu_query_fields} --format=csv,noheader,nounits",
    ]
    if include_processes:
        lines += [
            f"echo {SECTION_PROC}",
            f"nvidia-smi --query-compute-apps={proc_query_fields} --format=csv,noheader,nounits || true",
        ]
    else:
        lines += [
            f"echo {SECTION_PROC}",
        ]
    return "\n".join(lines) + "\n"


def split_sections(stdout: str) -> tuple[str, str]:
    gpu_marker = f"{SECTION_GPU}\n"
    proc_marker = f"{SECTION_PROC}\n"
    if gpu_marker not in stdout:
        raise ValueError("missing GPU section marker")
    before, after_gpu = stdout.split(gpu_marker, 1)
    _ = before  # unused
    if proc_marker not in after_gpu:
        raise ValueError("missing PROC section marker")
    gpu_section, proc_section = after_gpu.split(proc_marker, 1)
    return gpu_section.strip(), proc_section.strip()


def parse_csv_rows(text: str) -> list[list[str]]:
    if not text.strip():
        return []
    reader = csv.reader(text.splitlines(), skipinitialspace=True)
    rows: list[list[str]] = []
    for row in reader:
        if not row:
            continue
        rows.append(row)
    return rows


def parse_nvidia_sections(
    gpu_section: str, proc_section: str
) -> tuple[list[GpuInfo], dict[str, list[ProcessInfo]]]:
    gpu_rows = parse_csv_rows(gpu_section)
    proc_rows = parse_csv_rows(proc_section)

    processes_by_uuid: dict[str, list[ProcessInfo]] = defaultdict(list)
    for row in proc_rows:
        if len(row) < 4:
            continue
        gpu_uuid, pid_raw, name, used_mem_raw = row[:4]
        pid = safe_int(pid_raw)
        if pid is None:
            continue
        proc = ProcessInfo(
            gpu_uuid=gpu_uuid.strip(),
            pid=pid,
            name=name.strip(),
            used_memory_mib=safe_float(used_mem_raw),
        )
        processes_by_uuid[proc.gpu_uuid].append(proc)

    for gpu_uuid, proc_list in processes_by_uuid.items():
        proc_list.sort(key=lambda proc: proc.used_memory_mib or 0.0, reverse=True)

    gpus: list[GpuInfo] = []
    for row in gpu_rows:
        if len(row) < 10:
            continue
        index_raw, uuid, name = row[0:3]
        index = safe_int(index_raw)
        if index is None:
            continue
        memory_total = safe_float(row[3])
        memory_used = safe_float(row[4])
        util_gpu = safe_float(row[5])
        util_mem = safe_float(row[6])
        temperature = safe_float(row[7])
        power_draw = safe_float(row[8])
        power_limit = safe_float(row[9])
        procs = tuple(processes_by_uuid.get(uuid.strip(), []))
        gpus.append(
            GpuInfo(
                index=index,
                uuid=uuid.strip(),
                name=name.strip(),
                memory_total_mib=memory_total,
                memory_used_mib=memory_used,
                util_gpu_pct=util_gpu,
                util_mem_pct=util_mem,
                temperature_c=temperature,
                power_draw_w=power_draw,
                power_limit_w=power_limit,
                processes=procs,
            )
        )
    return gpus, processes_by_uuid


async def run_subprocess(
    argv: Sequence[str],
    *,
    timeout_s: float,
    input_text: str | None = None,
) -> tuple[int, str, str]:
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if input_text is not None else None,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input=input_text.encode() if input_text is not None else None),
            timeout=timeout_s,
        )
    except TimeoutError:
        process.terminate()
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=2.0)
        except TimeoutError:
            process.kill()
            stdout_bytes, stderr_bytes = await process.communicate()
        return 124, stdout_bytes.decode(errors="replace"), stderr_bytes.decode(errors="replace")
    return (
        process.returncode or 0,
        stdout_bytes.decode(errors="replace"),
        stderr_bytes.decode(errors="replace"),
    )


def is_command_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def choose_runner(requested: str) -> str:
    if requested != "auto":
        return requested
    if is_command_available("srun"):
        return "srun"
    if is_command_available("ssh"):
        return "ssh"
    return "local"


def is_local_node(node: str) -> bool:
    local = socket.gethostname()
    if node == local:
        return True
    # Some clusters use short hostnames in slurm but fqdn in hostname.
    return local.split(".", 1)[0] == node.split(".", 1)[0]


async def fetch_node(
    node: NodeInfo,
    *,
    runner: str,
    timeout_s: float,
    include_processes: bool,
    partition_override: str | None,
    srun_immediate_s: int,
    srun_extra: Sequence[str],
    ssh_extra: Sequence[str],
) -> NodeResult:
    start = time.monotonic()
    script = build_nvidia_query_script(include_processes=include_processes)

    if runner == "local" or (runner == "auto" and is_local_node(node.name)):
        argv = ["bash", "-lc", script]
        rc, stdout, stderr = await run_subprocess(argv, timeout_s=timeout_s)
    elif runner == "srun":
        partition = partition_override
        if partition is None and node.partitions:
            partition = node.partitions[0]

        srun_defaults: list[str] = []
        if partition and not srun_extra_has_partition(srun_extra):
            srun_defaults += ["-p", partition]
        if not srun_extra_has_job_name(srun_extra):
            srun_defaults += ["-J", "cluster_gputop"]
        if srun_immediate_s > 0 and not srun_extra_has_immediate(srun_extra):
            srun_defaults += ["-I", str(srun_immediate_s)]
        if not srun_extra_has_time_limit(srun_extra):
            srun_defaults += ["-t", "1"]

        argv = [
            "srun",
            "--quiet",
            "-N",
            "1",
            "-w",
            node.name,
            "--ntasks=1",
            *srun_defaults,
            *srun_extra,
            "bash",
            "-lc",
            script,
        ]
        rc, stdout, stderr = await run_subprocess(argv, timeout_s=timeout_s)
    elif runner == "ssh":
        argv = [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            f"ConnectTimeout={max(1, int(timeout_s))}",
            *ssh_extra,
            node.name,
            "bash",
            "-s",
        ]
        rc, stdout, stderr = await run_subprocess(argv, timeout_s=timeout_s, input_text=script)
    else:
        return NodeResult(node=node, gpus=(), error=f"unknown runner: {runner}")

    duration_s = time.monotonic() - start

    if rc != 0:
        error = (stderr or stdout).strip() or f"command failed with rc={rc}"
        error = " ".join(error.splitlines()[:3])
        return NodeResult(node=node, gpus=(), error=error, duration_s=duration_s)

    try:
        gpu_section, proc_section = split_sections(stdout)
        gpus, _ = parse_nvidia_sections(gpu_section, proc_section)
    except Exception as exc:  # noqa: BLE001
        raw = (stdout + "\n" + stderr).strip()
        raw = " ".join(raw.splitlines()[:3])
        return NodeResult(
            node=node,
            gpus=(),
            error=f"parse error: {exc}; raw={raw}",
            duration_s=duration_s,
        )

    return NodeResult(node=node, gpus=tuple(gpus), error=None, duration_s=duration_s)


async def discover_slurm_nodes(*, include_cpu_only: bool) -> list[NodeInfo]:
    if not is_command_available("scontrol"):
        return []
    rc, stdout, stderr = await run_subprocess(
        ["scontrol", "show", "nodes", "-o"],
        timeout_s=5.0,
    )
    if rc != 0:
        raise RuntimeError((stderr or stdout).strip() or "failed to run scontrol show nodes")
    nodes = parse_scontrol_nodes(stdout)
    if include_cpu_only:
        return nodes
    return [node for node in nodes if (node.gres_gpu_count or 0) > 0 or node.gres_gpu_type]


def parse_nodes_arg(value: str) -> list[str]:
    nodes: list[str] = []
    for part in value.split(","):
        node = part.strip()
        if node:
            nodes.append(node)
    return nodes


def read_node_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        nodes: list[str] = []
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            nodes.extend(parse_nodes_arg(line))
        return nodes


def format_pct(value: float | None) -> str:
    if value is None:
        return "  ?%"
    return f"{value:3.0f}%"


def format_mem(used_mib: float | None, total_mib: float | None) -> tuple[str, float | None]:
    if used_mib is None or total_mib is None or total_mib <= 0:
        return "?/?GiB", None
    used_gib = used_mib / 1024.0
    total_gib = total_mib / 1024.0
    pct = (used_mib / total_mib) * 100.0
    return f"{used_gib:4.1f}/{total_gib:4.1f}GiB", pct


def format_power(draw_w: float | None, limit_w: float | None) -> str:
    if draw_w is None and limit_w is None:
        return " ?W"
    if draw_w is None:
        return f" ?/{limit_w:.0f}W"
    if limit_w is None:
        return f"{draw_w:3.0f}W"
    return f"{draw_w:3.0f}/{limit_w:.0f}W"


def summarize_processes(processes: Sequence[ProcessInfo], *, top_procs: int) -> str:
    if not processes:
        return "0"
    total = len(processes)
    parts: list[str] = []
    shown = processes if top_procs <= 0 else processes[:top_procs]
    for proc in shown:
        mem = "?"
        if proc.used_memory_mib is not None:
            mem = f"{proc.used_memory_mib:.0f}MiB"
        name = os.path.basename(proc.name) if "/" in proc.name else proc.name
        parts.append(f"{proc.pid}:{name}:{mem}")
    suffix = ""
    if top_procs > 0 and total > top_procs:
        suffix = f" +{total - top_procs}"
    return f"{total} " + " ".join(parts) + suffix


def render(
    results: Sequence[NodeResult],
    *,
    runner: str,
    interval_s: float,
    show_process_lines: bool,
    top_procs: int,
    color: bool,
    ascii_bars: bool,
    bar_width: int,
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    node_count = len(results)
    gpu_count = sum(len(result.gpus) for result in results)

    header = f"cluster_gputop  {now}  interval={interval_s}s  runner={runner}  nodes={node_count}  gpus={gpu_count}"
    lines = [header, ""]

    term_width = shutil.get_terminal_size(fallback=(160, 40)).columns

    include_power = term_width >= 110
    include_temp = term_width >= 100
    include_procs = term_width >= 140

    headers = ["NODE", "STATE", "GPU", "NAME", "MEM", "GPU%", "MEM%"]
    if include_temp:
        headers.append("TEMP")
    if include_power:
        headers.append("PWR")
    if include_procs and not show_process_lines:
        headers.append("PROCS")

    rows: list[list[str]] = []
    extra_lines: list[str] = []

    for result in sorted(results, key=lambda r: r.node.name):
        node_label = result.node.name
        state = result.node.state or "-"

        if result.error:
            error_text = colorize("ERR", Ansi.red, enabled=color)
            rows.append([node_label, state, "-", error_text, "-", "-", "-"] + (["-"] if include_temp else []) + (["-"] if include_power else []) + (["-"] if (include_procs and not show_process_lines) else []))
            extra_lines.append(
                colorize(
                    f"{node_label}: {result.error}",
                    Ansi.red,
                    enabled=color,
                )
            )
            continue

        if not result.gpus:
            rows.append([node_label, state, "-", "no gpus", "-", "-", "-"] + (["-"] if include_temp else []) + (["-"] if include_power else []) + (["-"] if (include_procs and not show_process_lines) else []))
            continue

        first = True
        for gpu in sorted(result.gpus, key=lambda g: g.index):
            name = gpu.name
            mem_text, mem_pct = format_mem(gpu.memory_used_mib, gpu.memory_total_mib)
            mem_bar = format_bar(mem_pct, bar_width, ascii_only=ascii_bars)
            util_bar = format_bar(gpu.util_gpu_pct, bar_width, ascii_only=ascii_bars)

            mem_pct_text = format_pct(mem_pct)
            gpu_pct_text = format_pct(gpu.util_gpu_pct)
            mem_pct_colored = colorize(mem_pct_text, severity_color(mem_pct), enabled=color)
            gpu_pct_colored = colorize(gpu_pct_text, severity_color(gpu.util_gpu_pct), enabled=color)

            row = [
                node_label if first else "",
                state if first else "",
                str(gpu.index),
                name,
                f"{mem_text} {mem_bar}",
                f"{gpu_pct_colored} {util_bar}",
                mem_pct_colored,
            ]
            if include_temp:
                temp_text = " ?C" if gpu.temperature_c is None else f"{gpu.temperature_c:3.0f}C"
                row.append(colorize(temp_text, severity_color(gpu.temperature_c), enabled=color))
            if include_power:
                row.append(format_power(gpu.power_draw_w, gpu.power_limit_w))
            if include_procs and not show_process_lines:
                row.append(summarize_processes(gpu.processes, top_procs=top_procs))
            rows.append(row)

            if show_process_lines and gpu.processes:
                shown = gpu.processes if top_procs <= 0 else gpu.processes[:top_procs]
                for proc in shown:
                    mem = "?"
                    if proc.used_memory_mib is not None:
                        mem = f"{proc.used_memory_mib:.0f}MiB"
                    name_only = proc.name
                    extra_lines.append(f"  {node_label} GPU{gpu.index}  PID {proc.pid:<7} {mem:>7}  {name_only}")
                if top_procs > 0 and len(gpu.processes) > top_procs:
                    extra_lines.append(
                        f"  {node_label} GPU{gpu.index}  ... (+{len(gpu.processes) - top_procs} more)"
                    )
            first = False

    # Compute column widths with caps on wide fields.
    max_widths: dict[int, int] = {}
    for idx, header_cell in enumerate(headers):
        max_widths[idx] = max(8, len(header_cell))
    for row in rows:
        for idx, cell in enumerate(row):
            max_widths[idx] = max(max_widths.get(idx, 0), visible_len(cell))

    # Cap some columns to keep table readable.
    name_col = headers.index("NAME") if "NAME" in headers else None
    mem_col = headers.index("MEM") if "MEM" in headers else None
    procs_col = headers.index("PROCS") if "PROCS" in headers else None
    if name_col is not None:
        max_widths[name_col] = min(max_widths[name_col], 32)
    if mem_col is not None:
        max_widths[mem_col] = min(max_widths[mem_col], 32 + bar_width)
    if procs_col is not None:
        max_widths[procs_col] = min(max_widths[procs_col], 50)

    col_widths = [max_widths[i] for i in range(len(headers))]
    separator = "  "
    table_width = sum(col_widths) + len(separator) * (len(headers) - 1)

    if table_width > term_width:
        # If still too wide, aggressively shrink NAME/PROCS.
        if name_col is not None:
            col_widths[name_col] = max(12, col_widths[name_col] - (table_width - term_width))
        if procs_col is not None and sum(col_widths) + len(separator) * (len(headers) - 1) > term_width:
            overflow = (sum(col_widths) + len(separator) * (len(headers) - 1)) - term_width
            col_widths[procs_col] = max(12, col_widths[procs_col] - overflow)

    header_line = separator.join(pad(h, col_widths[i]) for i, h in enumerate(headers))
    rule_line = separator.join("-" * col_widths[i] for i in range(len(headers)))
    lines.extend([header_line, rule_line])

    for row in rows:
        rendered_cells: list[str] = []
        for idx, cell in enumerate(row):
            cell_text = truncate(cell, col_widths[idx])
            rendered_cells.append(pad(cell_text, col_widths[idx]))
        lines.append(separator.join(rendered_cells))

    if extra_lines:
        lines.append("")
        lines.append(colorize("Errors / Processes:", Ansi.bold, enabled=color))
        for line in extra_lines[: min(len(extra_lines), 200)]:
            if visible_len(line) > term_width:
                lines.append(truncate(line, term_width))
            else:
                lines.append(line)

    return "\n".join(lines) + "\n"


def clear_screen() -> None:
    sys.stdout.write("\x1b[2J\x1b[H")


def hide_cursor() -> None:
    sys.stdout.write("\x1b[?25l")


def show_cursor() -> None:
    sys.stdout.write("\x1b[?25h")


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cluster_gputop.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(
            """\
            在终端里查看 Slurm 集群各节点 GPU 状态（类似 nvitop 的概览）。

            默认会从 Slurm 自动发现带 GPU 的节点，并使用 srun 远程执行 nvidia-smi。
            """
        ),
    )

    node_group = parser.add_mutually_exclusive_group()
    node_group.add_argument("--nodes", help="指定节点列表（逗号分隔），如 gpu10,gpu11")
    node_group.add_argument("--node-file", help="从文件读取节点列表（每行一个，支持逗号）")

    parser.add_argument(
        "--include-cpu-only",
        action="store_true",
        help="Slurm 自动发现时也包含非 GPU 节点",
    )
    parser.add_argument(
        "--runner",
        choices=["auto", "local", "ssh", "srun"],
        default="auto",
        help="远程执行方式（默认 auto）",
    )
    parser.add_argument(
        "--partition",
        default="",
        help="srun 使用的分区（覆盖自动从节点信息推断；为空表示自动）",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="刷新间隔秒数（默认 2）")
    parser.add_argument("--once", action="store_true", help="只输出一次后退出")
    parser.add_argument("--timeout", type=float, default=12.0, help="单节点超时秒数（默认 12）")
    parser.add_argument("--concurrency", type=int, default=8, help="并发查询节点数（默认 8）")

    parser.add_argument(
        "--no-processes",
        action="store_true",
        help="不查询进程（更快）",
    )
    parser.add_argument(
        "--show-process-lines",
        action="store_true",
        help="像 nvitop 一样把进程明细按行打印（会占更多行）",
    )
    parser.add_argument("--top-procs", type=int, default=3, help="每张卡最多显示的进程数（默认 3）")

    parser.add_argument("--no-color", action="store_true", help="禁用 ANSI 颜色")
    parser.add_argument("--ascii-bars", action="store_true", help="用 ASCII 字符绘制进度条（兼容性更好）")
    parser.add_argument("--bar-width", type=int, default=10, help="进度条宽度（默认 10）")

    parser.add_argument(
        "--srun-immediate",
        type=float,
        default=0.0,
        help="srun -I/--immediate 等待秒数（0 表示不加；默认 0）",
    )
    parser.add_argument(
        "--srun-extra",
        default="",
        help="额外传给 srun 的参数（字符串，会按 shell 风格拆分）",
    )
    parser.add_argument(
        "--ssh-extra",
        default="",
        help="额外传给 ssh 的参数（字符串，会按 shell 风格拆分）",
    )

    return parser.parse_args(list(argv))


def shell_split(value: str) -> list[str]:
    # We avoid importing shlex globally for startup speed; it's stdlib anyway.
    import shlex

    return shlex.split(value) if value.strip() else []


async def main_async(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    color = not args.no_color and sys.stdout.isatty()
    include_processes = not args.no_processes
    runner = choose_runner(args.runner)

    partition_override = args.partition.strip() or None
    srun_immediate_s = int(max(0.0, float(args.srun_immediate)))

    slurm_nodes: list[NodeInfo] = []
    if is_command_available("scontrol"):
        try:
            slurm_nodes = await discover_slurm_nodes(include_cpu_only=True)
        except Exception:
            slurm_nodes = []
    slurm_by_name = {node.name: node for node in slurm_nodes}

    if args.nodes:
        requested = parse_nodes_arg(args.nodes)
        nodes = [slurm_by_name.get(node, NodeInfo(name=node)) for node in requested]
    elif args.node_file:
        requested = read_node_file(args.node_file)
        nodes = [slurm_by_name.get(node, NodeInfo(name=node)) for node in requested]
    else:
        nodes = slurm_nodes
        if not args.include_cpu_only:
            nodes = [
                node
                for node in nodes
                if (node.gres_gpu_count or 0) > 0 or node.gres_gpu_type
            ]

    if not nodes:
        raise SystemExit("No nodes found. Use --nodes or ensure Slurm is available.")

    sem = asyncio.Semaphore(max(1, int(args.concurrency)))
    srun_extra = shell_split(args.srun_extra)
    ssh_extra = shell_split(args.ssh_extra)

    async def one(node: NodeInfo) -> NodeResult:
        async with sem:
            effective_runner = "local" if is_local_node(node.name) else runner
            return await fetch_node(
                node,
                runner=effective_runner,
                timeout_s=float(args.timeout),
                include_processes=include_processes,
                partition_override=partition_override,
                srun_immediate_s=srun_immediate_s,
                srun_extra=srun_extra,
                ssh_extra=ssh_extra,
            )

    stop = asyncio.Event()

    def on_signal(_signum: int, _frame) -> None:  # type: ignore[override]
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    if sys.stdout.isatty():
        hide_cursor()

    try:
        while True:
            start = time.monotonic()
            results = await asyncio.gather(*(one(node) for node in nodes))

            if sys.stdout.isatty():
                clear_screen()
            output = render(
                results,
                runner=runner,
                interval_s=float(args.interval),
                show_process_lines=bool(args.show_process_lines),
                top_procs=int(args.top_procs),
                color=color,
                ascii_bars=bool(args.ascii_bars),
                bar_width=max(3, int(args.bar_width)),
            )
            sys.stdout.write(output)
            sys.stdout.flush()

            if args.once:
                return 0

            elapsed = time.monotonic() - start
            sleep_s = max(0.0, float(args.interval) - elapsed)

            try:
                await asyncio.wait_for(stop.wait(), timeout=sleep_s)
                return 0
            except TimeoutError:
                continue
    finally:
        if sys.stdout.isatty():
            show_cursor()


def main(argv: Sequence[str]) -> int:
    try:
        return asyncio.run(main_async(argv))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
