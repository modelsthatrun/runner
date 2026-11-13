"""Host memory and CPU state for CPU rows (docs/protocol.md, "CPU rows"), read from /proc and /sys with the standard
library only, so the tool's dependencies, and every model's lock, stay as they are.

A CPU row's memory is the peak resident set (VmHWM) of the process that ran the model, as the kernel keeps it:
loading included, and for llama-server the mapped GGUF file's pages too, because they sit in RAM while it runs.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

KIB = 1024


def proc_status(pid: int | str = "self") -> dict[str, int]:
    """The memory lines of /proc/<pid>/status, in KiB: VmHWM, VmRSS, VmSwap, RssAnon, RssFile, …"""
    out: dict[str, int] = {}
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        m = re.match(r"^(Vm\w+|Rss\w+):\s+(\d+) kB", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def peak_mib(status: dict[str, int]) -> int:
    return round(status.get("VmHWM", 0) / KIB)


def meminfo() -> dict[str, int]:
    """/proc/meminfo, in KiB."""
    out: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        m = re.match(r"^(\w+):\s+(\d+)", line)
        if m:
            out[m.group(1)] = int(m.group(2))
    return out


def available_mib() -> int:
    return round(meminfo().get("MemAvailable", 0) / KIB)


def total_mib() -> int:
    return round(meminfo().get("MemTotal", 0) / KIB)


def cores() -> tuple[int, int]:
    """Physical cores and logical CPUs, from /proc/cpuinfo (one core per physical id and core id)."""
    physical: set[tuple[str, str]] = set()
    logical = 0
    pid = cid = None
    for line in Path("/proc/cpuinfo").read_text().splitlines() + [""]:
        if line.startswith("processor"):
            logical += 1
        elif line.startswith("physical id"):
            pid = line.split(":", 1)[1].strip()
        elif line.startswith("core id"):
            cid = line.split(":", 1)[1].strip()
        elif not line.strip():
            if cid is not None:
                physical.add((pid or "0", cid))
            pid = cid = None
    n = os.cpu_count() or logical or 1
    return (len(physical) or n), (logical or n)


def cpu_list(text: str) -> set[int]:
    """A kernel CPU list such as "0,8" or "0-3,8-11" as a set of logical CPUs."""
    out: set[int] = set()
    for part in text.strip().split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out.update(range(int(lo), int(hi) + 1))
        elif part:
            out.add(int(part))
    return out


def physical_cores() -> list[set[int]]:
    """The logical CPUs of each physical core (its hyperthread siblings), lowest first."""
    seen: dict[frozenset[int], None] = {}
    for f in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"):
        try:
            seen[frozenset(cpu_list(Path(f).read_text()))] = None
        except (OSError, ValueError):
            pass
    return sorted((set(c) for c in seen), key=min)


def throttle_count() -> int | None:
    """How many times the CPU's cores and package have throttled for heat since boot, or None where Linux doesn't say."""
    files = glob.glob("/sys/devices/system/cpu/cpu[0-9]*/thermal_throttle/*_throttle_count")
    if not files:
        return None
    total = 0
    for f in files:
        try:
            total += int(Path(f).read_text().strip())
        except (OSError, ValueError):
            pass
    return total


def power_limit_w() -> float | None:
    """The CPU package's sustained power limit (Intel RAPL long term, PL1), or None where it can't be read."""
    for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:[0-9]")):
        for i in range(4):
            name = Path(d, f"constraint_{i}_name")
            if name.is_file() and name.read_text().strip() == "long_term":
                try:
                    return int(Path(d, f"constraint_{i}_power_limit_uw").read_text()) / 1e6
                except (OSError, ValueError):
                    return None
    return None


def descendants(pid: int) -> set[int]:
    """pid and every process below it, from /proc."""
    children: dict[int, list[int]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            ppid = int(Path(d, "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            continue
        children.setdefault(ppid, []).append(int(d.name))
    out, todo = set(), [pid]
    while todo:
        p = todo.pop()
        if p in out:
            continue
        out.add(p)
        todo += children.get(p, [])
    return out
