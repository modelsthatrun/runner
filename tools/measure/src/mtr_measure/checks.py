"""Checks around a run (docs/protocol.md, "Machine").

A failed check stops the run. With --force the run goes ahead as a dry run, and no run record
is written, because its numbers were not taken under the protocol.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from . import machine
from .gpu import MIB, GpuProcess, Samples, process_name

# Programs a pause hook can stop for a timed run (desktop.py), by hook name. Matched on the process
# name. Only the hooks in machine.toml count: elsewhere these are ordinary programs.
PAUSABLE_PROGRAMS = {"sunshine": "Sunshine streaming host", "steam": "Steam"}

# A desktop on the measuring GPU holds some memory while idle: about 0.8 GiB on the reference machine
# (Xorg, gnome-shell and a few apps), more with a browser open. Above this, it's worth knowing what holds it.
BASELINE_WARN_MIB = 2048
# The idle baseline before and after the run may differ by this much before the peak is suspect.
DRIFT_LIMIT_MIB = 64
# Device-level used memory may move by this much within the one-second baseline window.
SPREAD_LIMIT_MIB = 64
# Mean GPU utilization while idle. A video or game in the background shows up here.
IDLE_UTIL_LIMIT = 10
# The protocol polls at 10 ms or less.
MAX_POLL_GAP_MS = 10.0
# Other programs' share of all logical CPUs, before and during a run (cpu.py). An idle desktop used 1-5%,
# with one-second spikes to 10%, on the reference machine; a build used most of the CPU.
CPU_BUSY_LIMIT = 0.15


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    codes: list[str] = field(default_factory=list)  # the result code of each error (errors.py)

    def ok(self) -> bool:
        return not self.errors

    def error(self, code: str, message: str) -> None:
        self.errors.append(message)
        self.codes.append(code)


def pause_tip(name: str) -> str:
    """For a program a pause hook can stop (a streaming host, a game launcher), how to have timed runs pause it."""
    key = next((k for k in PAUSABLE_PROGRAMS if name.lower().startswith(k)), None)
    if not key:
        return ""
    return (f" Timed runs can pause it and start it again: put pause = [\"{key}\"] in "
            f"{machine.config_dir() / 'machine.toml'}")


def blocking_programs() -> dict[str, str]:
    """The programs this machine pauses for a timed run (machine.toml), which must be gone while it runs."""
    return {k: v for k, v in PAUSABLE_PROGRAMS.items() if k in machine.pause_hooks()}


def _blocking(name: str) -> str | None:
    lower = name.lower()
    for key, label in blocking_programs().items():
        if lower.startswith(key):
            return label
    return None


def running_blockers() -> dict[str, list[str]]:
    """Paused programs anywhere on the machine, by label, whether or not they hold GPU memory now."""
    found: dict[str, list[str]] = {}
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        name = process_name(int(d.name))
        label = _blocking(name)
        if label:
            found.setdefault(label, []).append(f"{name} (pid {d.name})")
    return found


def before_run(procs: list[GpuProcess], baseline: Samples, cpu_others: float | None = None) -> Report:
    r = Report()
    for p in procs:
        mem = f"{p.used_mib} MiB" if p.used_mib is not None else "memory n/a"
        r.lines.append(f"gpu process: {p.pid} {p.kind} {p.name} {mem}")
    for label, names in running_blockers().items():
        r.error("DESKTOP", f"{label} is running ({', '.join(names[:3])}), and machine.toml pauses it for timed runs.")
    me = os.getpid()
    others = [p for p in procs if "C" in p.kind and p.pid != me and not _blocking(p.name)]
    for p in others:
        r.error("GPU_BUSY", f"Another compute process holds the GPU: {p.name} (pid {p.pid}). Stop it before measuring.{pause_tip(p.name)}")

    base_mib = baseline.median_bytes / MIB
    r.lines.append(
        f"baseline before load: median {base_mib:.0f} MiB, spread {baseline.spread_mib:.0f} MiB, "
        f"{len(baseline.used)} samples, max gap {baseline.max_gap_ms:.1f} ms"
    )
    if base_mib > BASELINE_WARN_MIB:
        r.warnings.append(f"Other programs hold {base_mib:.0f} MiB of GPU memory. The VRAM peak counts only what the model adds, "
                          "but the model has that much less room.")
    if baseline.spread_mib > SPREAD_LIMIT_MIB:
        r.error("GPU_BUSY", f"Idle device memory moved by {baseline.spread_mib:.0f} MiB within a second. Something else is using the GPU.")
    util = sum(baseline.utilization) / max(len(baseline.utilization), 1)
    r.lines.append(f"idle utilization: mean {util:.0f}%")
    if util > IDLE_UTIL_LIMIT:
        r.error("GPU_BUSY", f"The idle GPU is {util:.0f}% busy. Close videos, games or other GPU work.")
    if cpu_others is not None:
        r.lines.append(f"other programs' cpu before the run: {cpu_others:.1%} of all logical CPUs")
        if cpu_others > CPU_BUSY_LIMIT:
            r.error("CPU_BUSY", f"Other programs used {cpu_others:.0%} of the CPU while idle, above the {CPU_BUSY_LIMIT:.0%} limit. "
                                "A build, download or encoder slows the timed calls.")
    return r


def after_run(before: Samples, after: Samples, during: Samples, cpu_others: float | None = None) -> Report:
    r = Report()
    drift = (after.median_bytes - before.median_bytes) / MIB
    r.lines.append(f"baseline after the worker exited: median {after.median_bytes / MIB:.0f} MiB (drift {drift:+.0f} MiB)")
    gaps = sorted(during.gaps_ms)
    p99 = gaps[int(0.99 * (len(gaps) - 1))] if gaps else 0.0
    r.lines.append(f"polling during the run: {len(during.used)} samples, gap p99 {p99:.1f} ms, max {during.max_gap_ms:.1f} ms")
    if abs(drift) > DRIFT_LIMIT_MIB:
        r.error("DRIFT", f"The idle baseline moved by {drift:+.0f} MiB during the run, so the VRAM peak is unreliable. Measure again.")
    if p99 > MAX_POLL_GAP_MS:
        r.error("POLL_GAP", f"Memory polling fell behind (gap p99 {p99:.1f} ms, the protocol allows {MAX_POLL_GAP_MS:.0f} ms).")
    elif during.max_gap_ms > MAX_POLL_GAP_MS:
        r.warnings.append(f"One polling gap reached {during.max_gap_ms:.1f} ms.")
    if cpu_others is not None:
        r.lines.append(f"other programs' cpu during the run: {cpu_others:.1%} of all logical CPUs")
        if cpu_others > CPU_BUSY_LIMIT:
            r.error("CPU_BUSY", f"Other programs used {cpu_others:.0%} of the CPU during the run, above the {CPU_BUSY_LIMIT:.0%} limit, "
                                "so the timings are unreliable. Measure again.")
    return r
