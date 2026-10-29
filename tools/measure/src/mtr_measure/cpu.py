"""How much of the CPU other programs used over a window of time.

A timed call covers pre-processing and post-processing on the CPU, so a build, a download or an
encoder running at the same time slows it. The tool reads the machine's CPU time from /proc/stat
at both ends of a window and takes away its own share and its children's (the worker, and the
llama-server it starts). What is left is everything else, as a fraction of all logical CPUs.
"""

from __future__ import annotations

import os
import resource
from dataclasses import dataclass

TICKS = os.sysconf("SC_CLK_TCK")


def machine() -> tuple[int, int]:
    """Total and busy CPU time of the whole machine so far, in clock ticks (all CPUs together)."""
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:]]
    idle = v[3] + v[4]  # idle and iowait
    total = sum(v[:8])  # guest time is already counted in user
    return total, total - idle


def own_ticks() -> float:
    """CPU time of this process and its finished children, in clock ticks."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return (s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime) * TICKS


@dataclass
class Window:
    """Start it, do the work (wait for the worker), stop it: others() is the other programs' share."""

    total: int = 0
    busy: int = 0
    own: float = 0.0

    def start(self) -> "Window":
        self.total, self.busy = machine()
        self.own = own_ticks()
        return self

    def others(self) -> float:
        total, busy = machine()
        dt = total - self.total
        if dt <= 0:
            return 0.0
        theirs = (busy - self.busy) - (own_ticks() - self.own)
        return max(0.0, min(1.0, theirs / dt))
