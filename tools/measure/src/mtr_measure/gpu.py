"""NVML access: device facts, device-level used memory, GPU processes and the background poller.

Only the parent process uses NVML. It never creates a CUDA context, so its readings are the
device as the model's worker process sees it.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import pynvml

MIB = 1024 * 1024


def _str(v: str | bytes) -> str:
    return v.decode() if isinstance(v, bytes) else v


@dataclass
class GpuProcess:
    pid: int
    kind: str  # "C" (compute), "G" (graphics) or "C+G"
    used_mib: int | None
    name: str


def process_name(pid: int) -> str:
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[0].decode(errors="replace")
        if not cmd:
            return Path(f"/proc/{pid}/comm").read_text().strip()
        # Electron apps rewrite argv into one space-separated string; keep the program only.
        return cmd.split(" --", 1)[0].rsplit("/", 1)[-1]
    except OSError:
        return "?"


class Gpu:
    def __init__(self, index: int = 0):
        pynvml.nvmlInit()
        self.index = index
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(index)

    def close(self) -> None:
        pynvml.nvmlShutdown()

    def name(self) -> str:
        return _str(pynvml.nvmlDeviceGetName(self.handle))

    def driver(self) -> str:
        return _str(pynvml.nvmlSystemGetDriverVersion())

    def total_bytes(self) -> int:
        return int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).total)

    def compute_capability(self) -> str:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(self.handle)
        return f"{major}.{minor}"

    def power_limit_w(self) -> float | None:
        """The enforced power limit. Laptops and tuned cards often run below the default."""
        try:
            return round(pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000, 1)
        except pynvml.NVMLError:
            return None

    def power_default_w(self) -> float | None:
        try:
            return round(pynvml.nvmlDeviceGetPowerManagementDefaultLimit(self.handle) / 1000, 1)
        except pynvml.NVMLError:
            return None

    def used_bytes(self) -> int:
        return int(pynvml.nvmlDeviceGetMemoryInfo(self.handle).used)

    def utilization(self) -> int:
        return int(pynvml.nvmlDeviceGetUtilizationRates(self.handle).gpu)

    def processes(self) -> list[GpuProcess]:
        kinds: dict[int, set[str]] = {}
        used: dict[int, int | None] = {}
        for kind, fn in (("C", pynvml.nvmlDeviceGetComputeRunningProcesses), ("G", pynvml.nvmlDeviceGetGraphicsRunningProcesses)):
            for p in fn(self.handle):
                kinds.setdefault(p.pid, set()).add(kind)
                mem = getattr(p, "usedGpuMemory", None)
                used[p.pid] = mem // MIB if isinstance(mem, int) else None
        return [
            GpuProcess(pid, "+".join(sorted(k)), used[pid], process_name(pid))
            for pid, k in sorted(kinds.items())
        ]


@dataclass
class Samples:
    """Device-level used memory, sampled at a fixed interval."""

    used: list[int]
    gaps_ms: list[float]
    utilization: list[int]

    @property
    def median_bytes(self) -> int:
        return int(statistics.median(self.used))

    @property
    def peak_bytes(self) -> int:
        return max(self.used)

    @property
    def spread_mib(self) -> float:
        return (max(self.used) - min(self.used)) / MIB

    @property
    def max_gap_ms(self) -> float:
        return max(self.gaps_ms, default=0.0)


class Poller(threading.Thread):
    """Polls device used memory every `interval_ms` until stopped (protocol: 10 ms or less).

    It runs in the parent, which only waits on the worker, so it never competes with the
    timed loop for the worker's GIL.
    """

    def __init__(self, gpu: Gpu, interval_ms: float, with_utilization: bool = False):
        super().__init__(daemon=True)
        self.gpu = gpu
        self.interval = interval_ms / 1000
        self.with_utilization = with_utilization
        self.samples = Samples([], [], [])
        self._stop_event = threading.Event()

    def run(self) -> None:
        start = time.perf_counter()
        last = None
        k = 0
        while not self._stop_event.is_set():
            now = time.perf_counter()
            self.samples.used.append(self.gpu.used_bytes())
            if self.with_utilization:
                self.samples.utilization.append(self.gpu.utilization())
            if last is not None:
                self.samples.gaps_ms.append((now - last) * 1000)
            last = now
            k += 1
            delay = start + k * self.interval - time.perf_counter()
            if delay > 0:
                self._stop_event.wait(delay)

    def stop(self) -> Samples:
        self._stop_event.set()
        self.join()
        return self.samples


def sample_for(gpu: Gpu, seconds: float, interval_ms: float) -> Samples:
    """Sample for a fixed time, with utilization, for the idle baseline before and after a run."""
    p = Poller(gpu, interval_ms, with_utilization=True)
    p.start()
    time.sleep(seconds)
    return p.stop()
