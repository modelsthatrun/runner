"""This machine: its optional configuration, the facts a record carries about it, and where the tool came from.

The configuration is ~/.config/modelsthatrun/machine.toml (MTR_CONFIG_DIR moves it). Nothing is needed on
most machines. A machine that streams its desktop or runs a game launcher on the measuring GPU can name
pause hooks (desktop.py), which stop those programs for each timed run and start them again afterwards:

    pause = ["sunshine", "steam"]

Records never carry the host name, the user name or any path.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from functools import cache
from pathlib import Path

from . import paths

TOOL_NAME = "mtr"
# The tool's own version, recorded in provenance. pyproject.toml's version stays put: every model's uv.lock
# records it, and changing it would change every lock.
TOOL_VERSION = "0.2.0"


def config_dir() -> Path:
    return Path(os.environ.get("MTR_CONFIG_DIR", "~/.config/modelsthatrun")).expanduser()


@cache
def config() -> dict:
    path = config_dir() / "machine.toml"
    if not path.is_file():
        return {}
    text = path.read_text()
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10: the file holds one list, so read just that
        found = re.search(r"^pause\s*=\s*\[([^\]]*)\]", text, re.M)
        return {"pause": re.findall(r"""["']([^"']+)["']""", found.group(1))} if found else {}
    return tomllib.loads(text)


def pause_hooks() -> list[str]:
    """The pause hooks this machine asked for, by name (desktop.HOOKS)."""
    return [str(h) for h in config().get("pause", [])]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return re.sub(r"\s+", " ", line.split(":", 1)[1]).strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def os_name() -> str:
    libc, version = platform.libc_ver()
    return f"{platform.system()} {platform.release()}" + (f" {libc} {version}" if libc else "")


def facts(gpu) -> dict:
    """What a record says about the machine, beyond the GPU name and driver (gpu.Gpu)."""
    return {
        "cpu": cpu_model(),
        "gpu_memory_mib": round(gpu.total_bytes() / (1024 * 1024)),
        "compute_capability": gpu.compute_capability(),
        "os": os_name(),
        "power_limit_w": gpu.power_limit_w(),
        "power_default_w": gpu.power_default_w(),
    }


def _git(*args: str) -> str | None:
    try:
        proc = subprocess.run(["git", "-C", str(paths.root()), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return proc.stdout if proc.returncode == 0 else None


def tool(model_dir: Path | None = None) -> dict:
    """The tool's version and commit, and whether the tool, the timing images or the model directory had uncommitted changes."""
    commit = (_git("rev-parse", "HEAD") or "").strip() or None
    dirty = None
    if commit:
        where = ["tools/measure", "timing"]
        if model_dir is not None:
            try:
                where.append(str(model_dir.resolve().relative_to(paths.root())))
            except ValueError:
                pass  # a model outside the checkout can't be traced to the commit
        status = _git("status", "--porcelain", "--", *where)
        dirty = None if status is None else bool(status.strip())
    return {"name": TOOL_NAME, "version": TOOL_VERSION, "commit": commit, "dirty": dirty}


def upstream_has(commit: str) -> bool | None:
    """Whether a remote-tracking branch contains the commit, so others can check it out. None if unknown."""
    out = _git("branch", "-r", "--contains", commit)
    return None if out is None else bool(out.strip())


def pin_gpu(index: int) -> None:
    """Run everything this process starts, and torch in this process, on one GPU. NVML indexes devices in
    PCI bus order, so CUDA has to as well for --gpu to name the same device in both."""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(index)


def provenance(model_dir: Path, inputs: list[Path] | None) -> dict:
    """The record's provenance block: the tool, and the hashes of the adapter, its lock and the inputs."""
    from . import record

    return {
        "tool": tool(model_dir),
        **record.fingerprint(model_dir),
        "input_sha256": [record.sha256_file(f) for f in inputs] if inputs is not None else None,
    }


# The oldest Linux driver each CUDA version of the PyTorch wheels needs (NVIDIA's CUDA toolkit release notes).
MIN_DRIVER = {"13.0": "580", "12.9": "575", "12.8": "570", "12.6": "560", "12.4": "550", "12.1": "530", "11.8": "520"}


def cuda_of_lock(lock: Path) -> str | None:
    """The CUDA version a model's environment runs, from the CUDA runtime its uv.lock pins, e.g. "13.0"."""
    if not lock.is_file():
        return None
    found = re.search(r'^name = "nvidia-cuda-runtime(?:-cu1\d)?"\nversion = "(\d+)\.(\d+)', lock.read_text(), re.M)
    return f"{found.group(1)}.{found.group(2)}" if found else None


def driver_cuda() -> str | None:
    """The newest CUDA version the installed NVIDIA driver supports, e.g. "13.0". None without a driver."""
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            v = pynvml.nvmlSystemGetCudaDriverVersion_v2()
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return None
    return f"{v // 1000}.{v % 1000 // 10}"


def cuda_fits(need: str | None, have: str | None) -> bool:
    """Whether a driver that supports CUDA `have` runs an environment built for CUDA `need`. Unknown counts as yes."""
    if not need or not have:
        return True
    return tuple(map(int, need.split("."))) <= tuple(map(int, have.split(".")))


def driver_advice(need: str) -> str:
    want = MIN_DRIVER.get(need)
    return f"NVIDIA driver {want} or newer" if want else f"an NVIDIA driver that supports CUDA {need}"
