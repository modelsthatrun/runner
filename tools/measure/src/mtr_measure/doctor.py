"""mtr doctor: is this machine ready to measure, and what would its records say about it?

It looks at everything a measurement depends on, changes nothing and loads no model:
    the OS and the NVIDIA driver, each GPU (memory, compute capability, power limit, what holds it)
    uv, git and whether this checkout is a public, unmodified commit (records from it can be submitted)
    nvcc, needed once to build llama.cpp for language models
    the timing images, the disk space, how busy the CPU is, the pause hooks and the inbox sign-in
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from pathlib import Path

from . import checks, cpu, machine, paths, timing
from .cli import Parser
from .errors import MeasureError, emit

# A timed run of a large model needs its weights, its environment and room to spare.
DISK_WARN_GIB = 30
# The oldest CUDA toolkit whose nvcc builds for each compute capability (llama.cpp, language models).
NVCC_FOR_CC = {"12.0": "12.8", "10.0": "12.8", "9.0": "12.0", "8.9": "11.8"}


def line(ok: bool | None, what: str, detail: str = "") -> None:
    mark = {True: "ok  ", False: "FAIL", None: "warn"}[ok]
    print(f"{mark}  {what}{f': {detail}' if detail else ''}", flush=True)


def version_of(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (out.stdout or out.stderr).strip()
    return text.splitlines()[-1] if out.returncode == 0 and text else None


def main(argv: list[str]) -> int:
    Parser(prog="mtr doctor", description="Check this machine for measuring. Changes nothing.").parse_args(argv)
    warnings: list[str] = []
    facts: dict = {}

    if platform.system() != "Linux":
        line(False, "operating system", platform.system())
        raise MeasureError("UNSUPPORTED", f"The tool runs on Linux only for now, not {platform.system()}")
    line(True, "operating system", machine.os_name())
    facts["os"] = machine.os_name()
    facts["cpu"] = machine.cpu_model()
    line(True, "cpu", facts["cpu"])

    try:
        import pynvml

        pynvml.nvmlInit()
    except Exception as e:  # no driver, no NVML library, no device
        line(False, "NVIDIA driver", str(e))
        raise MeasureError("UNSUPPORTED", "NVML isn't available: this needs an NVIDIA GPU and its driver") from None
    from .gpu import Gpu

    count = pynvml.nvmlDeviceGetCount()
    pynvml.nvmlShutdown()
    gpus = []
    for i in range(count):
        g = Gpu(i)
        try:
            info = {"index": i, "name": g.name(), "driver": g.driver(), "memory_mib": round(g.total_bytes() / 2**20),
                    "used_mib": round(g.used_bytes() / 2**20), "compute_capability": g.compute_capability(),
                    "power_limit_w": g.power_limit_w(), "power_default_w": g.power_default_w(),
                    # Programs the pause hooks stop for a timed run don't count as busy.
                    "compute_processes": [p.name for p in g.processes() if "C" in p.kind and not checks._blocking(p.name)]}
        finally:
            g.close()
        gpus.append(info)
        limited = info["power_limit_w"] and info["power_default_w"] and info["power_limit_w"] < info["power_default_w"] * 0.98
        line(True, f"gpu {i}", f"{info['name']}, {info['memory_mib']} MiB ({info['used_mib']} in use), compute {info['compute_capability']}, "
                               f"driver {info['driver']}, power limit {info['power_limit_w']} of {info['power_default_w']} W")
        if limited:
            warnings.append(f"GPU {i} runs below its default power limit; records say so, and its numbers will be slower")
            line(None, f"gpu {i} power", "below the default limit: records carry it, and pages mark it")
        if info["compute_processes"]:
            tips = "".join(checks.pause_tip(n) for n in info["compute_processes"])
            warnings.append(f"GPU {i} has compute processes: {', '.join(info['compute_processes'])}. Timed runs refuse to start while they run.{tips}")
            line(None, f"gpu {i} busy", ", ".join(info["compute_processes"]) + (f".{tips}" if tips else ""))
    facts["gpus"] = gpus

    # Each model's environment is built for one CUDA version; the driver must support it.
    from .main import adapters, model_cuda

    have = machine.driver_cuda()
    needs = {d.name: model_cuda(d) for d in adapters()}
    too_new = sorted(n for n, c in needs.items() if not machine.cuda_fits(c, have))
    facts["driver_cuda"] = have
    if too_new:
        newest = max((c for c in needs.values() if c), key=lambda c: tuple(map(int, c.split("."))))
        line(None, "cuda", f"the driver supports CUDA up to {have}; {len(too_new)} of {len(needs)} models need more. "
                           f"Install {machine.driver_advice(newest)} to run them all")
        warnings.append(f"{len(too_new)} models need a newer driver than this one (CUDA up to {have}): {', '.join(too_new)}")
    else:
        line(True, "cuda", f"the driver supports CUDA up to {have}, enough for every model")

    uv = version_of([shutil.which("uv") or "uv", "--version"])
    line(bool(uv), "uv", uv or "not found: https://docs.astral.sh/uv/")
    if not uv:
        raise MeasureError("UNSUPPORTED", "uv isn't installed. Get it from https://docs.astral.sh/uv/")

    tool = machine.tool()
    facts["tool"] = tool
    if not tool["commit"]:
        line(None, "checkout", "not a git checkout: records can't be traced to a commit, so they can't be submitted")
        warnings.append("Not a git checkout. Clone the repository to submit results")
    else:
        public = machine.upstream_has(tool["commit"])
        state = "modified" if tool["dirty"] else "unmodified"
        line(None if tool["dirty"] or public is False else True, "checkout", f"{tool['commit'][:12]}, tools/measure {state}"
             + ("" if public else "; this commit isn't on a remote branch yet"))
        if tool["dirty"]:
            warnings.append("tools/measure has uncommitted changes: records measured now can't be submitted")
        if public is False:
            warnings.append("HEAD isn't on github.com/modelsthatrun/runner yet. Records must come from a commit there: `git pull`, "
                            "or for a new adapter, wait until its pull request is merged")

    nvcc = None
    try:
        from . import engine

        nvcc = f"{engine.nvcc()} (CUDA {engine.nvcc_version(engine.nvcc())})"
    except MeasureError:
        pass
    line(True if nvcc else None, "nvcc", nvcc or "not found: needed only for language models (llama.cpp is built once)")
    # Building llama.cpp also needs CMake and a C++ compiler, and an nvcc new enough for the GPU.
    for tool, why in (("cmake", "CMake"), ("c++", "a C++ compiler")):
        if not shutil.which(tool):
            line(None, tool, f"not found: language models need {why} to build llama.cpp once")
            warnings.append(f"{why} is missing, so language models can't build llama.cpp")
    if nvcc and gpus:
        cc = max(g["compute_capability"] for g in gpus)
        need = NVCC_FOR_CC.get(cc)
        have = engine.nvcc_version(engine.nvcc())
        if need and have != "unknown" and tuple(map(int, have.split("."))) < tuple(map(int, need.split("."))):
            line(None, "nvcc", f"CUDA {have} can't build for compute {cc}; language models need CUDA {need} or newer")
            warnings.append(f"nvcc {have} is too old for compute capability {cc}: install CUDA {need} or newer for language models")

    try:
        timing.image_for("object-detection")
        line(True, "timing images", "match timing/timing.toml")
    except MeasureError as e:
        line(False, "timing images", str(e))
        warnings.append(str(e))

    for where in (paths.root(), Path("~/.cache").expanduser()):
        where.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(where).free / 2**30
        line(free >= DISK_WARN_GIB or None, f"disk at {'the checkout' if where == paths.root() else '~/.cache'}", f"{free:.0f} GiB free")
        if free < DISK_WARN_GIB:
            warnings.append(f"Only {free:.0f} GiB free at {where}. Large models need tens of GB; free it with `mtr clean <model> --yes`")

    window = cpu.Window().start()
    time.sleep(1.0)
    busy = window.others()
    line(busy <= 0.15 or None, "cpu", f"other programs use {busy:.0%} of it now (timed runs refuse above 15%)")
    if busy > 0.15:
        warnings.append(f"Other programs use {busy:.0%} of the CPU. Timed runs refuse above 15%")

    hooks = machine.pause_hooks()
    line(True, "pause hooks", ", ".join(hooks) if hooks else f"none. Add them in {machine.config_dir() / 'machine.toml'} only if a streaming host or game launcher shares the GPU")

    from . import submit

    cred = submit.credentials()
    line(True if cred else None, "inbox", f"signed in as {cred['login']}" if cred else "not signed in: `mtr login` before `mtr submit`")

    msg = f"Ready: {len(gpus)} GPU(s)" + (f", {len(warnings)} warning(s)" if warnings else "")
    return emit("OK", msg, facts, warnings)
