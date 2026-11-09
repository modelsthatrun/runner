"""mtr profile: where one checkpoint's GPU time goes, kernel by kernel (docs/protocol.md, "Profiles").

    mtr profile <model> --size <size> [--input <file>]   models/<id>/profiles/<size>.json. Pauses nothing

A timed call includes the CPU's pre- and post-processing, kernel launches and the GPU's own work. The profile
splits the GPU's part by kernel. With the GPU's spec sheet, each kernel's time can then be scaled by the unit
it runs on (tensor cores at its types, CUDA cores, or memory bandwidth) to estimate the call on another GPU,
with the CPU's part left as it is.

It is not a timed measurement. It runs the model's worker once:
  1. load the checkpoint as `mtr run` does, at the same precision and on the same input
  2. WARMUP calls, then PLAIN calls timed like a timed run's, without the profiler
  3. PROFILER_WARMUP + PROFILED calls under torch.profiler (CUPTI). The first PROFILER_WARMUP are dropped.
Each profiled call is one window, closed by a synchronize, so every kernel it launched ran inside it. Per
kernel name, the record keeps the median over the windows of its launches, its GPU time, and its GPU-bound
time: the time of launches the GPU went straight on from, because the next work was already queued or the
CPU was waiting for it. Kernel times come from the GPU's own clock, so the profiler's CPU overhead doesn't
change them; it only makes fewer launches look GPU-bound, which widens an estimate's range.

Usage as a worker: python -m mtr_measure.profile <spec.json>
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import statistics
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from . import adapter as ad
from . import checks, cpu, machine, record
from .errors import MeasureError
from .gpu import MIB, Gpu, sample_for

PROFILE_VERSION = 1
WARMUP = 10
PLAIN = 50
PROFILER_WARMUP = 3
PROFILED = 20
# A text-to-image call makes one image: seconds of work and hundreds of thousands of kernels, so fewer calls
# give as steady a median and keep the trace small enough to hold in memory.
GEN_WARMUP = 2
GEN_PLAIN = 5
GEN_PROFILER_WARMUP = 1
GEN_PROFILED = 3
# A launch that starts within this many microseconds of the previous GPU work ending was waiting in the queue.
BOUND_GAP_US = 2.0
CALL = "mtr.call"
GPU_CATS = {"kernel": "kernel", "gpu_memcpy": "memcpy", "gpu_memset": "memset"}


# --- aggregation (no GPU needed; tests/test_profile.py) -------------------------------------------


def call_windows(events: list[dict]) -> list[tuple[float, float]]:
    """The profiled calls, as (start, end) in µs on the trace's clock, in order."""
    out = [(float(e["ts"]), float(e["ts"]) + float(e["dur"])) for e in events if e.get("cat") == "user_annotation" and e.get("name") == CALL]
    return sorted(out)


def gpu_events(events: list[dict]) -> list[dict]:
    out = []
    for e in events:
        kind = GPU_CATS.get(e.get("cat", ""))
        if kind and e.get("ph", "X") == "X" and "dur" in e:
            out.append({"name": str(e["name"]), "kind": kind, "ts": float(e["ts"]), "dur": float(e["dur"])})
    return sorted(out, key=lambda e: e["ts"])


def one_call(evs: list[dict], gap_us: float) -> tuple[dict[tuple[str, str], list[float]], float]:
    """For one window: per (name, kind) [launches, µs, GPU-bound µs], and the busy time (union of intervals)."""
    per: dict[tuple[str, str], list[float]] = {}
    busy = 0.0
    run_start = run_end = None
    for i, e in enumerate(evs):
        start, end = e["ts"], e["ts"] + e["dur"]
        # GPU-bound: the next work started as soon as this ended (it was queued behind it), or this is the
        # last work of the call, which the closing synchronize waited for.
        nxt = evs[i + 1]["ts"] if i + 1 < len(evs) else None
        bound = nxt is None or nxt - end <= gap_us
        slot = per.setdefault((e["name"], e["kind"]), [0.0, 0.0, 0.0])
        slot[0] += 1
        slot[1] += e["dur"]
        slot[2] += e["dur"] if bound else 0.0
        if run_end is None or start > run_end:
            if run_end is not None:
                busy += run_end - run_start
            run_start, run_end = start, end
        else:
            run_end = max(run_end, end)
    if run_end is not None:
        busy += run_end - run_start
    return per, busy


def aggregate(events: list[dict], skip: int, gap_us: float = BOUND_GAP_US) -> dict:
    """Medians over the profiled calls, dropping the first `skip`: kernels by name, busy and wall time per call."""
    windows = call_windows(events)[skip:]
    if not windows:
        raise MeasureError("FAILED", "The profiler recorded no calls")
    gpu = gpu_events(events)
    calls = []
    for a, b in windows:
        calls.append(one_call([e for e in gpu if a <= e["ts"] < b], gap_us))
    keys = sorted({k for per, _ in calls for k in per})

    def med(values: list[float]) -> float:
        return statistics.median(values) if values else 0.0

    kernels = []
    for name, kind in keys:
        rows = [per.get((name, kind), [0.0, 0.0, 0.0]) for per, _ in calls]
        ms = med([r[1] for r in rows]) / 1000
        bound = min(ms, med([r[2] for r in rows]) / 1000)
        kernels.append({"name": name, "kind": kind, "count": med([r[0] for r in rows]), "ms": round(ms, 5), "bound_ms": round(bound, 5)})
    kernels.sort(key=lambda k: -k["ms"])
    return {
        "kernels": kernels,
        "gpu_busy_ms": round(med([busy for _, busy in calls]) / 1000, 4),
        "profiled_wall_ms": round(med([b - a for a, b in windows]) / 1000, 4),
        "calls": len(windows),
    }


# --- worker (runs in the model's environment) -----------------------------------------------------


def run(spec: dict) -> dict:
    a = ad.load_adapter(Path(spec["adapter"]))
    size = spec["size"]
    ad.check_size(a, size)
    precision = ad.choose_precision(list(a.PRECISIONS))
    gen = ad.kind(a) == "generation"
    if gen:
        from . import generation, protocol

        x = generation.make(a, size, generation.read_prompt([Path(p) for p in spec["inputs"]]), protocol.GEN_SEED)
        warmup, n_plain, p_warmup, n_profiled = GEN_WARMUP, GEN_PLAIN, GEN_PROFILER_WARMUP, GEN_PROFILED
    else:
        x = ad.prepare_input(a, [Path(p) for p in spec["inputs"]])
        warmup, n_plain, p_warmup, n_profiled = WARMUP, PLAIN, PROFILER_WARMUP, PROFILED

    import torch
    from torch.profiler import ProfilerActivity, profile, record_function

    if not torch.cuda.is_available():
        raise MeasureError("FAILED", "torch can't see a CUDA device")
    if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
        raise MeasureError("FAILED", "This GPU only emulates bf16, and the protocol doesn't time an emulation (docs/protocol.md, \"Scope\")")
    sync = torch.cuda.synchronize

    model = a.load(size, precision)
    sync()
    for _ in range(warmup):
        a.predict(model, x)
        sync()
    plain = []
    for _ in range(n_plain):
        sync()
        t0 = time.perf_counter()
        a.predict(model, x)
        sync()
        plain.append((time.perf_counter() - t0) * 1000)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(p_warmup + n_profiled):
            sync()
            with record_function(CALL):
                a.predict(model, x)
                sync()
    with tempfile.TemporaryDirectory(prefix="mtr-profile-") as tmp:
        trace = Path(tmp) / "trace.json"
        prof.export_chrome_trace(str(trace))
        events = json.loads(trace.read_text()).get("traceEvents", [])
    agg = aggregate(events, p_warmup)

    cudnn = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None
    return {
        "precision": precision,
        "input": ad.input_size(a, size, x),
        "warmup": warmup,
        "plain_ms": plain,
        **agg,
        "weights": [str(Path(p).resolve()) for p in a.weight_files(size)],
        "env": {
            "cuda": str(torch.version.cuda),
            "torch": torch.__version__,
            "cudnn": str(cudnn) if cudnn is not None else None,
            "python": platform.python_version(),
            "package": ad.package(a),
            "arch_list": list(torch.cuda.get_arch_list()),
            # Read after load and warm-up: packages set these when imported or when a model loads.
            "flags": {
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "float32_matmul_precision": str(torch.get_float32_matmul_precision()),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            },
        },
        "device": torch.cuda.get_device_name(),
    }


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = {"ok": True, **run(spec)}
    except MeasureError as e:
        result = {"ok": False, "code": e.code, "error": str(e)}
    except Exception as e:  # the adapter's own code failed; keep the traceback for the log
        from .worker import is_oom

        code = "OOM" if is_oom(e) else "ADAPTER_RUNTIME"
        result = {"ok": False, "code": code, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    Path(spec["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


# --- the command ----------------------------------------------------------------------------------


def profile_path(model_dir: Path, size: str) -> Path:
    return model_dir / "profiles" / f"{size}.json"


def build(*, model: str, size: str, weights: list[dict], worker: dict, gpu_name: str, driver: str, facts: dict, provenance: dict, profiled: str) -> dict:
    plain = worker["plain_ms"]
    return {
        "kind": "profile",
        "model": model,
        "size": size,
        "weights": weights,
        "profile": {
            "version": PROFILE_VERSION,
            "precision": worker["precision"],
            "input": worker["input"],
            "calls": {"warmup": worker.get("warmup", WARMUP), "plain": len(plain), "profiled": worker["calls"]},
            "wall_ms": {"median": round(statistics.median(plain), 3), "p90": round(record.percentile(plain, 90), 3)},
            "profiled_wall_ms": worker["profiled_wall_ms"],
            "gpu_busy_ms": worker["gpu_busy_ms"],
            "bound_gap_us": BOUND_GAP_US,
            "kernels": worker["kernels"],
            "env": {"gpu": gpu_name, "driver": driver, **worker["env"], **facts},
            "profiled": profiled,
        },
        "provenance": provenance,
    }


def cmd_profile(argv: list[str], log: Any) -> int:
    from .cli import Parser, check_paths, check_poll, default_inputs, finish, model_args, spawn_worker

    p: argparse.ArgumentParser = Parser(
        prog="mtr profile",
        description="Split one checkpoint's GPU time by kernel on this GPU, for estimates on other GPUs. Pauses nothing; not a timed run.",
    )
    model_args(p, input_required=False)
    args = p.parse_args(argv)
    check_poll(args)
    model_dir = check_paths(args)
    model_id = args.model_id or model_dir.name
    out = profile_path(model_dir, args.size)
    log_path = out.with_suffix(".log")
    if ad.kind_of_file(model_dir / "adapter.py") == "llm":
        raise MeasureError("USAGE", "Language models need no profile: their estimates come from the run record (decode reads every weight per token)")

    warnings: list[str] = []
    log(f"mtr profile {model_id} / {args.size} (not a timed measurement: nothing is paused)")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    try:
        inputs = [f.resolve() for f in args.inputs] if args.inputs else default_inputs(model_dir, timed=True)
        for f in inputs:
            log(f"input: {f} sha256 {record.sha256_file(f)}")
        spec = {"adapter": str(model_dir / "adapter.py"), "size": args.size, "inputs": [str(f) for f in inputs]}
        gpu = Gpu(args.gpu)
        try:
            gpu_name, driver, facts = gpu.name(), gpu.driver(), machine.facts(gpu)
            window = cpu.Window().start()
            before = sample_for(gpu, 1.0, args.poll_ms)
            util = sum(before.utilization) / max(len(before.utilization), 1)
            log(f"gpu: {gpu_name}, driver {driver}, {before.median_bytes / MIB:.0f} MiB in use, idle utilization {util:.0f}%; "
                f"other programs use {window.others():.0%} of the CPU")
            # Kernel times are the GPU's own, so the desktop may stay. A process that only holds a CUDA context
            # runs no kernels; one that does shows up in the utilization, which is what stops the profile.
            me = os.getpid()
            for proc in gpu.processes():
                if "C" in proc.kind and proc.pid != me and not checks._blocking(proc.name):
                    warnings.append(f"Another compute process holds the GPU: {proc.name} (pid {proc.pid}). It was idle at the start; "
                                    "if it ran kernels during the profile, they shared the SMs.")
                    log(f"WARNING: {warnings[-1]}")
            if util > checks.IDLE_UTIL_LIMIT:
                raise MeasureError("GPU_BUSY", f"The idle GPU is {util:.0f}% busy. Close videos, games or other GPU work.")
            counts = (GEN_WARMUP, GEN_PLAIN, GEN_PROFILER_WARMUP, GEN_PROFILED) if ad.kind_of_file(model_dir / "adapter.py") == "generation" \
                else (WARMUP, PLAIN, PROFILER_WARMUP, PROFILED)
            log("running the worker: load, {} warm-up and {} plain calls, then {} + {} calls under the profiler".format(*counts))
            worker, _ = spawn_worker(spec, gpu, args.poll_ms, args.gpu, "mtr_measure.profile")
        finally:
            gpu.close()
        weights = record.weight_entries([Path(w) for w in worker["weights"]])
    except FileNotFoundError as e:
        return finish(log, log_path, "WEIGHTS_MISSING", str(e), warnings=warnings)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e), warnings=warnings)

    profiled = datetime.now().astimezone().isoformat(timespec="seconds")
    rec = build(model=model_id, size=args.size, weights=weights, worker=worker, gpu_name=gpu_name, driver=driver, facts=facts,
                provenance=machine.provenance(model_dir, inputs), profiled=profiled)
    pr = rec["profile"]
    wall = pr["wall_ms"]["median"]
    log(f"precision {pr['precision']}, input {pr['input']}, torch {pr['env']['torch']}, cuDNN {pr['env']['cudnn']}")
    log(f"flags after load: {pr['env']['flags']}")
    log(f"plain calls: median {wall} ms; under the profiler {pr['profiled_wall_ms']} ms")
    log(f"gpu busy: {pr['gpu_busy_ms']} ms per call ({pr['gpu_busy_ms'] / wall:.0%} of a plain call), {len(pr['kernels'])} kernels and copies")
    for k in pr["kernels"][:12]:
        log(f"  {k['ms']:8.3f} ms  bound {k['bound_ms']:7.3f}  x{k['count']:<5g} {k['kind']:<7} {k['name'][:140]}")
    for k in pr["kernels"]:
        log(f"kernel: {json.dumps(k)}", echo=False)
    run = model_dir / "runs" / f"{args.size}.json"
    if run.is_file():
        rm = json.loads(run.read_text()).get("measurement", {})
        ref = rm.get("latency_ms", {}).get("median")
        if ref:
            log(f"the run record's median: {ref} ms ({wall / ref - 1:+.1%})")
            if abs(wall / ref - 1) > 0.15 or rm.get("precision") != pr["precision"] or rm.get("input") != pr["input"]:
                warnings.append(f"This profile ({wall} ms, {pr['precision']}, {pr['input']}) doesn't match the run record ({ref} ms, "
                                f"{rm.get('precision')}, {rm.get('input')}), so estimates won't use it. Run it again on a quieter machine.")
    problems = record.validate(rec, "profile-record")
    for prob in problems:
        log(f"SCHEMA: {prob}")
    if problems:
        return finish(log, log_path, "SCHEMA", f"The profile fails its schema: {problems[0]}", warnings=warnings)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {out}")
    share = pr["gpu_busy_ms"] / wall
    return finish(log, log_path, "OK", f"{model_id} / {args.size}: GPU busy {pr['gpu_busy_ms']} of {wall} ms per call ({share:.0%})",
                  {"profile": str(out), "wall_ms": wall, "gpu_busy_ms": pr["gpu_busy_ms"]}, warnings)


if __name__ == "__main__":
    raise SystemExit(main())
