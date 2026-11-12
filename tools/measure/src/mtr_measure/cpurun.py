"""CPU rows: `mtr check` and `mtr run` with --runtime pytorch-cpu, onnxruntime-cpu or llamacpp-cpu (docs/protocol.md,
"CPU rows").

The same official code as the GPU row, with the GPU hidden from the worker (CUDA_VISIBLE_DEVICES set to nothing,
MTR_DEVICE=cpu). A CPU row holds no GPU, so a timed run pauses nothing; instead the machine must be quiet: other
programs may use at most CPU_BUSY_LIMIT_CPU of the CPU before and during each run, the CPU must not throttle for
heat, the worker must not be swapped out, and the timed calls must stay steady.

Proof that the CPU computes what the GPU does is a match, not an eval: `check` compares its output on the timing input
with the GPU check's (match.py). A checkpoint whose call runs past CPU_CALL_LIMIT_S, whose official code doesn't run
on a CPU, or which computes in half precision there gets a record that says so instead of a number, written by a
quiet `mtr run`, never typed.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from . import adapter as ad
from . import checks, cpu, hostmem, machine, protocol, record, runtimes
from .errors import MeasureError
from .ort import PrepareFailed

# The not-timed reasons a check passes on to the run, which records them (and the schema's names for them).
NOT_TIMED = {"CPU_UNSUPPORTED": "unsupported", "PRECISION": "precision", "OVER_LIMIT": "over-limit"}
# Room a run needs in free memory beyond what the check saw its worker hold.
RAM_MARGIN = 1.2
RAM_MARGIN_MIB = 1024
POLL_S = 1.0
# The parent looks for the row's processes on the GPU every this many polls.
GPU_EVERY = 5
# The quiet window before each run in which other programs' share of the CPU is measured.
BASELINE_S = 5.0


class OverLimit(MeasureError):
    def __init__(self, call_s: float, beat: dict):
        super().__init__("OVER_LIMIT", f"A call ran for {call_s:.0f} s, past the {protocol.CPU_CALL_LIMIT_S:.0f} s a CPU row times, so it was stopped")
        self.call_s = call_s
        self.beat = beat


class WorkerFailed(MeasureError):
    def __init__(self, worker: dict):
        super().__init__(worker.get("code", "ADAPTER_RUNTIME"), worker.get("error", "the worker failed"))
        self.worker = worker


def worker_env() -> dict:
    env = {k: v for k, v in os.environ.items() if k != "CUDA_DEVICE_ORDER"}
    env.update(CUDA_VISIBLE_DEVICES="", MTR_DEVICE="cpu")
    return env


def threads() -> int:
    """The protocol's thread count: the physical cores, as PyTorch, ONNX Runtime and llama.cpp each default to."""
    return hostmem.cores()[0]


def open_gpu(index: int):
    """NVML, to check that no process of the row touches the GPU. None on a machine without it."""
    try:
        from .gpu import Gpu

        return Gpu(index)
    except Exception:
        return None


def spawn(spec: dict, module: str, gpu, log, limit_s: float | None = protocol.CPU_CALL_LIMIT_S) -> tuple[dict, dict | None]:
    """Run the worker on the CPU. Stops it when one call runs past limit_s (OverLimit), or when any process it started
    shows up on the GPU (GPU_TOUCHED). Returns its result and its last heartbeat."""
    with tempfile.TemporaryDirectory(prefix="mtr-cpu-") as tmp:
        spec_path, result_path, beat_path = Path(tmp) / "spec.json", Path(tmp) / "result.json", Path(tmp) / "beat.json"
        spec_path.write_text(json.dumps({**spec, "result": str(result_path), "heartbeat": str(beat_path)}))
        proc = subprocess.Popen([sys.executable, "-m", module, str(spec_path)], env=worker_env(), start_new_session=True)
        beat = None
        polls = 0
        try:
            while True:
                try:
                    proc.wait(timeout=POLL_S)
                    break
                except subprocess.TimeoutExpired:
                    pass
                try:
                    beat = json.loads(beat_path.read_text())
                except (OSError, json.JSONDecodeError):
                    pass
                if limit_s and beat and beat.get("phase") in ("warmup", "timed", "check"):
                    running = time.time() - float(beat["started"])
                    if running > limit_s:
                        stop(proc)
                        raise OverLimit(running, beat)
                polls += 1
                if gpu is not None and polls % GPU_EVERY == 0:
                    mine = hostmem.descendants(proc.pid)
                    on_gpu = [p for p in gpu.processes() if p.pid in mine]
                    if on_gpu:
                        stop(proc)
                        raise MeasureError("GPU_TOUCHED", f"{on_gpu[0].name} (pid {on_gpu[0].pid}) of a CPU row appeared on the GPU")
        except BaseException:
            stop(proc)
            raise
        worker = json.loads(result_path.read_text()) if result_path.is_file() else None
    if not worker:
        raise MeasureError("ADAPTER_RUNTIME", f"The worker exited with code {proc.returncode} without a result")
    if not worker.get("ok"):
        if worker.get("traceback"):
            print(worker["traceback"], file=sys.stderr, flush=True)
        raise WorkerFailed(worker)
    return worker, beat


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


# --- matching the GPU's output -------------------------------------------------------------------


def match_result(model_dir: Path, size: str, runtime: str, kind: str, check_notes: dict) -> dict | None:
    """The record's runtime.match for this check, or None when the reference is missing or stale. Raises NOT_MATCHED
    when both exist and disagree."""
    if kind == "generation":
        return check_notes.get("image_match")
    from . import match
    from .runqueue import read_check

    ref = record.reference_path(model_dir, size)
    got = record.prediction_path(model_dir, size, runtime)
    try:
        read_check(model_dir, size)  # the GPU check, current with this adapter and lock
    except MeasureError:
        return None
    if not ref.is_file() or not got.is_file():
        return None
    try:
        res = match.compare(ref, got, runtime)
    except match.MatchError as e:
        raise MeasureError("NOT_MATCHED", f"The CPU's output can't be compared with the GPU's: {e}") from e
    if not res["passed"]:
        raise MeasureError("NOT_MATCHED", f"The CPU's output on the timing input differs from the GPU's: {match.describe(res)}")
    return {k: res[k] for k in ("type", "reference", "metrics", "tolerance")}


def image_match(worker: dict, png: Path) -> dict | None:
    """A text-to-image check has no reference: fp32 and fp16 draw other pixels. It checks the image isn't blank."""
    if not png.is_file():
        return None
    import numpy as np
    from PIL import Image

    with Image.open(png) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32)
    std = float(a.std())
    if std < 2.0:
        raise MeasureError("NOT_MATCHED", f"The CPU's image is blank (pixel standard deviation {std:.2f})")
    w, h = (int(v) for v in worker["input"].split("x"))
    return {"type": "image", "reference": None, "metrics": {"width": float(w), "height": float(h), "pixel_std": round(std, 2)},
            "tolerance": {"pixel_std": 2.0}}


def require_ready(model_dir: Path, size: str, runtime: str, check: dict) -> dict | None:
    """What a queued or timed CPU row needs from its check: a match with the GPU, or a reason for no number. Returns
    the match (None for a not-timed check)."""
    if check.get("not_timed") or check.get("over_limit"):
        # The check's finding is the row's result, which a quiet run confirms (cmd_run): no output to match.
        return None
    m = check.get("match") or match_result(model_dir, size, runtime, check.get("kind", "image"), {})
    if not m:
        raise MeasureError("NOT_MATCHED", f"{model_dir.name} / {size} on {runtime} has no match with the GPU's output yet. Run "
                                          f"`mtr check {model_dir.name} --size {size}` on the GPU (it saves the reference), then "
                                          f"`mtr check … --runtime {runtime}` again")
    return m


# --- check ---------------------------------------------------------------------------------------


def cmd_check(args, model_dir: Path, model_id: str, kind: str, log, finish, write_check, default_inputs, read_task) -> int:
    runtime = args.runtime
    log_path = record.check_path(model_dir, args.size, runtime).with_suffix(".log")
    log(f"mtr check {model_id} / {args.size} on {runtime} (not a measurement: nothing is paused, no record is written)")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    if kind == "llm":
        from . import cpullm

        return cpullm.cmd_check(args, model_dir, model_id, log, finish, write_check, log_path)
    gen = kind == "generation"
    inputs = [f.resolve() for f in args.inputs] if args.inputs else default_inputs(model_dir, timed=False)
    task = args.task or read_task(model_dir)
    for f in inputs:
        log(f"input: {f}")
    n = threads()
    pred_path = record.prediction_path(model_dir, args.size, runtime)
    pred_path.unlink(missing_ok=True)
    spec = {"adapter": str(model_dir / "adapter.py"), "size": args.size, "inputs": [str(f) for f in inputs],
            "warmup": protocol.CPU_CHECK_WARMUP, "timed": protocol.CPU_CHECK_TIMED, "mode": "check", "task": task,
            "runtime": runtime, "threads": n, "prediction_out": str(pred_path)}
    base = {"model": model_id, "kind": kind, "runtime": runtime, "task": task, "threads": n}
    try:
        if runtime == "onnxruntime-cpu":
            from . import ort

            info = ort.run_prepare({"adapter": str(model_dir / "adapter.py"), "size": args.size, "model": model_id,
                                    "rebuild": bool(getattr(args, "rebuild", False))}, log)
            spec["onnx"] = info["onnx"]
            base["runtime_info"] = info["runtime"]
            base["onnx"] = info["onnx"]
            ex = info["runtime"]["export"]
            log(f"onnx: {info['onnx']} ({'kept' if info.get('reused') else 'exported'}), opset {ex['opset']}, sha256 {ex['onnx_sha256'][:12]}…")
        window = cpu.Window().start()
        time.sleep(1.0)
        log(f"cpu: {machine.cpu_model()}, {n} threads; {hostmem.available_mib()} MiB of memory free; "
            f"other programs use {window.others():.0%} of the CPU")
        log(f"running the worker on the CPU, the GPU hidden: load, {spec['warmup']} warm-up and {spec['timed']} timed calls")
        gpu = open_gpu(args.gpu)
        try:
            worker, _ = spawn(spec, "mtr_measure.worker", gpu, log)
        finally:
            if gpu is not None:
                gpu.close()
    except OverLimit as e:
        log(f"OVER_LIMIT: {e}")
        check = {**base, "over_limit": {"call_s": round(e.call_s, 1)}, "weight_files": e.beat.get("weights", [])}
        path = write_check(model_dir, args.size, check, runtime)
        log(f"wrote {path}. A quiet `mtr run --runtime {runtime}` confirms it and records that this checkpoint isn't timed on a CPU")
        return finish(log, log_path, "OK", f"{model_id} / {args.size} on {runtime}: one call ran past {protocol.CPU_CALL_LIMIT_S:.0f} s",
                      {"check": str(path), "over_limit": True}, [str(e)])
    except (WorkerFailed, PrepareFailed) as e:
        log(f"FAILED: {e}")
        if e.code in ("CPU_UNSUPPORTED", "PRECISION"):
            files = e.worker.get("weights", []) if isinstance(e, WorkerFailed) else e.weights
            check = {**base, "not_timed": {"code": e.code, "reason": NOT_TIMED[e.code], "detail": str(e)}, "weight_files": files}
            path = write_check(model_dir, args.size, check, runtime, ok=False)
            log(f"wrote {path}. `mtr run --runtime {runtime}` records why this row has no number")
        return finish(log, log_path, e.code, str(e))
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))

    approx_ms = round(sorted(worker["times_ms"])[len(worker["times_ms"]) // 2], 1) if worker["times_ms"] else None
    ram = hostmem.peak_mib(worker["memory"])
    for line in worker.get("checks", []):
        log(line)
    log(f"package: {worker['env']['package']}, {runtime} {worker['runtime_version']}, precision {worker['precision']}, "
        f"{worker['threads']} threads, input {worker['input']}, load {worker['load_s']:.1f} s")
    log(f"output of predict(): {worker['output']}")
    log(f"approximate: {approx_ms} ms per call, {ram} MiB of memory at its peak (not measurements: other programs share the CPU)")
    notes = {"image_match": image_match(worker, pred_path.with_suffix(".png"))} if gen else {}
    warnings: list[str] = []
    check = {**base, "precision": worker["precision"], "input": worker["input"], "package": worker["env"]["package"],
             "runtime_version": worker["runtime_version"], "load_s": round(worker["load_s"], 2), "approx_ms": approx_ms,
             "approx_ram_mib": ram, "threads": worker["threads"],
             "weight_files": sorted(set(worker["weights"]) | set(worker.get("weight_links", [])))}
    try:
        m = match_result(model_dir, args.size, runtime, kind, notes)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))
    if m:
        from . import match

        log(f"match with the GPU: {match.describe({'metrics': m['metrics'], 'tolerance': m['tolerance']})}" if m["reference"] else
            f"image: {m['metrics']['width']:.0f}x{m['metrics']['height']:.0f}, not blank. Look at it: {pred_path.with_suffix('.png')}")
    else:
        warnings.append("No match with the GPU's output yet: run `mtr check` on the GPU for this size (it saves the reference), "
                        "then this check again. The queue refuses the row until then")
    check["match"] = m
    path = write_check(model_dir, args.size, check, runtime)
    log(f"wrote {path}")
    log(f"next: `mtr queue add {model_id} --size {args.size} --runtime {runtime}`")
    return finish(log, log_path, "OK", f"{model_id} / {args.size} works on {runtime}: load {worker['load_s']:.1f} s, about {approx_ms} ms and {ram} MiB",
                  {"check": str(path), "approx_ms": approx_ms, "approx_ram_mib": ram, "matched": bool(m)}, warnings)


# --- run -----------------------------------------------------------------------------------------


def calls_for(kind: str, check: dict) -> tuple[int, int]:
    """Warm-up and timed calls for this row: generation's, or the slow tier's when one call takes over CPU_SLOW_CALL_S."""
    if kind == "generation":
        return protocol.CPU_GEN_WARMUP, protocol.CPU_GEN_TIMED
    approx = check.get("approx_ms")
    if check.get("over_limit") or (approx and approx > protocol.CPU_SLOW_CALL_S * 1000):
        return protocol.CPU_SLOW_WARMUP, protocol.CPU_SLOW_TIMED
    return protocol.CPU_WARMUP, protocol.CPU_TIMED


def estimate_s(check: dict) -> float:
    """How long a CPU row's timed run takes, from its check: three runs of load, the warm-up floor and the timed calls."""
    from .runqueue import BASELINE_S as QUEUE_BASELINE_S, WORKER_START_S

    if check.get("kind") == "llm":
        return protocol.RUNS * (float(check.get("approx_run_s") or 600) + BASELINE_S)
    if check.get("not_timed"):
        return 60.0
    load = float(check.get("load_s") or 0)
    if check.get("over_limit"):
        return load + protocol.CPU_CALL_LIMIT_S + 60
    warmup, timed = calls_for(check.get("kind", "image"), check)
    call = float(check.get("approx_ms") or 0) / 1000
    per_run = WORKER_START_S + BASELINE_S + QUEUE_BASELINE_S + load + max(protocol.CPU_WARMUP_MIN_S, warmup * call) + timed * call
    return protocol.RUNS * per_run


def steadiness(times: list[float]) -> float | None:
    """|median of the first half - median of the second half| / median, or None with fewer than 10 calls."""
    if len(times) < 10:
        return None
    half = len(times) // 2
    return abs(statistics.median(times[:half]) - statistics.median(times[half:])) / statistics.median(times)


def cmd_run(args, model_dir: Path, model_id: str, kind: str, log, finish) -> int:
    from .runqueue import read_check

    runtime = args.runtime
    sfx = runtimes.suffix(runtime)
    out = (args.out or model_dir / "runs" / f"{args.size}{sfx}.json").resolve()
    failed_log = out.with_name(f"{args.size}{sfx}.failed.log")
    dry = args.dry_run or args.force
    warnings: list[str] = []
    log(f"mtr run {model_id} / {args.size} on {runtime}, protocol {protocol.PROTOCOL_VERSION} (the GPU hidden; nothing is paused)")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    if kind == "llm":
        from . import cpullm

        return cpullm.cmd_run(args, model_dir, model_id, log, finish, out, failed_log, dry)
    try:
        check = read_check(model_dir, args.size, runtime, allow_not_timed=True)
        m = require_ready(model_dir, args.size, runtime, check)
        inputs = [f.resolve() for f in args.inputs] if args.inputs else None
        if inputs is None:
            from .cli import default_inputs

            inputs = default_inputs(model_dir, timed=True)
        for f in inputs:
            log(f"input: {f} sha256 {record.sha256_file(f)}")
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, failed_log, e.code, str(e))
    warmup, timed = calls_for(kind, check)
    n = threads()
    facts = machine.cpu_facts()
    spec = {"adapter": str(model_dir / "adapter.py"), "size": args.size, "inputs": [str(f) for f in inputs], "warmup": warmup,
            "timed": timed, "runtime": runtime, "threads": n, "warmup_min_s": protocol.CPU_WARMUP_MIN_S}
    if runtime == "onnxruntime-cpu" and check.get("not_timed"):
        # Its ONNX file computes in half precision: a property of the file, which a run confirms by inspecting it again.
        from . import ort

        try:
            ort.run_prepare({"adapter": str(model_dir / "adapter.py"), "size": args.size, "model": model_id}, log)
        except PrepareFailed as e:
            if e.code in NOT_TIMED:
                return not_timed(args, model_dir, model_id, kind, runtime, e, check, facts, inputs, out, failed_log, dry, log, finish, warnings)
            log(f"FAILED: {e}")
            return finish(log, failed_log, e.code, str(e))
        return finish(log, failed_log, "NOT_CHECKED", "The ONNX file no longer gives the check's result. Run `mtr check --runtime onnxruntime-cpu` again")
    if runtime == "onnxruntime-cpu":
        from . import ort

        try:
            spec["onnx"] = ort.verify(check)
        except MeasureError as e:
            log(f"FAILED: {e}")
            return finish(log, failed_log, e.code, str(e))
    log(f"cpu: {facts['cpu']}, {facts['cpu_cores']} cores, {facts['cpu_threads']} logical CPUs, {n} threads, "
        f"{facts['ram_total_mib']} MiB of memory" + (f", sustained power limit {facts['cpu_power_limit_w']:g} W" if facts.get("cpu_power_limit_w") else ""))
    log(f"{protocol.RUNS} runs, each a fresh process: load, at least {warmup} warm-up calls and {protocol.CPU_WARMUP_MIN_S:.0f} s of them, "
        f"then {timed} timed calls")
    need = int(check.get("approx_ram_mib") or 0)
    runs: list[dict] = []
    clean = True
    gpu = open_gpu(args.gpu)
    try:
        for i in range(protocol.RUNS):
            window = cpu.Window().start()
            time.sleep(BASELINE_S)
            pre = checks.before_cpu_run(window.others(), hostmem.available_mib(), need)
            from .cli import report

            report(log, pre)
            warnings += pre.warnings
            if not pre.ok():
                clean = False
                if not args.force:
                    raise MeasureError(pre.codes[0], pre.errors[0])
            log(f"run {i + 1}/{protocol.RUNS}")
            throttled = hostmem.throttle_count()
            window = cpu.Window().start()
            try:
                worker, _ = spawn(spec, "mtr_measure.worker", gpu, log)
            except (OverLimit, WorkerFailed) as e:
                others = window.others()
                if e.code in NOT_TIMED and others <= checks.CPU_BUSY_LIMIT_CPU and pre.ok():
                    return not_timed(args, model_dir, model_id, kind, runtime, e, check, facts, inputs, out, failed_log, dry, log, finish, warnings)
                if e.code in NOT_TIMED:
                    raise MeasureError("CPU_BUSY", f"{e} while other programs used {others:.0%} of the CPU, so it isn't recorded. Run it again when the machine is quiet") from e
                raise
            others = window.others()
            post = checks.after_cpu_run(others, throttled, hostmem.throttle_count(), worker["memory"], steadiness(worker["times_ms"]))
            report(log, post)
            warnings += post.warnings
            if not post.ok():
                clean = False
                if not args.force:
                    raise MeasureError(post.codes[0], post.errors[0])
            times = worker["times_ms"]
            log(f"run {i + 1}: median {statistics.median(times):.3f} ms, {len(times)} timed calls after {worker['warmup']} warm-up, "
                f"ram peak {hostmem.peak_mib(worker['memory'])} MiB, {worker['threads']} threads, other cpu {others:.1%}")
            log(f"run {i + 1} timed iterations (ms): " + " ".join(f"{t:.3f}" for t in times), echo=False)
            runs.append({"worker": worker})
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, failed_log, e.code, str(e), warnings=warnings)
    finally:
        if gpu is not None:
            gpu.close()

    if not m and kind != "generation":
        # The check stopped a call past the limit, and this quiet run didn't: the row has a number, but no match yet.
        return finish(log, failed_log, "NOT_MATCHED", "Every call ran within the limit this time, but the check never matched the GPU's "
                                                      f"output. Run `mtr check --runtime {runtime}` again on a quiet machine", warnings=warnings)
    if not m:
        return finish(log, failed_log, "NOT_MATCHED", f"No image was checked for this row. Run `mtr check --runtime {runtime}` again", warnings=warnings)
    worker = runs[-1]["worker"]
    try:
        weights = record.weight_entries([Path(p) for p in worker["weights"]])
    except FileNotFoundError as e:
        return finish(log, failed_log, "WEIGHTS_MISSING", str(e))
    rt = runtime_block(runtime, worker, check, m)
    rec = record.build_cpu(model=model_id, size=args.size, weights=weights, runs=runs, runtime=rt, warmup=warmup,
                           measured=datetime.now().astimezone().isoformat(timespec="seconds"), machine=facts,
                           provenance=machine.provenance(model_dir, inputs), kind=kind)
    meas = rec["measurement"]
    spread = record.spread(meas["runs"], "median")
    log(f"latency: median {meas['latency_ms']['median']} ms, p90 {meas['latency_ms']['p90']} ms (medians of {len(runs)} runs; "
        f"spread {spread:.1%}, the protocol allows {protocol.MAX_SPREAD:.0%})")
    log(f"ram peak: {meas['ram_peak_mib']} MiB, loading included; {rt['threads']} threads; precision {meas['precision']}")
    return write(rec, runs, kind, spread, clean, dry, args, out, failed_log, log, finish, warnings,
                 {"latency_median_ms": meas["latency_ms"]["median"], "latency_p90_ms": meas["latency_ms"]["p90"],
                  "ram_peak_mib": meas["ram_peak_mib"], "spread": round(spread, 4)},
                 f"median {meas['latency_ms']['median']} ms on the CPU, RAM peak {meas['ram_peak_mib']} MiB")


def runtime_block(runtime: str, worker: dict, check: dict, m: dict | None) -> dict:
    rt: dict = {"name": runtime, "version": worker["runtime_version"], "status": "runs-as-is", "threads": int(worker["threads"]), "match": m}
    info = check.get("runtime_info") or {}
    if info.get("status"):
        rt["status"] = info["status"]
    if info.get("patch"):
        rt["patch"] = info["patch"]
    if info.get("export"):
        rt["export"] = info["export"]
    return rt


def write(rec: dict, runs: list[dict], kind: str, spread: float, clean: bool, dry: bool, args, out: Path, failed_log: Path,
          log, finish, warnings: list[str], numbers: dict, headline: str) -> int:
    code, message = "OK", ""
    problems = record.validate(rec, "cpu-run-record")
    for prob in problems:
        log(f"SCHEMA: {prob}")
    if problems:
        code, message = "SCHEMA", f"The record fails the schema: {problems[0]}"
    elif spread > protocol.MAX_SPREAD:
        code, message = "VARIANCE", f"The {len(runs)} runs spread by {spread:.1%}, above the {protocol.MAX_SPREAD:.0%} the protocol allows"
    elif not clean:
        code, message = "FAILED", "A check failed, so this run can't be a record"
    if dry or code != "OK":
        log(f"No run record written ({message or ('--force' if args.force else '--dry-run')}). The record would have been:")
        log(json.dumps(rec, indent=2))
        if code == "OK":
            return finish(log, out.with_name(out.name[: -len(".json")] + ".dry-run.log"), "OK", f"Dry run of {rec['model']} / {rec['size']}: no record written", numbers, warnings)
        return finish(log, failed_log, code, message, numbers, warnings)
    if rec.get("provenance", {}).get("tool", {}).get("dirty"):
        warnings.append("tools/measure or the model directory has uncommitted changes, so this record can't be submitted")
    outputs = {"record": str(out), **numbers}
    out.parent.mkdir(parents=True, exist_ok=True)
    times_path = out.with_name(out.name[: -len(".json")] + ".times.json")
    if runs:
        times = record.times(runs, kind)
        problems = record.validate(times, "run-times")
        if problems:
            return finish(log, failed_log, "SCHEMA", f"The timings fail their schema: {problems[0]}", numbers, warnings)
        times_path.write_text(json.dumps(times) + "\n")
        outputs["times"] = str(times_path)
    else:
        times_path.unlink(missing_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {out}" + (f" and {times_path.name}" if runs else ""))
    return finish(log, out.with_suffix(".log"), "OK", f"{rec['model']} / {rec['size']}: {headline}", outputs, warnings)


def runtime_version(runtime: str) -> str:
    """The runtime's version in this environment, for a record whose worker never got to report it."""
    import importlib.metadata

    if runtime == "llamacpp-cpu":
        from .engine import LLAMA_CPP

        return LLAMA_CPP["build"]
    try:
        return importlib.metadata.version("onnxruntime" if runtime == "onnxruntime-cpu" else "torch").split("+")[0]
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def not_timed(args, model_dir: Path, model_id: str, kind: str, runtime: str, e: MeasureError, check: dict, facts: dict,
              inputs: list[Path] | None, out: Path, failed_log: Path, dry: bool, log, finish, warnings: list[str]) -> int:
    """The record of a CPU row with no number: the run's own finding, on a quiet machine."""
    reason = NOT_TIMED[e.code]
    if isinstance(e, OverLimit):
        files = e.beat.get("weights")
    elif isinstance(e, PrepareFailed):
        files = e.weights
    else:
        files = e.worker.get("weights")
    files = files or check.get("weight_files") or []
    try:
        weights = record.weight_entries([Path(p) for p in files])
    except (FileNotFoundError, ValueError) as err:
        return finish(log, failed_log, "WEIGHTS_MISSING", str(err))
    env = {k: check[k] for k in ("package",) if check.get(k)}
    version = check.get("runtime_version") or (check.get("runtime_info") or {}).get("version") or runtime_version(runtime)
    rec = record.build_not_timed(
        model=model_id, size=args.size, weights=weights, kind=kind, runtime=runtime, version=version, reason=reason, detail=str(e),
        measured=datetime.now().astimezone().isoformat(timespec="seconds"), machine=facts, env={**env, "python": sys.version.split()[0]},
        input=check.get("input"), limit_s=protocol.CPU_CALL_LIMIT_S if reason == "over-limit" else None,
        call_s=getattr(e, "call_s", None), provenance=machine.provenance(model_dir, inputs),
    )
    log(f"{e.code}: {e}")
    log(f"no number for this row: {reason}. The record says so")
    return write(rec, [], kind, 0.0, True, dry, args, out, failed_log, log, finish, warnings, {"not_timed": reason},
                 f"not timed on the CPU ({reason})")
