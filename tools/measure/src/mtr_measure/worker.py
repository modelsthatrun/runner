"""The worker process: loads the adapter and the checkpoint, then times predict().

The parent starts it after taking the idle baseline and polls device memory while it runs.
Running the model in its own process means its CUDA context is gone when it exits, so the
parent can take the baseline again afterwards. A timed measurement starts it once per run.
Usage: python -m mtr_measure.worker <spec.json>

In "check" mode it makes a few calls and also checks the weight files and, when the spec names a
task, the output: to_prediction() returns the task's standard type, and predict() finds something
on the task's timing input. Whatever fails, it writes a result with a code (errors.py), so the
parent can report it.

A CPU row (runtimes.CPU_RUNTIMES) runs with the GPU hidden: the parent sets CUDA_VISIBLE_DEVICES to nothing and
MTR_DEVICE to cpu. The worker then asks for fp32, sets torch's threads, watches the first warm-up call for half
precision (which a CPU only emulates), writes a heartbeat before every call so the parent can stop one that runs
past the protocol's limit, and reports its own peak memory.
"""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from . import adapter as ad
from . import runtimes
from .errors import MeasureError

# What a failed worker can still tell the parent: the weight files, for a CPU row's record of why it has no number.
_state: dict[str, Any] = {}


def describe(value: Any, depth: int = 0) -> str:
    """A short description of predict()'s output for the log, whatever its type."""
    if depth > 2:
        return "…"
    shape = getattr(value, "shape", None)
    if shape is not None:
        device = getattr(value, "device", None)
        dtype = getattr(value, "dtype", None)
        return f"{type(value).__name__}{list(shape)} {dtype}{f' {device}' if device is not None else ''}"
    if isinstance(value, dict):
        items = [f"{k}: {describe(v, depth + 1)}" for k, v in list(value.items())[:6]]
        return "{" + ", ".join(items) + (", …" if len(value) > 6 else "") + "}"
    if isinstance(value, (list, tuple)):
        inner = ", ".join(describe(v, depth + 1) for v in value[:3])
        return f"{type(value).__name__}[{len(value)}]({inner}{', …' if len(value) > 3 else ''})"
    return type(value).__name__


class Heartbeat:
    """Before each call, the call's number and start time in a small file the parent reads (cli.py, cpurun.py)."""

    def __init__(self, path: str | None, info: dict):
        self.path = Path(path) if path else None
        self.info = info

    def __call__(self, phase: str, n: int = 0) -> None:
        if self.path:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps({**self.info, "phase": phase, "call": n, "started": time.time()}))
            tmp.replace(self.path)


def half_precision_probe(fn) -> tuple[Any, dict[str, int], int]:
    """Runs fn() and returns its output, the ops that produced fp16 or bf16 tensors (with counts), and how many ops ran.
    The probe sits in torch's dispatcher, so it sees autocast, .half() and TorchScript alike."""
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    halves = (torch.float16, torch.bfloat16)
    seen: dict[str, int] = {}
    total = [0]

    class Probe(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            total[0] += 1
            for t in tree_flatten(out)[0]:
                if isinstance(t, torch.Tensor) and t.dtype in halves:
                    name = str(func.overloadpacket.__name__) if hasattr(func, "overloadpacket") else str(func)
                    seen[name] = seen.get(name, 0) + 1
                    break
            return out

    with Probe():
        out = fn()
    return out, seen, total[0]


def run(spec: dict) -> dict:
    a = ad.load_adapter(Path(spec["adapter"]))
    size = spec["size"]
    ad.check_size(a, size)
    runtime = spec.get("runtime", runtimes.DEFAULT)
    cpu = runtimes.is_cpu(runtime)
    _state["weights"] = [str(Path(p).resolve()) for p in a.weight_files(size)]
    if runtime == "tensorrt":
        from . import trt

        # The engine's precision is the official TensorRT step's, not the PyTorch row's (docs/protocol.md, "Runtimes").
        precision = trt.adapter_spec(a)["precision"]
    elif cpu:
        # fp32 is the only precision every CPU runs natively (docs/protocol.md, "CPU rows").
        precision = "fp32"
    else:
        precision = ad.choose_precision(list(a.PRECISIONS))
    gen = ad.kind(a) == "generation"
    if gen:
        from . import generation, protocol

        # The tool makes the call's input from the timing prompt and the adapter's GENERATION.
        x = generation.make(a, size, generation.read_prompt([Path(p) for p in spec["inputs"]]), protocol.GEN_SEED)
        if cpu and x.generator == "cuda":
            # The GPU is hidden. A generator on the CPU draws other noise for the same seed, so the image differs from the
            # GPU row's, and the cost of the call doesn't.
            from dataclasses import replace

            x = replace(x, generator="cpu")
    else:
        x = ad.prepare_input(a, [Path(p) for p in spec["inputs"]])

    import torch

    if cpu:
        if torch.cuda.is_available():
            raise MeasureError("GPU_TOUCHED", "torch can see a CUDA device in a CPU row's worker; the GPU should be hidden")
        torch.set_num_threads(int(spec["threads"]))

        def sync() -> None:
            pass
    else:
        if not torch.cuda.is_available():
            raise MeasureError("FAILED", "torch can't see a CUDA device")
        if precision == "bf16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise MeasureError("FAILED", "This GPU only emulates bf16, and the protocol doesn't time an emulation (docs/protocol.md, \"Scope\")")
        sync = torch.cuda.synchronize

    beat = Heartbeat(spec.get("heartbeat"), {"weights": _state["weights"]})
    t = time.perf_counter()
    beat("load")
    try:
        if runtime == "tensorrt":
            model = a.load_engine(size, Path(spec["engine"]))
        elif runtime == "onnxruntime-cpu":
            # The session is checked after the timed calls (threads_used): a package may make it on the first call.
            model = a.load_onnx(size, Path(spec["onnx"]))
        else:
            model = a.load(size, precision)
    except MeasureError:
        raise
    except Exception as e:
        unsupported(a, cpu, e)
        raise
    sync()
    load_s = time.perf_counter() - t

    output = "n/a"
    half: dict[str, int] = {}
    probed_ops = None
    caller_cpus = None
    warm_start = time.perf_counter()
    warmup_min_s = float(spec.get("warmup_min_s", 0))
    # A PyTorch CPU row's second warm-up call runs under the precision probe. Not the first: a package may trace or
    # compile its model on the first call (RF-DETR's TorchScript does), and tracing under the probe fails.
    probe_at = 1 if cpu and runtime == "pytorch-cpu" else None
    warmup = max(int(spec["warmup"]), 2) if probe_at is not None else int(spec["warmup"])
    i = 0
    while i < warmup or (i > 0 and time.perf_counter() - warm_start < warmup_min_s):
        beat("warmup", i)
        try:
            if i == probe_at:
                out, half, probed_ops = half_precision_probe(lambda: a.predict(model, x))
            else:
                out = a.predict(model, x)
        except MeasureError:
            raise
        except Exception as e:
            if i == 0:
                unsupported(a, cpu, e)
            raise
        sync()
        if i == 0:
            output = describe(out)
            if runtime == "onnxruntime-cpu":
                from . import ort

                caller_cpus = ort.pin_caller()
        if i == probe_at:
            if half:
                ops = ", ".join(f"{k} ({v})" for k, v in sorted(half.items(), key=lambda kv: -kv[1])[:6])
                raise MeasureError("PRECISION", f"On the CPU the official code computed in half precision: {ops}. A CPU only emulates "
                                                "fp16 and bf16, so this row isn't timed")
            if probed_ops == 0:
                raise MeasureError("ADAPTER_RUNTIME", "The precision probe saw no torch operation in predict(), so it can't say what ran")
        del out
        i += 1
    warmup_done = i
    if getattr(a, "CPU_UNSUPPORTED", None) and cpu:
        raise MeasureError("ADAPTER_INTERFACE", f"The adapter declares CPU_UNSUPPORTED, but the call ran on the CPU. Remove the declaration")

    times_ms: list[float] = []
    # A CPU row's timed calls also last at least timed_min_s, so a fast model's median covers its slow stretches too.
    timed_min_s = float(spec.get("timed_min_s", 0))
    timed_start = time.perf_counter()
    n = 0
    while n < int(spec["timed"]) or time.perf_counter() - timed_start < timed_min_s:
        beat("timed", n)
        sync()
        t0 = time.perf_counter()
        a.predict(model, x)
        sync()
        times_ms.append((time.perf_counter() - t0) * 1000)
        n += 1

    result = {
        "precision": precision,
        "runtime": runtime,
        "input": ad.input_size(a, size, x, runtime),
        "latency_scope": ad.latency_scope(a),
        "times_ms": times_ms,
        "warmup": warmup_done,
        "torch_max_reserved_bytes": None if cpu else int(torch.cuda.max_memory_reserved()),
        "weights": [str(Path(p).resolve()) for p in a.weight_files(size)],
        "weight_links": [str(Path(p).absolute()) for p in a.weight_files(size)],
        "env": {
            "cuda": str(torch.version.cuda),
            "torch": torch.__version__,
            "python": platform.python_version(),
            "package": ad.package(a),
        },
        "load_s": load_s,
        "output": output,
        "device": "cpu" if cpu else torch.cuda.get_device_name(),
    }
    if cpu:
        from . import hostmem

        result["threads"] = threads_used(a, model, runtime)
        result["memory"] = hostmem.proc_status()
        if runtime == "onnxruntime-cpu":
            result["runtime_version"] = onnxruntime_version()
            result["onnx"] = spec["onnx"]
            result["caller_cpus"] = caller_cpus
        else:
            result["runtime_version"] = torch.__version__.split("+")[0]
    if gen:
        result["generation"] = generation.recorded(x, generation.scheduler(a, model))
    if spec.get("mode") == "check":
        result["checks"] = check_more(a, model, x, size, spec.get("task"), [Path(p) for p in spec["inputs"]], spec.get("prediction_out"), beat,
                                      spec.get("onnx_reference_out"))
    return result


def unsupported(a: Any, cpu: bool, e: BaseException) -> None:
    """On a CPU row, a load or first call that fails in an adapter that declares CPU_UNSUPPORTED is the declared result."""
    why = getattr(a, "CPU_UNSUPPORTED", None)
    if cpu and why:
        raise MeasureError("CPU_UNSUPPORTED", f"{why} ({type(e).__name__}: {str(e)[:300]})") from e


def threads_used(a: Any, model: Any, runtime: str) -> int:
    """The threads the computation ran on, read back after the timed calls: torch's, or the ONNX Runtime session's."""
    if runtime == "onnxruntime-cpu":
        from . import ort

        return ort.check_session(a.onnx_session(model))
    import torch

    return int(torch.get_num_threads())


def onnxruntime_version() -> str:
    import onnxruntime

    return str(onnxruntime.__version__)


def check_more(a: Any, model: Any, x: Any, size: str, task: str | None, inputs: list[Path] | None = None,
               prediction_out: str | None = None, beat: Any = None, onnx_reference_out: str | None = None) -> list[str]:
    """What `mtr check` verifies beyond a working predict(). Raises on the first problem. `beat` marks the start of its own
    predict() call, so a CPU row's limit counts that call alone."""
    beat = beat or (lambda phase, n=0: None)
    notes = []
    missing = [str(p) for p in a.weight_files(size) if not Path(p).is_file()]
    if missing:
        raise MeasureError("WEIGHTS_MISSING", f"weight_files() lists files that don't exist: {', '.join(missing)}")
    if not a.weight_files(size):
        raise MeasureError("ADAPTER_INTERFACE", "weight_files() returned no files")
    notes.append(f"weight files: {len(a.weight_files(size))} present")
    if ad.kind(a) == "generation":
        from . import generation

        beat("check")
        img = generation.check_image(a.predict(model, x), x)
        notes.append(f"predict(): a {img.size[0]}x{img.size[1]} image, {x.steps} steps at guidance {x.guidance:g}, seed {x.seed} on {x.generator}")
        if prediction_out:
            Path(prediction_out).parent.mkdir(parents=True, exist_ok=True)
            img.save(Path(prediction_out).with_suffix(".png"))
        return notes
    if task:
        from . import timing
        from .predictions import PredictionError

        try:
            beat("check")
            pred = ad.to_prediction(a, model, x, a.predict(model, x), task)
        except PredictionError as e:
            raise MeasureError("ADAPTER_RUNTIME", f"to_prediction() on predict()'s output: {e}") from e
        # Alpha and Depth are one map per image, so they have no length.
        what = f"with {len(pred)} item(s)" if hasattr(pred, "__len__") else "of one map"
        notes.append(f"to_prediction(): {type(pred).__name__} {what} for {task}")
        if hasattr(pred, "__len__") and len(pred) == 0:
            # A display threshold also drops NaN scores, so a model whose outputs are all NaN gets here with nothing
            # found. And a call that finds nothing skips the work that depends on what was found.
            default = timing.input_for(task)
            if default is not None and inputs and [p.resolve() for p in inputs] == [default.resolve()]:
                raise MeasureError(
                    "ADAPTER_RUNTIME",
                    f"predict() found nothing on {default.name}, the input every {task} model is timed on. A score threshold "
                    "also drops NaN scores, so look at the model's raw outputs before timing it",
                )
            notes.append("predict() found nothing on this input")
        if prediction_out:
            from . import match

            try:
                match.save(pred, task, Path(prediction_out))
                notes.append(f"prediction saved for the CPU rows' comparison: {Path(prediction_out).name}")
            except match.MatchError as e:
                notes.append(f"prediction not saved: {e}")
            if onnx_reference_out and callable(getattr(a, "predict_onnx_input", None)):
                # The same model on the official ONNX code's input, so an ONNX Runtime row is compared on equal input.
                beat("check")
                onnx_pred = ad.to_prediction(a, model, x, a.predict_onnx_input(model, x), task)
                match.save(onnx_pred, task, Path(onnx_reference_out))
                notes.append(f"the same on the ONNX code's input: {type(onnx_pred).__name__} with {len(onnx_pred) if hasattr(onnx_pred, '__len__') else 1} item(s), "
                             f"saved as {Path(onnx_reference_out).name}")
    else:
        notes.append("to_prediction(): not checked (the adapter declares no TASK and there was no --task)")
    return notes


def is_oom(e: BaseException) -> bool:
    if type(e).__name__ == "OutOfMemoryError":
        return True
    return "out of memory" in str(e).lower()


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = {"ok": True, **run(spec)}
    except MeasureError as e:
        result = {"ok": False, "code": e.code, "error": str(e)}
    except Exception as e:  # the adapter's own code failed; keep the traceback for the log
        code = "OOM" if is_oom(e) else "ADAPTER_RUNTIME"
        result = {"ok": False, "code": code, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    if not result["ok"] and "weights" in _state:
        result["weights"] = _state["weights"]
    Path(spec["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
