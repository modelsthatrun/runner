"""The worker process: loads the adapter and the checkpoint, then times predict().

The parent starts it after taking the idle baseline and polls device memory while it runs.
Running the model in its own process means its CUDA context is gone when it exits, so the
parent can take the baseline again afterwards. A timed measurement starts it once per run.
Usage: python -m mtr_measure.worker <spec.json>

In "check" mode it makes a few calls and also checks the weight files and, when the spec names a
task, that to_prediction() returns the task's standard type. Whatever fails, it writes a result
with a code (errors.py), so the parent can report it.
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
from .errors import MeasureError


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


def run(spec: dict) -> dict:
    a = ad.load_adapter(Path(spec["adapter"]))
    size = spec["size"]
    ad.check_size(a, size)
    precision = ad.choose_precision(list(a.PRECISIONS))
    x = ad.prepare_input(a, [Path(p) for p in spec["inputs"]])

    import torch

    if not torch.cuda.is_available():
        raise MeasureError("FAILED", "torch can't see a CUDA device")
    sync = torch.cuda.synchronize

    t = time.perf_counter()
    model = a.load(size, precision)
    sync()
    load_s = time.perf_counter() - t

    output = "n/a"
    for i in range(spec["warmup"]):
        out = a.predict(model, x)
        sync()
        if i == 0:
            output = describe(out)
        del out

    times_ms: list[float] = []
    for _ in range(spec["timed"]):
        sync()
        t0 = time.perf_counter()
        a.predict(model, x)
        sync()
        times_ms.append((time.perf_counter() - t0) * 1000)

    result = {
        "precision": precision,
        "input": ad.input_size(a, size, x),
        "latency_scope": ad.latency_scope(a),
        "times_ms": times_ms,
        "torch_max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
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
        "device": torch.cuda.get_device_name(),
    }
    if spec.get("mode") == "check":
        result["checks"] = check_more(a, model, x, size, spec.get("task"))
    return result


def check_more(a: Any, model: Any, x: Any, size: str, task: str | None) -> list[str]:
    """What `mtr check` verifies beyond a working predict(). Raises on the first problem."""
    notes = []
    missing = [str(p) for p in a.weight_files(size) if not Path(p).is_file()]
    if missing:
        raise MeasureError("WEIGHTS_MISSING", f"weight_files() lists files that don't exist: {', '.join(missing)}")
    if not a.weight_files(size):
        raise MeasureError("ADAPTER_INTERFACE", "weight_files() returned no files")
    notes.append(f"weight files: {len(a.weight_files(size))} present")
    if task:
        pred = ad.to_prediction(a, model, x, a.predict(model, x), task)
        # Alpha and Depth are one map per image, so they have no length.
        what = f"with {len(pred)} item(s)" if hasattr(pred, "__len__") else "of one map"
        notes.append(f"to_prediction(): {type(pred).__name__} {what} for {task}")
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
    Path(spec["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
