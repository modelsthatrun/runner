"""ONNX Runtime rows on the CPU (docs/protocol.md, "CPU rows"): the authors' own ONNX export of the released weights,
or the ONNX file they publish, run by their own ONNX Runtime inference code on the CPU.

    mtr check <model> --size <size> --runtime onnxruntime-cpu [--rebuild]    export (or download) and check it
    mtr run <model> --size <size> --runtime onnxruntime-cpu                  time it: runs/<size>.onnxruntime-cpu.json

An adapter with an ONNX Runtime row declares (docs/adapters.md):

    ONNXRUNTIME: dict            {"status": "runs-as-is"}, plus "patch": "fixes/<file>" when status is runs-with-fixes,
                                 and "input": "WxH" (or one per size) when the official ONNX inference code feeds another
                                 input than the PyTorch row's
    onnx_file(size, out)         the ONNX file the official ONNX Runtime code runs, written into `out`: the official export
                                 of the released checkpoint, or the file the authors publish. Returns {"onnx": Path,
                                 "exporter": "the export, or where the file comes from", "args": [...]}. Without it,
                                 export_onnx(), the TensorRT row's export (trt.py), when the two rows share one
    load_onnx(size, onnx)        an onnxruntime.InferenceSession made the way the official inference code makes it, and
                                 whatever predict() needs. predict() and to_prediction() take what it returns
    onnx_session(model)          the InferenceSession in what load_onnx() returned, for the tool to check: it computes on
                                 the CPU provider, and in fp32

The export runs in its own process with the GPU hidden. ONNX files stay in ~/.cache/modelsthatrun/onnx/<model>/<size>/
(MTR_ONNX moves it), never in the checkout, and are never published. `check` keeps a file and exports again only with
--rebuild, or when the weights or the export changed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

from .errors import MeasureError

STATUSES = ("runs-as-is", "runs-with-fixes")
# What a failed prepare can still tell the parent: the weight files, for a record of why the row has no number.
_state: dict[str, Any] = {}


class PrepareFailed(MeasureError):
    def __init__(self, code: str, message: str, weights: list[str]):
        super().__init__(code, message)
        self.weights = weights


def root() -> Path:
    return Path(os.environ.get("MTR_ONNX", "~/.cache/modelsthatrun/onnx")).expanduser()


def model_root(model_id: str) -> Path:
    return root() / model_id


def onnx_dir(model_id: str, size: str) -> Path:
    return model_root(model_id) / size


def adapter_spec(a: Any) -> dict:
    """The adapter's ONNXRUNTIME, checked. Raises when the adapter has no ONNX Runtime row."""
    spec = getattr(a, "ONNXRUNTIME", None)
    if spec is None:
        raise MeasureError("USAGE", "This model has no ONNX Runtime row: its adapter declares no ONNXRUNTIME, because its authors "
                                    "publish no ONNX export with ONNX Runtime inference code (docs/adapters.md)")
    sources = [n for n in ("onnx_file", "export_onnx") if callable(getattr(a, n, None))]
    hooks = all(callable(getattr(a, n, None)) for n in ("load_onnx", "onnx_session"))
    if not isinstance(spec, dict) or not hooks or not sources:
        raise MeasureError("ADAPTER_INTERFACE", "An adapter with ONNXRUNTIME needs load_onnx(), onnx_session() and onnx_file() (or export_onnx())")
    status = spec.get("status", "runs-as-is")
    if status not in STATUSES:
        raise MeasureError("ADAPTER_INTERFACE", f"ONNXRUNTIME['status'] must be one of {STATUSES}, got {status!r}")
    if status == "runs-with-fixes" and not spec.get("patch"):
        raise MeasureError("ADAPTER_INTERFACE", "ONNXRUNTIME['status'] is runs-with-fixes, so it needs 'patch': the file under fixes/")
    return {"status": status, "patch": spec.get("patch"), "source": sources[0]}


def prepare(spec: dict, say=print) -> dict:
    """Export (or fetch) the ONNX file in this process, or reuse the kept one. Returns onnx.json's content."""
    from . import adapter as ad
    from .record import sha256_file
    from .trt import onnx_opset, self_contained

    a = ad.load_adapter(Path(spec["adapter"]))
    size = spec["size"]
    ad.check_size(a, size)
    rt = adapter_spec(a)
    _state["weights"] = [str(Path(p).resolve()) for p in a.weight_files(size)]
    out = onnx_dir(spec["model"], size)
    weights = [{"file": Path(p).name, "sha256": sha256_file(Path(p))} for p in a.weight_files(size) if Path(p).is_file()]
    key = {"weights": weights, "source": rt["source"], "status": rt["status"], "patch": rt["patch"], "package": ad.package(a)}
    path = out / "onnx.json"
    if path.is_file() and not spec.get("rebuild"):
        info = json.loads(path.read_text())
        onnx = Path(info["onnx"])
        if info.get("key") == key and onnx.is_file() and sha256_file(onnx) == info["runtime"]["export"]["onnx_sha256"]:
            say(f"onnx: kept {onnx} (sha256 {info['runtime']['export']['onnx_sha256'][:12]}…). --rebuild exports it again")
            return {**info, "reused": True}
        say("onnx: the kept file doesn't match the weights or the export; exporting again")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    say(f"onnx: {rt['source']}({size!r}) into {out}")
    t0 = time.perf_counter()
    exported = getattr(a, rt["source"])(size, out)
    onnx_path = Path(exported["onnx"])
    if not onnx_path.is_file():
        raise MeasureError("ADAPTER_RUNTIME", f"{rt['source']}() returned {onnx_path}, which doesn't exist")
    self_contained(onnx_path)
    half = half_precision(onnx_path)
    if half:
        raise MeasureError("PRECISION", f"{onnx_path.name} holds {half}. A CPU only emulates fp16 and bf16, so this row isn't timed")
    opset = onnx_opset(onnx_path)
    say(f"onnx: {onnx_path.name}, {onnx_path.stat().st_size / 2**20:.1f} MiB, opset {opset}, {time.perf_counter() - t0:.0f} s")
    info = {
        "key": key,
        "runtime": {
            "name": "onnxruntime-cpu",
            "status": rt["status"],
            "patch": rt["patch"],
            "export": {
                "exporter": str(exported["exporter"]),
                "args": [str(x) for x in exported.get("args", [])],
                "opset": opset,
                "onnx_sha256": sha256_file(onnx_path),
                "onnx_bytes": onnx_path.stat().st_size,
            },
        },
        "onnx": str(onnx_path),
        "made": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    path.write_text(json.dumps(info, indent=2) + "\n")
    return {**info, "reused": False}


def half_precision(path: Path) -> str:
    """What in the ONNX graph is fp16 or bf16: its inputs, outputs or weights. Empty when it is all fp32."""
    try:
        import onnx
    except ImportError:
        return ""
    m = onnx.load(str(path), load_external_data=False)
    halves = {onnx.TensorProto.FLOAT16: "fp16", onnx.TensorProto.BFLOAT16: "bf16"}
    found = [f"{halves[t.data_type]} weight {t.name}" for t in m.graph.initializer if t.data_type in halves]
    for v in list(m.graph.input) + list(m.graph.output):
        if v.type.tensor_type.elem_type in halves:
            found.append(f"{halves[v.type.tensor_type.elem_type]} {v.name}")
    return ", ".join(found[:5]) + (f" and {len(found) - 5} more" if len(found) > 5 else "")


def check_session(sess: Any) -> int:
    """The session computes on the CPU provider alone, and takes fp32. Returns its intra-op threads: ONNX Runtime's
    default of 0 means one per physical core."""
    from . import hostmem

    providers = list(sess.get_providers())
    if providers != ["CPUExecutionProvider"]:
        raise MeasureError("GPU_TOUCHED", f"The ONNX Runtime session runs on {', '.join(providers)}, not the CPU provider alone")
    half = [i.name for i in sess.get_inputs() if i.type in ("tensor(float16)", "tensor(bfloat16)")]
    if half:
        raise MeasureError("PRECISION", f"The ONNX file takes {', '.join(half)} in half precision, which a CPU only emulates")
    n = int(sess.get_session_options().intra_op_num_threads)
    return n or hostmem.cores()[0]


def free_core(pinned: list[set[int]], cores: list[set[int]], allowed: set[int]) -> set[int] | None:
    """The logical CPUs of the first physical core that no pinned thread holds, or None when no thread is pinned or
    every core is held."""
    if not pinned:
        return None
    held = set().union(*pinned)
    for core in cores:
        if core <= allowed and not core & held:
            return set(core)
    return None


def pin_caller() -> list[int] | None:
    """Keep the calling thread on the physical core ONNX Runtime's default thread pool leaves free.

    With its default threads, ONNX Runtime pins its intra-op workers one per physical core and leaves one core for the
    thread that calls run(), which computes in every parallel operator too. It doesn't pin that thread. When the
    scheduler moves it onto a worker's core, the two share the core and every operator waits on them: a call then takes
    about 40% longer, for seconds at a time, depending on what else runs on the machine and not on the model. So the
    tool pins the calling thread, and only it, to the core the pool left free. Call it after the first call, so the
    pools that start then (ONNX Runtime's, and torch's for pre- and post-processing) keep their own placement.
    Returns the calling thread's CPUs, or None when the session pinned nothing (a session made with explicit threads).
    """
    import os
    import threading

    from . import hostmem

    me = threading.get_native_id()
    allowed = os.sched_getaffinity(0)
    pinned = []
    for t in Path("/proc/self/task").iterdir():
        tid = int(t.name)
        if tid == me:
            continue
        try:
            cpus = os.sched_getaffinity(tid)
        except OSError:  # the thread ended
            continue
        if cpus < allowed:
            pinned.append(cpus)
    core = free_core(pinned, hostmem.physical_cores(), allowed)
    if core is None:
        return None
    os.sched_setaffinity(0, core)  # pid 0 is the calling thread alone
    return sorted(core)


def run_prepare(spec: dict, log=None) -> dict:
    """prepare() in its own process, in the model's environment, with the GPU hidden. Its lines go to `log` too."""
    from .cpurun import worker_env

    with tempfile.TemporaryDirectory(prefix="mtr-onnx-") as tmp:
        spec_path, result_path = Path(tmp) / "spec.json", Path(tmp) / "result.json"
        spec_path.write_text(json.dumps({**spec, "result": str(result_path)}))
        proc = subprocess.run([sys.executable, "-m", "mtr_measure.ort", str(spec_path)], env=worker_env())
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if not result:
        raise MeasureError("ADAPTER_RUNTIME", f"The ONNX export exited with code {proc.returncode} without a result")
    if not result.get("ok"):
        if result.get("traceback"):
            print(result["traceback"], file=sys.stderr, flush=True)
        raise PrepareFailed(result.get("code", "ADAPTER_RUNTIME"), result.get("error", "the ONNX export failed"), result.get("weights") or [])
    for line in result.get("lines", []) if log else []:
        log(line, echo=False)
    return result["info"]


def verify(check: dict) -> str:
    """The ONNX file the check ran, still the same bytes. Returns its path."""
    from .record import sha256_file

    info = check.get("runtime_info") or {}
    path = Path(check.get("onnx") or "")
    want = (info.get("export") or {}).get("onnx_sha256")
    if not want or not path.is_file():
        raise MeasureError("NOT_CHECKED", f"The ONNX file {path} is gone. Run `mtr check --runtime onnxruntime-cpu` again")
    if sha256_file(path) != want:
        raise MeasureError("NOT_CHECKED", f"The ONNX file {path} changed since it was checked. Run `mtr check --runtime onnxruntime-cpu` again")
    return str(path)


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    lines: list[str] = []

    def say(line: str) -> None:
        print(line, flush=True)
        lines.append(line)

    try:
        result = {"ok": True, "info": prepare(spec, say), "lines": lines}
    except MeasureError as e:
        result = {"ok": False, "code": e.code, "error": str(e)}
    except Exception as e:  # the export raised; keep the traceback for the log
        result = {"ok": False, "code": "ADAPTER_RUNTIME", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    if not result["ok"] and "weights" in _state:
        result["weights"] = _state["weights"]
    Path(spec["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
