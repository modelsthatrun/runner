"""TensorRT rows (docs/protocol.md, "Runtimes"): the authors' own ONNX export of the released weights, and an engine
built from it the way their `trtexec --onnx=… --saveEngine=… --fp16` command builds one.

    mtr check <model> --size <size> --runtime tensorrt [--rebuild]       export, build, and check the engine
    mtr eval <model> --size <size> --benchmark … --runtime tensorrt      score that engine
    mtr queue add <model> --size <size> --runtime tensorrt               refused until the eval above exists
    mtr run <model> --size <size> --runtime tensorrt                     time it: runs/<size>.tensorrt.json

An adapter with a TensorRT row declares (docs/adapters.md):

    TENSORRT: dict              {"precision": "fp16", "builder": {"fp16": True}, "status": "runs-as-is"}, plus
                                "patch": "fixes/<file>" when status is "runs-with-fixes". `builder` holds only the
                                flags the official command names. "input": "WxH" (or one per size) when the engine's
                                input, fixed at export, differs from the PyTorch row's
    export_onnx(size, out)      runs the official export on the released checkpoint, writing into the directory `out`.
                                Returns {"onnx": Path, "exporter": str, "args": [str, ...]}
  or
    export_engine(size, out)    for a package whose official TensorRT step is one call that exports and builds, such as
                                Ultralytics' export(format="engine"): runs it on the released checkpoint into `out`.
                                Returns what export_onnx() returns, plus {"engine": Path} and optionally
                                {"builder": {...}}, the settings the package passes TensorRT
    load_engine(size, engine)   loads the engine file with the official TensorRT inference code. predict(),
                                eval_predict() and to_prediction() take what it returns, as they take what load() returns

With export_onnx(), the tool builds the engine. The wheels hold no `trtexec`, so it uses TensorRT's Python API, set as trtexec sets it
(samples/common/sampleEngines.cpp in NVIDIA/TensorRT): the flags the command names, trtexec's 8 timing iterations per
tactic, and for an input whose ONNX shape is dynamic one optimization profile with every dynamic dimension at 1, which
is what trtexec does when no shapes are given. Everything else stays at TensorRT's defaults. With export_engine(), the
package builds it, with its own settings, and the record says so.

Engines and ONNX files stay in ~/.cache/modelsthatrun/trt/<model>/<size>/ (MTR_TRT moves it), never in the checkout,
and are never published. `check` keeps an engine and builds it again only with --rebuild, or when the weights, the
export, the builder settings, the TensorRT version or the GPU changed. TensorRT picks kernels by timing them, so a
rebuild gives other bytes: the eval of the old engine no longer counts, and the queue refuses until the new one is
scored. A new runtime is scored before it is timed, because a check can pass on outputs that are all NaN.

Run as `python -m mtr_measure.trt <spec.json>`, it exports and builds in its own process before the check's worker
starts, so the builder's memory never shows up in the check's numbers and the parent never holds a CUDA context.
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

RUNTIMES = ("pytorch-cuda", "tensorrt")
DEFAULT = "pytorch-cuda"
# What trtexec sets that the API leaves elsewhere: --avgTiming defaults to 8 there, and dynamic dimensions without
# --shapes become 1 (kDEFAULT_DIMENSION).
TRTEXEC_AVG_TIMING = 8
TRTEXEC_DEFAULT_DIMENSION = 1
BUILDER_FLAGS = {"fp16": "FP16", "bf16": "BF16", "int8": "INT8"}
STATUSES = ("runs-as-is", "runs-with-fixes")


def suffix(runtime: str) -> str:
    """The infix every per-size file of a runtime row carries: runs/<size>.tensorrt.json, evals/<size>/<b>.tensorrt.json."""
    return "" if runtime == DEFAULT else f".{runtime}"


def check_runtime(runtime: str) -> str:
    if runtime not in RUNTIMES:
        raise MeasureError("USAGE", f"--runtime must be one of {', '.join(RUNTIMES)}, got {runtime!r}")
    return runtime


def root() -> Path:
    return Path(os.environ.get("MTR_TRT", "~/.cache/modelsthatrun/trt")).expanduser()


def model_root(model_id: str) -> Path:
    return root() / model_id


def engine_dir(model_id: str, size: str) -> Path:
    return model_root(model_id) / size


def info_path(model_id: str, size: str) -> Path:
    return engine_dir(model_id, size) / "engine.json"


def adapter_spec(a: Any) -> dict:
    """The adapter's TENSORRT, checked. Raises when the adapter has no TensorRT row."""
    spec = getattr(a, "TENSORRT", None)
    if spec is None:
        raise MeasureError("USAGE", "This model has no TensorRT row: its adapter declares no TENSORRT (docs/adapters.md)")
    exports = [n for n in ("export_onnx", "export_engine") if callable(getattr(a, n, None))]
    missing = [n for n in ("load_engine",) if not callable(getattr(a, n, None))] + ([] if exports else ["export_onnx() or export_engine()"])
    if missing or not isinstance(spec, dict):
        raise MeasureError("ADAPTER_INTERFACE", f"An adapter with TENSORRT needs load_engine() and export_onnx() or export_engine(); missing {', '.join(missing) or 'a dict'}")
    if len(exports) == 2:
        raise MeasureError("ADAPTER_INTERFACE", "An adapter with TENSORRT defines export_onnx() or export_engine(), not both")
    if spec.get("precision") not in ("fp16", "bf16", "fp32"):
        raise MeasureError("ADAPTER_INTERFACE", f"TENSORRT['precision'] must be fp16, bf16 or fp32, got {spec.get('precision')!r}")
    builder = spec.get("builder", {})
    unknown = [k for k in builder if k not in BUILDER_FLAGS]
    if not isinstance(builder, dict) or unknown:
        raise MeasureError("ADAPTER_INTERFACE", f"TENSORRT['builder'] takes only the flags {', '.join(BUILDER_FLAGS)}, not {', '.join(unknown)}")
    status = spec.get("status", "runs-as-is")
    if status not in STATUSES:
        raise MeasureError("ADAPTER_INTERFACE", f"TENSORRT['status'] must be one of {STATUSES}, got {status!r}")
    if status == "runs-with-fixes" and not spec.get("patch"):
        raise MeasureError("ADAPTER_INTERFACE", "TENSORRT['status'] is runs-with-fixes, so it needs 'patch': the file under fixes/")
    return {"precision": spec["precision"], "builder": dict(builder), "status": status, "patch": spec.get("patch")}


def builder_settings(spec: dict, package: str | None = None) -> dict:
    """What the record says about the build: the official command's flags, and how trtexec fills in the rest. When the
    package builds the engine itself (export_engine()), the flags are the ones its call sets, and the rest is its own."""
    flags = [BUILDER_FLAGS[k] for k, on in sorted(spec["builder"].items()) if on]
    if package:
        return {"api": f"{package}, which builds with TensorRT's Python API", "flags": flags, "other": "the package's own settings"}
    return {
        "api": "TensorRT Python API, set as trtexec sets it",
        "flags": flags,
        "avg_timing_iterations": TRTEXEC_AVG_TIMING,
        "dynamic_dimensions": TRTEXEC_DEFAULT_DIMENSION,
        "other": "TensorRT defaults",
    }


def trt_version() -> str:
    import tensorrt

    return tensorrt.__version__


def sha256(path: Path) -> str:
    from .record import sha256_file

    return sha256_file(path)


def onnx_opset(path: Path) -> int | None:
    """The default-domain opset the file declares. Exporters may write another than they were asked for."""
    try:
        import onnx
    except ImportError:
        return None
    m = onnx.load(str(path), load_external_data=False)
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx"):
            return int(o.version)
    return None


def self_contained(path: Path) -> None:
    """The engine is built from this one file, and its hash must cover every weight, so nothing may sit outside it."""
    try:
        import onnx
        from onnx.external_data_helper import uses_external_data
    except ImportError:
        return
    m = onnx.load(str(path), load_external_data=False)
    if any(uses_external_data(t) for t in m.graph.initializer):
        raise MeasureError("ADAPTER_RUNTIME", f"{path.name} keeps weights in an external data file, so its sha256 doesn't cover them. "
                                              "Export a self-contained ONNX file")


# --- building ------------------------------------------------------------------------------------


def build_engine(onnx_path: Path, engine_path: Path, spec: dict, say=print) -> dict:
    """Parse the ONNX file and build a serialized engine, as `trtexec --onnx --saveEngine` with the spec's flags does."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise MeasureError("ENGINE", f"TensorRT couldn't parse {onnx_path.name}: {errors}")
    config = builder.create_builder_config()
    for key, on in spec["builder"].items():
        if on:
            config.set_flag(getattr(trt.BuilderFlag, BUILDER_FLAGS[key]))
    config.avg_timing_iterations = TRTEXEC_AVG_TIMING
    shapes: dict[str, list[int]] = {}
    profile = None
    for i in range(network.num_inputs):
        t = network.get_input(i)
        dims = list(t.shape)
        if any(d < 0 for d in dims):
            if t.is_shape_tensor:
                raise MeasureError("ENGINE", f"Input {t.name} is a dynamic shape tensor, which trtexec's defaults don't cover")
            fixed = [d if d > 0 else TRTEXEC_DEFAULT_DIMENSION for d in dims]
            profile = profile or builder.create_optimization_profile()
            profile.set_shape(t.name, fixed, fixed, fixed)
            shapes[t.name] = fixed
            say(f"input {t.name}: dynamic {dims}, built for {fixed} as trtexec does without --shapes")
        else:
            shapes[t.name] = dims
    if profile is not None:
        config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise MeasureError("ENGINE", f"TensorRT couldn't build an engine from {onnx_path.name}. Read the log above")
    engine_path.write_bytes(bytes(plan))
    return {"build_s": round(build_s, 1), "shapes": shapes, "optimization_level": int(config.builder_optimization_level)}


def prepare(spec: dict, say=print) -> dict:
    """Export and build in this process (see the module doc), or reuse the kept engine. Returns engine.json's content."""
    from . import adapter as ad

    a = ad.load_adapter(Path(spec["adapter"]))
    size = spec["size"]
    ad.check_size(a, size)
    rt = adapter_spec(a)
    version = trt_version()
    out = engine_dir(spec["model"], size)
    weights = [{"file": Path(p).name, "sha256": sha256(Path(p))} for p in a.weight_files(size) if Path(p).is_file()]
    package = ad.package(a) if callable(getattr(a, "export_engine", None)) else None
    key = {"tensorrt": version, "builder": builder_settings(rt, package), "weights": weights, "gpu": spec["gpu"],
           "precision": rt["precision"], "status": rt["status"], "patch": rt["patch"]}
    path = info_path(spec["model"], size)
    if path.is_file() and not spec.get("rebuild"):
        info = json.loads(path.read_text())
        engine = Path(info["engine"]["path"])
        if info.get("key") == key and engine.is_file() and sha256(engine) == info["engine"]["sha256"]:
            say(f"engine: kept {engine} (sha256 {info['engine']['sha256'][:12]}…, built {info['engine']['built']}). "
                "--rebuild builds it again, which needs a new eval")
            return {**info, "reused": True}
        say("engine: the kept one doesn't match the weights, the export, the builder settings, TensorRT or the GPU; building again")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    call = "export_engine" if package else "export_onnx"
    say(f"export: {call}({size!r}) into {out}" + (f", {package}'s own export and TensorRT build; this takes minutes" if package else ""))
    t0 = time.perf_counter()
    exported = getattr(a, call)(size, out)
    export_s = time.perf_counter() - t0
    onnx_path = Path(exported["onnx"])
    if not onnx_path.is_file():
        raise MeasureError("ADAPTER_RUNTIME", f"{call}() returned {onnx_path}, which doesn't exist")
    self_contained(onnx_path)
    opset = onnx_opset(onnx_path)
    say(f"export: {onnx_path.name}, {onnx_path.stat().st_size / 2**20:.1f} MiB, opset {opset}" + ("" if package else f", {export_s:.0f} s"))

    if package:
        engine_path = Path(exported["engine"])
        if not engine_path.is_file():
            raise MeasureError("ADAPTER_RUNTIME", f"export_engine() returned {engine_path}, which doesn't exist")
        settings = {**builder_settings(rt, package), **dict(exported.get("builder") or {})}
        built = {"build_s": round(export_s, 1)}
        say(f"build: {engine_path.name}, {engine_path.stat().st_size / 2**20:.1f} MiB, {export_s:.0f} s with the export")
    else:
        engine_path = out / "model.engine"
        say(f"build: TensorRT {version}, flags {', '.join(builder_settings(rt)['flags']) or 'none'}; this takes minutes")
        built = build_engine(onnx_path, engine_path, rt, say)
        settings = {**builder_settings(rt), "shapes": built["shapes"], "optimization_level": built["optimization_level"]}
        say(f"build: {engine_path.name}, {engine_path.stat().st_size / 2**20:.1f} MiB, {built['build_s']:.0f} s")
    info = {
        "key": key,
        "runtime": {
            "name": "tensorrt",
            "version": version,
            "status": rt["status"],
            "patch": rt["patch"],
            "export": {
                "exporter": str(exported["exporter"]),
                "args": [str(x) for x in exported.get("args", [])],
                "opset": opset,
                "onnx_sha256": sha256(onnx_path),
                "onnx_bytes": onnx_path.stat().st_size,
            },
            "engine": {
                "sha256": sha256(engine_path),
                "bytes": engine_path.stat().st_size,
                "builder": settings,
                "built": datetime.now().astimezone().isoformat(timespec="seconds"),
                "build_s": built["build_s"],
            },
        },
        "precision": rt["precision"],
        "engine": {"path": str(engine_path), "sha256": None},
        "onnx": str(onnx_path),
    }
    info["engine"]["sha256"] = info["runtime"]["engine"]["sha256"]
    info["engine"]["built"] = info["runtime"]["engine"]["built"]
    path.write_text(json.dumps(info, indent=2) + "\n")
    return {**info, "reused": False}


def run_prepare(spec: dict, gpu_index: int, log=None) -> dict:
    """prepare() in its own process, in the model's environment, as the parent's child. Its lines go to `log` too."""
    with tempfile.TemporaryDirectory(prefix="mtr-trt-") as tmp:
        spec_path, result_path = Path(tmp) / "spec.json", Path(tmp) / "result.json"
        spec_path.write_text(json.dumps({**spec, "result": str(result_path)}))
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu_index)}
        proc = subprocess.run([sys.executable, "-m", "mtr_measure.trt", str(spec_path)], env=env)
        result = json.loads(result_path.read_text()) if result_path.is_file() else None
    if not result:
        raise MeasureError("ENGINE", f"The TensorRT build exited with code {proc.returncode} without a result")
    if not result.get("ok"):
        if result.get("traceback"):
            print(result["traceback"], file=sys.stderr, flush=True)
        raise MeasureError(result.get("code", "ENGINE"), result.get("error", "the TensorRT build failed"))
    for line in result.get("lines", []) if log else []:
        log(line, echo=False)
    return result["info"]


# --- what later commands read ---------------------------------------------------------------------


def runtime_block(check: dict) -> dict:
    """measurement.runtime of a run record, from the check that built the engine."""
    rt = dict(check["runtime_info"])
    if not rt.get("patch"):
        rt.pop("patch", None)
    return rt


def eval_runtime(check: dict) -> dict:
    """An eval record's runtime: the engine it scored."""
    rt = check["runtime_info"]
    return {"name": rt["name"], "version": rt["version"], "engine_sha256": rt["engine"]["sha256"]}


def verify_engine(check: dict) -> Path:
    """The engine the check built, still the same bytes. A rebuild since then needs a new check, and a new eval."""
    path = Path(check["engine"])
    want = check["runtime_info"]["engine"]["sha256"]
    if not path.is_file():
        raise MeasureError("NOT_CHECKED", f"The engine {path} is gone. Run `mtr check --runtime tensorrt` again")
    if sha256(path) != want:
        raise MeasureError("NOT_CHECKED", f"The engine {path} changed since it was checked. Run `mtr check --runtime tensorrt` again")
    return path


def eval_of_engine(model_dir: Path, size: str, engine_sha: str) -> list[Path]:
    """Eval records of this size that scored this engine."""
    d = model_dir / "evals" / size
    found = []
    for f in sorted(d.glob("*.tensorrt.json")) if d.is_dir() else []:
        if f.name.count(".") != 2:
            continue  # <benchmark>.tensorrt.predictions.json and the like
        try:
            rec = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        if (rec.get("runtime") or {}).get("engine_sha256") == engine_sha:
            found.append(f)
    return found


def require_eval(model_dir: Path, size: str, check: dict) -> Path:
    """A runtime row is scored before it is timed (docs/protocol.md, "Runtimes")."""
    sha = check["runtime_info"]["engine"]["sha256"]
    found = eval_of_engine(model_dir, size, sha)
    if not found:
        raise MeasureError("NOT_SCORED", f"No eval record of {model_dir.name} / {size}'s TensorRT engine (sha256 {sha[:12]}…). Score it first: "
                                         f"`mtr eval {model_dir.name} --size {size} --benchmark <task benchmark> --runtime tensorrt`")
    return found[0]


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
    except Exception as e:  # the export or the build raised; keep the traceback for the log
        result = {"ok": False, "code": "ENGINE", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    Path(spec["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
