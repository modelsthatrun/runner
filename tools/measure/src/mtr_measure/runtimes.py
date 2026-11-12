"""Runtimes: how a checkpoint's call executes (docs/protocol.md, "Runtimes").

    pytorch-cuda     the official PyTorch code on the GPU (llama.cpp on the GPU for language models). The default:
                     records name no runtime
    tensorrt         an engine built from the authors' own ONNX export (trt.py). Image models
    pytorch-cpu      the official PyTorch or diffusers code with the GPU hidden. Image and text-to-image models
    onnxruntime-cpu  the authors' ONNX export, run by their own ONNX Runtime code on the CPU (ort.py). Image models
    llamacpp-cpu     the pinned llama.cpp build with the GPU hidden and no layer on it. Language models

Every per-size file of a runtime other than the default carries it as an infix: runs/<size>.<runtime>.json,
.check/<size>.<runtime>.json, evals/<size>/<benchmark>.<runtime>.json. Sizes may hold dots (1.6b, 0.6b-q8_0), so
a file name is read by peeling known runtimes off its end (split_name), never by counting dots.
"""

from __future__ import annotations

from .errors import MeasureError

DEFAULT = "pytorch-cuda"
GPU_RUNTIMES = ("pytorch-cuda", "tensorrt")
CPU_RUNTIMES = ("pytorch-cpu", "onnxruntime-cpu", "llamacpp-cpu")
RUNTIMES = GPU_RUNTIMES + CPU_RUNTIMES

# Which kinds of model each runtime takes.
KINDS = {
    "pytorch-cuda": ("image", "llm", "generation"),
    "tensorrt": ("image",),
    "pytorch-cpu": ("image", "generation"),
    "onnxruntime-cpu": ("image",),
    "llamacpp-cpu": ("llm",),
}
# The optional dependency group of the model's uv project that a runtime needs (main.py).
UV_GROUPS = {"tensorrt": "tensorrt", "onnxruntime-cpu": "onnxruntime"}
# What a runtime is called in the log and on pages.
LABELS = {"pytorch-cuda": "PyTorch", "tensorrt": "TensorRT", "pytorch-cpu": "PyTorch on the CPU",
          "onnxruntime-cpu": "ONNX Runtime on the CPU", "llamacpp-cpu": "llama.cpp on the CPU"}


def is_cpu(runtime: str | None) -> bool:
    return runtime in CPU_RUNTIMES


def suffix(runtime: str | None) -> str:
    """The infix every per-size file of a runtime carries: "" for the default, ".tensorrt", ".pytorch-cpu", …"""
    return "" if not runtime or runtime == DEFAULT else f".{runtime}"


def check(kind: str, runtime: str) -> str:
    """The runtime, if it takes this kind of model."""
    if runtime not in RUNTIMES:
        raise MeasureError("USAGE", f"--runtime must be one of {', '.join(RUNTIMES)}, got {runtime!r}")
    if kind not in KINDS[runtime]:
        names = {"image": "image", "llm": "language", "generation": "text-to-image"}
        takes = " and ".join(names[k] for k in KINDS[runtime]) + " models" + (" only" if len(KINDS[runtime]) == 1 else "")
        cpu = {"image": "pytorch-cpu", "generation": "pytorch-cpu", "llm": "llamacpp-cpu"}[kind]
        raise MeasureError("USAGE", f"--runtime {runtime} is for {takes}. A {names[kind]} model's CPU runtime is {cpu}")
    return runtime


def split_name(name: str, ext: str = ".json") -> tuple[str, str, str] | None:
    """A per-size file name as (size, runtime, rest): "1.6b.pytorch-cpu.times.json" -> ("1.6b", "pytorch-cpu",
    "times"). `rest` is what sits between the runtime and `ext` ("" for a record). None if it doesn't end in `ext`."""
    if not name.endswith(ext):
        return None
    stem = name[: -len(ext)]
    rest = ""
    for tail in ("times", "failed", "dry-run", "reference", "prediction", "server"):
        if stem.endswith(f".{tail}"):
            stem, rest = stem[: -len(tail) - 1], tail
            break
    for runtime in RUNTIMES:
        if runtime != DEFAULT and stem.endswith(f".{runtime}"):
            return stem[: -len(runtime) - 1], runtime, rest
    return stem, DEFAULT, rest


def record_files(names: list[str]) -> list[tuple[str, str, str]]:
    """The run records among file names in runs/: (file, size, runtime) for each <size>[.<runtime>].json."""
    out = []
    for n in sorted(names):
        parts = split_name(n)
        if parts and not parts[2]:
            out.append((n, parts[0], parts[1]))
    return out
