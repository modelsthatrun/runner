"""The per-model adapter: one `adapter.py` per model directory (models/_template/adapter.py).

Measuring knows nothing about tasks. An adapter tells it how to load a checkpoint, how to turn
the fixed input file into what the model takes, and how to run one end-to-end call. The output
step (outputs.py) uses the same adapter, plus to_prediction() to read predict()'s output.

Required, at module level (kind "image", the default):
    TASK: str                     the task it solves, e.g. "object-detection". It picks the timing
                                  image (timing/timing.toml) and the standard type to_prediction() returns
    SIZES: list[str]              checkpoint sizes, e.g. ["nano", "small"], as the catalog names them
    PRECISIONS: list[str]         what the official code supports: "fp16", "fp32", "bf16"
    PACKAGE: str                  installed distribution whose version is recorded
    INPUT: dict[str, str]         official default input resolution per size, "WxH"
    load(size, precision)         returns the loaded model, on the GPU
    predict(model, x)             one call from input to output, as a user's code would make it
    weight_files(size)            files that load() read, main checkpoint first

Optional:
    prepare_input(paths)          default: the single input file as an RGB PIL image
    input_size(size, x)           use instead of INPUT when the resolution depends on the input
    package_version()             use when PACKAGE is not an installed distribution, e.g. a git checkout
    LATENCY_SCOPE: str            what one predict() call covers, if it differs from the default
    to_prediction(model, x, out)  predict()'s output as the task's standard type (predictions.py),
                                  for the output step only. Not needed if predict() returns one
    EVAL: dict                    settings for `mtr eval` (eval.py), e.g. the official
                                  evaluation's lower score threshold
    eval_predict(model, x)        use instead of predict() for accuracy, when the official
                                  evaluation calls the model differently from its README

Language models (KIND = "llm") have no code: the pinned llama.cpp build runs them
(models/_template-llm/adapter.py). Required:
    KIND = "llm"
    TASK: str                     the task it solves, e.g. "chat"
    SIZES: list[str]              one row per GGUF file, e.g. "8b-q4_k_m"
    WEIGHTS: dict[str, dict]      per size: repo, revision (a 40-hex commit), files (GGUF, main
                                  file first; split files in order) and quant (e.g. "Q4_K_M")
Optional:
    SERVER_ARGS: list[str]        llama-server flags the official model card requires, such as
                                  "--jinja". Flags the protocol owns (LLM_PROTOCOL_FLAGS) are refused
    THINKING_OFF_ARGS: list[str]  for a model with a thinking mode: the llama-server flags that turn
                                  it off the way its model card says, e.g. ["--chat-template-kwargs",
                                  '{"enable_thinking": false}']. Used by `mtr eval` only
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from .errors import MeasureError

KNOWN_PRECISIONS = ("fp16", "fp32", "bf16")
DEFAULT_SCOPE = "predict(): pre-processing, forward pass and post-processing; input file decoded beforehand"


KINDS = ("image", "llm")
# llama-server flags the protocol sets for every language model; an adapter may not set them.
LLM_PROTOCOL_FLAGS = {
    "-c", "--ctx-size", "-np", "--parallel", "-ngl", "--gpu-layers", "--n-gpu-layers", "--host", "--port",
    "-b", "--batch-size", "-ub", "--ubatch-size", "-fa", "--flash-attn", "-ctk", "--cache-type-k",
    "-ctv", "--cache-type-v", "-t", "--threads", "-fit", "--fit", "-m", "--model", "-hf", "--hf-repo",
    "--no-mmap", "--mlock", "-lv", "--verbosity", "--log-verbosity", "--reasoning-budget", "-sm", "--split-mode", "-ts", "--tensor-split", "--n-cpu-moe", "-cmoe", "--cpu-moe",
}


class AdapterError(MeasureError):
    def __init__(self, message: str, code: str = "ADAPTER_INTERFACE"):
        super().__init__(code, message)


def weights_dir(model_id: str) -> Path:
    """The shared weights cache for one model, for packages that download into the working directory.

    Set MTR_WEIGHTS to move the cache. torch hub and Hugging Face already share their own caches.
    """
    root = Path(os.environ.get("MTR_WEIGHTS", "~/.cache/modelsthatrun/weights")).expanduser()
    d = root / model_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_adapter(path: Path) -> ModuleType:
    if not path.is_file():
        raise AdapterError(f"No adapter at {path}. Start from models/_template/adapter.py.")
    # Let the adapter import helper modules that sit next to it.
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("mtr_adapter", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    check_interface(module)
    return module


def kind(a: Any) -> str:
    k = getattr(a, "KIND", "image")
    if k not in KINDS:
        raise AdapterError(f"KIND must be one of {KINDS}, got {k!r}")
    return k


def kind_of_file(path: Path) -> str:
    """The adapter's kind without importing it: the parent process never imports a model's code."""
    found = re.search(r"""^KIND\s*=\s*["']([a-z]+)["']""", path.read_text(), re.M)
    return found.group(1) if found else "image"


def task_of_file(path: Path) -> str | None:
    """The adapter's TASK without importing it, like kind_of_file()."""
    found = re.search(r"""^TASK\s*=\s*["']([a-z0-9-]+)["']""", path.read_text(), re.M)
    return found.group(1) if found else None


def sizes_of_file(path: Path) -> list[str] | None:
    """The adapter's SIZES without importing it, when they are a literal list or list(<a dict literal>)."""
    import ast

    tree = ast.parse(path.read_text())
    assigned = {t.id: node.value for node in tree.body if isinstance(node, ast.Assign) for t in node.targets if isinstance(t, ast.Name)}
    value = assigned.get("SIZES")
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "list" \
            and len(value.args) == 1 and isinstance(value.args[0], ast.Name):
        value = assigned.get(value.args[0].id)
        keys = value.keys if isinstance(value, ast.Dict) else None
        return [k.value for k in keys] if keys and all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys) else None
    if isinstance(value, (ast.List, ast.Tuple)) and all(isinstance(e, ast.Constant) and isinstance(e.value, str) for e in value.elts):
        return [e.value for e in value.elts]
    return None


def check_llm_interface(a: Any) -> None:
    missing = [n for n in ("SIZES", "WEIGHTS") if not hasattr(a, n)]
    if missing:
        raise AdapterError(f"The adapter is missing {', '.join(missing)}")
    for size in a.SIZES:
        w = a.WEIGHTS.get(size)
        if not isinstance(w, dict):
            raise AdapterError(f'WEIGHTS has no entry for size "{size}"')
        lacking = [k for k in ("repo", "revision", "files", "quant") if not w.get(k)]
        if lacking:
            raise AdapterError(f'WEIGHTS["{size}"] is missing {", ".join(lacking)}')
        if not re.fullmatch(r"[0-9a-f]{40}", str(w["revision"])):
            raise AdapterError(f'WEIGHTS["{size}"]["revision"] must be the repository\'s 40-hex commit, not a branch. '
                               f"Get it from https://huggingface.co/api/models/{w['repo']} (\"sha\")")
        if not isinstance(w["files"], list) or not all(str(f).endswith(".gguf") for f in w["files"]):
            raise AdapterError(f'WEIGHTS["{size}"]["files"] must be a list of .gguf files, main file first')
    for name in ("SERVER_ARGS", "THINKING_OFF_ARGS"):
        args = getattr(a, name, [])
        if not isinstance(args, list) or not all(isinstance(x, str) for x in args):
            raise AdapterError(f"{name} must be a list of strings")
        owned = [x for x in args if x.split("=", 1)[0] in LLM_PROTOCOL_FLAGS]
        if owned:
            raise AdapterError(f"{name} may not set {', '.join(owned)}: the protocol sets those for every model")


def check_interface(a: Any) -> None:
    if kind(a) == "llm":
        check_llm_interface(a)
        return
    missing = [n for n in ("SIZES", "PRECISIONS", "load", "predict", "weight_files") if not hasattr(a, n)]
    if not hasattr(a, "package_version") and not hasattr(a, "PACKAGE"):
        missing.append("PACKAGE (or package_version())")
    if not hasattr(a, "input_size") and not hasattr(a, "INPUT"):
        missing.append("INPUT (or input_size())")
    if missing:
        raise AdapterError(f"The adapter is missing {', '.join(missing)}")
    unknown = [p for p in a.PRECISIONS if p not in KNOWN_PRECISIONS]
    if unknown or not a.PRECISIONS:
        raise AdapterError(f"PRECISIONS must list some of {KNOWN_PRECISIONS}, got {a.PRECISIONS}")


def choose_precision(supported: list[str]) -> str:
    """fp16 if the official code supports it, otherwise fp32 (docs/protocol.md, "Scope")."""
    if "fp16" in supported:
        return "fp16"
    if "fp32" in supported:
        return "fp32"
    raise AdapterError(f"The protocol measures fp16 or fp32, and the adapter supports only {supported}")


def check_size(a: Any, size: str) -> None:
    if size not in a.SIZES:
        raise AdapterError(f'Size "{size}" is not in the adapter\'s SIZES {a.SIZES}', "USAGE")


def prepare_input(a: Any, paths: list[Path]) -> Any:
    if hasattr(a, "prepare_input"):
        return a.prepare_input(paths)
    if len(paths) != 1:
        raise AdapterError("This adapter takes one input file. Define prepare_input(paths) for more.")
    from PIL import Image

    with Image.open(paths[0]) as im:
        return im.convert("RGB")


def input_size(a: Any, size: str, x: Any) -> str:
    value = a.input_size(size, x) if hasattr(a, "input_size") else a.INPUT.get(size)
    if not isinstance(value, str) or not re.fullmatch(r"\d+x\d+", value):
        raise AdapterError(f'The input resolution for "{size}" must be "WxH", got {value!r}')
    return value


def package(a: Any) -> str:
    if hasattr(a, "package_version"):
        return str(a.package_version())
    return f"{a.PACKAGE}=={importlib.metadata.version(a.PACKAGE)}"


def latency_scope(a: Any) -> str:
    return str(getattr(a, "LATENCY_SCOPE", DEFAULT_SCOPE))


def to_prediction(a: Any, model: Any, x: Any, out: Any, task: str) -> Any:
    """predict()'s output as the standard prediction type of the task (predictions.py)."""
    from . import predictions

    expected = predictions.task_type(task)
    if hasattr(a, "to_prediction"):
        try:
            pred = a.to_prediction(model, x, out)
        except NotImplementedError:
            raise AdapterError("The adapter's to_prediction() isn't written yet (models/_template/adapter.py)") from None
    elif isinstance(out, predictions.TYPES):
        pred = out
    else:
        raise AdapterError(
            f"predict() returns {type(out).__name__}. Add to_prediction(model, x, out) to the adapter, returning "
            f"mtr_measure.predictions.{expected.__name__} (models/_template/adapter.py)"
        )
    if not isinstance(pred, expected):
        raise AdapterError(f'"{task}" outputs are {expected.__name__}, but to_prediction() returned {type(pred).__name__}')
    return pred
