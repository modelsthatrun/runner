"""The parent side of `check` and `run` for language models (KIND = "llm").

The adapter of a language model is constants only, so the parent may import it. `check` builds the
engine and downloads the GGUF files if needed; `run` never downloads, so nothing slow happens while
a timed run holds the machine. The worker is llm.py.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from . import adapter as ad
from . import engine, hf, protocol
from .errors import MeasureError


def load(model_dir: Path, size: str) -> ModuleType:
    a = ad.load_adapter(model_dir / "adapter.py")
    ad.check_size(a, size)
    return a


def weight_paths(a: ModuleType, model_id: str, size: str, download: bool, say=print) -> list[Path]:
    w = a.WEIGHTS[size]
    if download:
        return [hf.ensure(model_id, w["repo"], w["revision"], f, say) for f in w["files"]]
    paths = [hf.local_path(model_id, w["repo"], w["revision"], f) for f in w["files"]]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise MeasureError("WEIGHTS_MISSING", f"Not downloaded: {', '.join(missing)}. Run `mtr check` first")
    return paths


def server(download: bool, say=print, gpu: int = 0) -> Path:
    if download:
        return engine.ensure(say, gpu)
    if not engine.server_path(gpu).is_file():
        raise MeasureError("ENGINE", "llama.cpp isn't built for this GPU yet. Run `mtr check` first, which builds it")
    return engine.server_path(gpu)


def server_args(a: ModuleType) -> list[str]:
    return [*protocol.LLM_SERVER_ARGS, *getattr(a, "SERVER_ARGS", [])]


def spec(a: ModuleType, model_id: str, size: str, run_dir: Path, check: bool, say=print, gpu: int = 0) -> tuple[dict, list[Path], Path]:
    """The worker spec for one run, the weight files and the server binary."""
    paths = weight_paths(a, model_id, size, download=check, say=say)
    binary = server(download=check, say=say, gpu=gpu)
    run_dir.mkdir(parents=True, exist_ok=True)
    s = {
        "server": str(binary),
        "gguf": str(paths[0]),
        "more_files": [str(p) for p in paths[1:]],
        "server_args": server_args(a),
        "prompt_tokens": protocol.LLM_PROMPT_TOKENS,
        "gen_tokens": protocol.LLM_GEN_TOKENS,
        "warmup": protocol.LLM_CHECK_WARMUP if check else protocol.LLM_WARMUP,
        "timed": protocol.LLM_CHECK_TIMED if check else protocol.LLM_TIMED,
        "server_log": str(run_dir / f"{size}.server.log"),
    }
    return s, paths, binary
