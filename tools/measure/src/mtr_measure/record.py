"""Statistics, weight hashes and the run record itself (schema/run-record.schema.json).

A record combines the independent runs of one measurement: its median and p90 are the medians
of the runs' values, and its VRAM peak is the highest run's.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from functools import cache
from pathlib import Path

import jsonschema

from .gpu import MIB
from .protocol import PROTOCOL_VERSION


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks, the same as numpy's default."""
    if not values:
        raise ValueError("no values")
    s = sorted(values)
    pos = (len(s) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_entries(paths: list[Path]) -> list[dict]:
    """sha256 and size of each weight file, in the adapter's order (main checkpoint first)."""
    if not paths:
        raise ValueError("weight_files() returned no files")
    out = []
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(f"weight_files() lists {p}, which doesn't exist")
        out.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return out


def summarize_run(worker: dict, baseline_bytes: int, peak_bytes: int) -> dict:
    """One run's numbers: latency median and p90 of its timed calls, and its peak over its own baseline."""
    times = worker["times_ms"]
    return {
        "median": round(statistics.median(times), 3),
        "p90": round(percentile(times, 90), 3),
        "vram_peak_mib": max(0, round((peak_bytes - baseline_bytes) / MIB)),
    }


def spread(runs: list[dict], key: str = "median") -> float:
    """(max - min) / median of one number across the runs, the number MAX_SPREAD limits. Image models
    use the latency median, language models the decode rate."""
    vals = [r[key] for r in runs]
    return (max(vals) - min(vals)) / statistics.median(vals)


def summarize_llm_run(worker: dict, baseline_bytes: int, peak_bytes: int) -> dict:
    """One language-model run: medians over its timed requests, and its peak over its own baseline."""
    reqs = worker["requests"]

    def med(k: str) -> float:
        return statistics.median(r[k] for r in reqs)

    return {
        "ttft_ms": round(med("ttft_ms"), 2),
        "ttft_p90_ms": round(percentile([r["ttft_ms"] for r in reqs], 90), 2),
        "e2e_ms": round(med("e2e_ms"), 1),
        "e2e_p90_ms": round(percentile([r["e2e_ms"] for r in reqs], 90), 1),
        "prefill_tps": round(med("prefill_tps"), 1),
        "decode_tps": round(med("decode_tps"), 2),
        "prompt_tokens": int(med("prompt_n")),
        "vram_peak_mib": max(0, round((peak_bytes - baseline_bytes) / MIB)),
    }


def build_llm(
    *,
    model: str,
    size: str,
    weights: list[dict],
    runs: list[dict],
    quant: str,
    engine: dict,
    server_args: list[str],
    gen_tokens: int,
    context: int,
    gpu_name: str,
    driver: str,
    warmup: int,
    measured: str,
    machine: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """A language-model run record (schema/llm-run-record.schema.json)."""
    s = [summarize_llm_run(r["worker"], r["baseline_bytes"], r["peak_bytes"]) for r in runs]
    first = runs[0]["worker"]

    def med(k: str, nd: int) -> float:
        return round(statistics.median(x[k] for x in s), nd)

    return {
        "kind": "llm",
        "model": model,
        "size": size,
        "weights": weights,
        "measurement": {
            "engine": engine,
            "quant": quant,
            "batch": 1,
            "context": context,
            "prompt_tokens": s[0]["prompt_tokens"],
            "prompt_sha256": sha256_file(Path(__file__).with_name("llm_prompt.txt")),
            "gen_tokens": gen_tokens,
            "ttft_ms": {"median": med("ttft_ms", 2), "p90": med("ttft_p90_ms", 2)},
            "e2e_ms": {"median": med("e2e_ms", 1), "p90": med("e2e_p90_ms", 1)},
            "prefill_tps": med("prefill_tps", 1),
            "decode_tps": med("decode_tps", 2),
            "vram_peak_mib": max(x["vram_peak_mib"] for x in s),
            "gpu_layers": first["gpu_layers"],
            "server_args": server_args,
            "iterations": {"warmup": warmup, "timed": len(first["requests"])},
            "env": {
                "gpu": gpu_name,
                "driver": driver,
                "python": first["python"],
                "baseline_mib": round(statistics.median(r["baseline_bytes"] for r in runs) / MIB),
                **(machine or {}),
            },
            "measured": measured,
            "protocol": PROTOCOL_VERSION,
            "runs": [{k: x[k] for k in ("ttft_ms", "prefill_tps", "decode_tps", "vram_peak_mib")} for x in s],
        },
        **({"provenance": provenance} if provenance else {}),
    }


def build(
    *,
    model: str,
    size: str,
    weights: list[dict],
    runs: list[dict],
    gpu_name: str,
    driver: str,
    image_digest: str | None,
    warmup: int,
    measured: str,
    machine: dict | None = None,
    provenance: dict | None = None,
) -> dict:
    """`runs` holds one {"worker", "baseline_bytes", "peak_bytes"} per independent run. `machine` adds the
    machine facts to env (machine.facts), and `provenance` says where the record came from."""
    if not runs:
        raise ValueError("no runs")
    summaries = [summarize_run(r["worker"], r["baseline_bytes"], r["peak_bytes"]) for r in runs]
    first = runs[0]["worker"]
    return {
        "kind": "image",
        "model": model,
        "size": size,
        "weights": weights,
        "measurement": {
            "precision": first["precision"],
            "batch": 1,
            "input": first["input"],
            "latency_ms": {
                "median": round(statistics.median(s["median"] for s in summaries), 3),
                "p90": round(statistics.median(s["p90"] for s in summaries), 3),
            },
            "latency_scope": first["latency_scope"],
            "vram_peak_mib": max(s["vram_peak_mib"] for s in summaries),
            "torch_max_reserved_mib": round(max(r["worker"]["torch_max_reserved_bytes"] for r in runs) / MIB),
            "iterations": {"warmup": warmup, "timed": len(first["times_ms"])},
            "env": {
                "gpu": gpu_name,
                "driver": driver,
                "cuda": first["env"]["cuda"],
                "torch": first["env"]["torch"],
                "python": first["env"]["python"],
                "package": first["env"]["package"],
                "image_digest": image_digest,
                "baseline_mib": round(statistics.median(r["baseline_bytes"] for r in runs) / MIB),
                **(machine or {}),
            },
            "measured": measured,
            "protocol": PROTOCOL_VERSION,
            "runs": summaries,
        },
        **({"provenance": provenance} if provenance else {}),
    }


LLM_TIME_KEYS = ("ttft_ms", "e2e_ms", "prefill_tps", "decode_tps", "prompt_n")


def times(runs: list[dict], llm: bool) -> dict:
    """The raw timings behind a record (schema/run-times.schema.json): every timed call of each run, or for a
    language model every timed request's numbers. They let anyone recompute the record's medians."""
    if llm:
        return {"kind": "llm", "runs": [[{k: q[k] for k in LLM_TIME_KEYS} for q in r["worker"]["requests"]] for r in runs]}
    return {"kind": "image", "runs": [list(r["worker"]["times_ms"]) for r in runs]}


def project_dir(model_dir: Path) -> Path:
    """The uv project a model runs in: its own pyproject.toml, or its parent's (tools/measure/examples/).
    A language model without one runs in the tool's own environment (main.py)."""
    for d in (model_dir, model_dir.parent):
        if (d / "pyproject.toml").is_file():
            return d
    from .adapter import kind_of_file

    if (model_dir / "adapter.py").is_file() and kind_of_file(model_dir / "adapter.py") == "llm":
        return Path(__file__).resolve().parents[2]
    return model_dir


def fingerprint(model_dir: Path) -> dict:
    """What a passed `check` is tied to, and what a record's provenance names. Editing the adapter or the lock invalidates a check."""
    lock = project_dir(model_dir) / "uv.lock"
    return {
        "adapter_sha256": sha256_file(model_dir / "adapter.py"),
        "lock_sha256": sha256_file(lock) if lock.is_file() else None,
    }


def check_path(model_dir: Path, size: str) -> Path:
    return model_dir / ".check" / f"{size}.json"


# ModelsThatRun's record schemas, generated from the schema package its maintainers keep; never edited here.
SCHEMA_DIR = Path(__file__).with_name("schema")


@cache
def _schema(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.schema.json").read_text())


def validate(record: dict, schema: str = "run-record") -> list[str]:
    """Problems against the schema the site uses (schema/<schema>.schema.json). Empty means valid."""
    v = jsonschema.Draft202012Validator(_schema(schema), format_checker=jsonschema.FormatChecker())
    return [f"{'.'.join(map(str, e.absolute_path)) or '(root)'}: {e.message}" for e in v.iter_errors(record)]
