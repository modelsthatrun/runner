"""mtr eval for language models (KIND = "llm"): lm-evaluation-harness against llama-server.

Usage:
    mtr eval <id> --size <size> --benchmark ifeval

It starts the pinned llama-server (engine.py) with the evaluation settings in protocol.py, points
lm-evaluation-harness (tools/measure/lmeval/, pinned) at its chat endpoint, and writes
models/<id>/evals/<size>/<benchmark>.json. The model's own chat template (inside the GGUF) formats
every prompt, and decoding is greedy unless the task's own configuration says otherwise.

The server opens LLM_EVAL_SLOTS slots for throughput. If the model and their context don't fit
next to the other programs on the GPU, or leave less than EVAL_VRAM_MARGIN_MIB free for them once loaded (llama-server
allocates its cache at load), it tries half as many, down to one, and records the number it used.
Nothing is timed, so nothing is paused. `--limit N` scores only the first N items and writes
no record.
"""

from __future__ import annotations

import json
import platform
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import llm, llmcli, machine, paths, protocol, record
from .cli import Log, Parser, finish
from .errors import MeasureError
from .gpu import Gpu
from .runqueue import read_check


def parse_args(argv: list[str]):
    p = Parser(prog="mtr eval", description="Score one language-model checkpoint on a benchmark with lm-evaluation-harness. Pauses nothing.")
    p.add_argument("model_dir", type=Path)
    p.add_argument("--size", required=True)
    p.add_argument("--benchmark", required=True, choices=sorted(protocol.LLM_BENCHMARKS))
    p.add_argument("--limit", type=int, help="Score only the first N items, to try it. Writes no record")
    p.add_argument("--model-id")
    p.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")
    return p.parse_args(argv)


def lmeval_project() -> Path:
    return paths.tool_dir() / "lmeval"


def lmeval_version() -> str:
    found = re.search(r'"lm-eval(?:\[[^\]]*\])?==([^"]+)"', (lmeval_project() / "pyproject.toml").read_text())
    return found.group(1) if found else "unknown"


def server_args(a, slots: int) -> list[str]:
    args = ["-c", str(slots * protocol.LLM_EVAL_SLOT_CONTEXT), "-np", str(slots), "-ngl", "999", "--fit", "off", "--no-webui", "-lv", "4"]
    if protocol.LLM_EVAL_THINKING == "off":
        args += list(getattr(a, "THINKING_OFF_ARGS", []))
    return args + list(getattr(a, "SERVER_ARGS", []))


def start_server(binary: Path, gguf: Path, args: list[str], log_path: Path) -> tuple[subprocess.Popen, int]:
    port = llm.free_port()
    cmd = [str(binary), "-m", str(gguf), *args, "--host", "127.0.0.1", "--port", str(port)]
    with log_path.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        llm.wait_ready(port, proc, log_path)
        layers = llm.gpu_layers(log_path)
        if layers and layers[0] < layers[1]:
            raise MeasureError("OOM", f"Only {layers[0]} of {layers[1]} layers fit on the GPU")
    except MeasureError:
        stop(proc)
        raise
    return proc, port


def free_mib(index: int) -> float:
    gpu = Gpu(index)
    try:
        return (gpu.total_bytes() - gpu.used_bytes()) / 2**20
    finally:
        gpu.close()


def stop(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def newest(directory: Path, pattern: str) -> Path:
    found = sorted(directory.rglob(pattern), key=lambda p: p.stat().st_mtime)
    if not found:
        raise MeasureError("EVAL", f"lm-eval wrote no {pattern} under {directory}")
    return found[-1]


def metric_key(k: str) -> str:
    """lm-eval's "name,filter" as one key: "exact_match,flexible-extract" -> "exact_match_flexible-extract"."""
    name, _, filt = k.partition(",")
    return name if filt in ("", "none") else f"{name}_{filt}"


def items_sha256(samples: Path) -> str:
    """sha256 of the benchmark items as scored, in order: pins the data without the dataset's files."""
    docs = [json.loads(line) for line in samples.read_text().splitlines() if line.strip()]
    docs.sort(key=lambda d: d["doc_id"])
    import hashlib

    return hashlib.sha256(json.dumps([d["doc"] for d in docs], sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def main(argv: list[str], log: Log) -> int:
    args = parse_args(argv)
    machine.pin_gpu(args.gpu)
    bench = protocol.LLM_BENCHMARKS[args.benchmark]
    model_dir = args.model_dir.resolve()
    model_id = args.model_id or model_dir.name
    out = model_dir / "evals" / args.size / f"{args.benchmark}.json"
    log_path = out.with_suffix(".log") if not args.limit else out.with_name(f"{args.benchmark}.limit.log")
    try:
        return evaluate(args, bench, model_dir, model_id, out, log_path, log)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))


def evaluate(args, bench: dict, model_dir: Path, model_id: str, out: Path, log_path: Path, log: Log) -> int:
    log(f"mtr-measure eval {model_id} / {args.size} on {bench['name']} with lm-evaluation-harness {lmeval_version()}"
        + (f" (first {args.limit} items only, no record)" if args.limit else ""))
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    read_check(model_dir, args.size)
    a = llmcli.load(model_dir, args.size)
    paths = llmcli.weight_paths(a, model_id, args.size, download=False)
    binary = llmcli.server(download=False, gpu=args.gpu)
    work = out.parent / f"{args.benchmark}{'.limit' if args.limit else ''}.lmeval"
    work.mkdir(parents=True, exist_ok=True)
    server_log = work / "server.log"

    slots = protocol.LLM_EVAL_SLOTS
    while True:
        sargs = server_args(a, slots)
        thinking = protocol.LLM_EVAL_THINKING if getattr(a, "THINKING_OFF_ARGS", None) else "not switchable"
        log(f"starting llama-server with {slots} slots of {protocol.LLM_EVAL_SLOT_CONTEXT} tokens, thinking {thinking}")
        try:
            proc, port = start_server(binary, paths[0], sargs, server_log)
        except MeasureError as e:
            if e.code != "OOM" or slots == 1:
                raise
            log(f"  doesn't fit next to the other programs on the GPU ({e}); trying {slots // 2} slots")
            slots //= 2
            continue
        left = free_mib(args.gpu)
        if left >= protocol.EVAL_VRAM_MARGIN_MIB:
            log(f"  loaded; {left:.0f} MiB of GPU memory left for other programs")
            break
        stop(proc)
        if slots == 1:
            raise MeasureError("VRAM_LOW", f"With one slot only {left:.0f} MiB of GPU memory would be left for other programs, "
                                           f"less than the {protocol.EVAL_VRAM_MARGIN_MIB} MiB margin")
        log(f"  only {left:.0f} MiB would be left for other programs; trying {slots // 2} slots")
        slots //= 2

    if protocol.LLM_EVAL_THINKING == "off":
        try:
            llm.check_no_thinking(port)
        except MeasureError:
            stop(proc)
            raise
    t0 = time.perf_counter()
    lm_log = work / "lm-eval.log"
    cmd = ["uv", "run", "--project", str(lmeval_project()), "lm-eval", "run",
           "--model", "local-chat-completions",
           "--model_args", f"model={model_id},base_url=http://127.0.0.1:{port}/v1/chat/completions,"
                           f"num_concurrent={slots},max_retries=3,timeout=1800",
           "--tasks", bench["task"], "--apply_chat_template", "--seed", "0",
           "--output_path", str(work), "--log_samples"]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    log(f"running lm-eval (its output: {lm_log})")
    try:
        with lm_log.open("w") as f:
            f.write("$ " + shlex.join(cmd) + "\n")
            f.flush()
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=work).returncode
        if proc.poll() is not None:
            raise llm.server_failure(server_log)
    finally:
        stop(proc)
    duration = time.perf_counter() - t0
    if rc != 0:
        tail = [l for l in lm_log.read_text(errors="replace").splitlines() if l.strip()][-5:]
        raise MeasureError("EVAL", f"lm-eval exited with code {rc}: {' | '.join(tail)}")

    results_file = newest(work, "results_*.json")
    samples = newest(work, f"samples_{bench['task']}_*.jsonl")
    results = json.loads(results_file.read_text())
    task_results = results["results"][bench["task"]]
    metrics = {metric_key(k): round(float(v) * 100, 2) for k, v in task_results.items()
               if isinstance(v, (int, float)) and "stderr" not in k and k.split(",")[0] not in ("alias", "sample_len")}
    if bench["metric"] not in metrics:
        raise MeasureError("EVAL", f"lm-eval's results have no {bench['metric']}: {', '.join(metrics)}")
    n = int(results.get("n-samples", {}).get(bench["task"], {}).get("effective") or sum(1 for _ in samples.open()))
    value = metrics[bench["metric"]]
    log(f"{bench['metric']}: {value} on {n} items in {duration:.0f} s ({slots} slots)")
    for k, v in metrics.items():
        log(f"  {k}: {v}", echo=False)
    numbers = {"value": value, "metrics": metrics, "items": n, "slots": slots}
    if args.limit:
        return finish(log, log_path, "OK", f"{bench['metric']} {value} on the first {n} items only: no record written", numbers)

    gpu = Gpu(args.gpu)
    try:
        gpu_name, driver = gpu.name(), gpu.driver()
    finally:
        gpu.close()
    eng = llmcli.engine.describe(binary)
    rec = {
        "kind": "llm",
        "model": model_id,
        "size": args.size,
        "weights": record.weight_entries(paths),
        "benchmark": {"id": args.benchmark, "name": bench["name"], "split": "as lm-eval defines it",
                      "n": n, "metric": bench["metric"], "data_sha256": items_sha256(samples)},
        "value": value,
        "metrics": metrics,
        "settings": {
            "quant": a.WEIGHTS[args.size]["quant"],
            "evaluator": f"lm-eval=={lmeval_version()}",
            "task": bench["task"],
            "task_version": results.get("versions", {}).get(bench["task"]),
            "num_fewshot": results.get("n-shot", {}).get(bench["task"]),
            "generation": results.get("configs", {}).get(bench["task"], {}).get("generation_kwargs"),
            "chat_template": "the model's own, from the GGUF (llama-server --jinja)",
            "thinking": protocol.LLM_EVAL_THINKING if getattr(a, "THINKING_OFF_ARGS", None) else "the model has no thinking switch",
            "slots": slots,
            "slot_context": protocol.LLM_EVAL_SLOT_CONTEXT,
            "server_args": sargs,
        },
        "env": {"gpu": gpu_name, "driver": driver, "python": platform.python_version(),
                "engine": f"{eng['name']} {eng['build']}", "engine_sha256": eng["binary_sha256"], "cuda": eng["cuda"]},
        "protocol": protocol.PROTOCOL_VERSION,
        "evaluated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(duration),
        "predictions_sha256": record.sha256_file(samples),
        "provenance": machine.provenance(model_dir, None),
    }
    problems = record.validate(rec, "eval-record")
    for prob in problems:
        log(f"SCHEMA: {prob}")
    if problems:
        return finish(log, log_path, "SCHEMA", f"The eval record fails the schema: {problems[0]}", numbers)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    log(f"wrote {out} (lm-eval's samples stay in {work.name}/ as evidence and are never pushed)")
    return finish(log, log_path, "OK", f"{model_id} / {args.size}: {bench['name']} {bench['metric']} {value}",
                  {"record": str(out), **numbers})
