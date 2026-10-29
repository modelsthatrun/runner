"""mtr check | run | eval | queue | new | clean: check, measure and score one checkpoint (docs/protocol.md).

The `mtr` command (main.py) runs these in the model's own uv environment:

    mtr check <model> --size <size> [--input <file>]     does the adapter work? Pauses nothing, writes no record
    mtr run <model> --size <size> [--input <file>]       one timed measurement: models/<id>/runs/<size>.json
    mtr eval <model> --size <size> --benchmark <name>    accuracy on a standard benchmark. Pauses nothing
    mtr queue add | list | run | drop                    batch timed runs into one pause (runqueue.py)
    mtr new <id> [--kind image|llm]                      a model directory from the template
    mtr clean <model> [--yes]                            remove the model's environment and weights

<model> is an id such as rf-detr, or a directory with adapter.py. Image models are timed on their task's
timing image (timing/timing.toml) unless --input names another file. Every command ends with one
`mtr-result: {json}` line that says what happened and what to do next (errors.py).

A timed run checks the machine, then starts the model's worker process RUNS times. Before each run it
takes the idle baseline, polls device memory while the worker runs and takes the baseline again
afterwards. The record combines the runs (record.py); the raw timings go next to it in <size>.times.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from . import adapter as ad
from . import checks, cpu, desktop, machine, paths, protocol, record, timing
from .errors import MeasureError, emit
from .gpu import MIB, Gpu, Poller, sample_for

BASELINE_SECONDS = 1.0
SUBCOMMANDS = ("check", "run", "eval", "queue", "new", "clean")


class Parser(argparse.ArgumentParser):
    """Usage errors become a result line instead of argparse's bare exit."""

    def error(self, message: str):  # type: ignore[override]
        raise MeasureError("USAGE", f"{self.prog}: {message}")


class Log:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, line: str = "", echo: bool = True) -> None:
        self.lines.append(line)
        if echo:
            print(line, flush=True)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def model_args(p: argparse.ArgumentParser, input_required: bool) -> None:
    p.add_argument("model_dir", type=paths.model_dir, help="The model: an id such as rf-detr, or its directory")
    p.add_argument("--size", required=True, help="Checkpoint size, as in the adapter's SIZES")
    p.add_argument("--input", dest="inputs", type=Path, action="append", required=input_required,
                   help="The input file (default: the task's timing image, timing/timing.toml). Repeat for models that take several")
    p.add_argument("--model-id", help="Model id for the record (default: the directory name)")
    p.add_argument("--poll-ms", type=float, default=5.0, help="NVML polling interval, 10 ms or less (default 5)")
    p.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")


def check_poll(args: argparse.Namespace) -> None:
    if not 0 < args.poll_ms <= checks.MAX_POLL_GAP_MS:
        raise MeasureError("USAGE", f"--poll-ms must be between 0 and {checks.MAX_POLL_GAP_MS:.0f}")


def check_paths(args: argparse.Namespace) -> Path:
    model_dir = args.model_dir.resolve()
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No adapter.py in {model_dir}. List the models with `mtr models`, or start one with `mtr new <id>`.")
    for f in args.inputs or []:
        if not f.is_file():
            raise MeasureError("USAGE", f"Input file not found: {f}")
    return model_dir


def read_task(model_dir: Path) -> str | None:
    return ad.task_of_file(model_dir / "adapter.py")


def default_inputs(model_dir: Path, timed: bool) -> list[Path]:
    """The task's timing image. `check` falls back to any timing image for a task without one; a timed run doesn't."""
    f = timing.image_for(read_task(model_dir))
    if f:
        return [f]
    if timed:
        raise MeasureError("USAGE", f"The adapter's TASK ({read_task(model_dir) or 'none'}) has no timing image in timing/timing.toml. "
                                    "Pass --input <file>")
    return [timing.any_image()]


def spawn_worker(spec: dict, gpu: Gpu, poll_ms: float, gpu_index: int, module: str = "mtr_measure.worker") -> tuple[dict, object]:
    """Start the worker with `spec`, polling device memory while it runs. Returns its result and the samples.
    Image models run in worker.py, language models in llm.py."""
    with tempfile.TemporaryDirectory(prefix="mtr-measure-") as tmp:
        spec_path, result_path = Path(tmp) / "spec.json", Path(tmp) / "result.json"
        spec_path.write_text(json.dumps({**spec, "result": str(result_path)}))
        env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": str(gpu_index)}
        poller = Poller(gpu, poll_ms)
        poller.start()
        proc = subprocess.run([sys.executable, "-m", module, str(spec_path)], env=env)
        during = poller.stop()
        worker = json.loads(result_path.read_text()) if result_path.is_file() else None
    if not worker:
        raise MeasureError("ADAPTER_RUNTIME", f"The worker exited with code {proc.returncode} without a result")
    if not worker.get("ok"):
        if worker.get("traceback"):
            print(worker["traceback"], file=sys.stderr, flush=True)
        raise WorkerFailed(worker)
    return worker, during


class WorkerFailed(MeasureError):
    def __init__(self, worker: dict):
        super().__init__(worker.get("code", "ADAPTER_RUNTIME"), worker.get("error", "the worker failed"))
        self.worker = worker


def report(log: Log, r: checks.Report) -> None:
    for line in r.lines:
        log(line, echo=False)
    for w in r.warnings:
        log(f"WARNING: {w}")
    for e in r.errors:
        log(f"PROBLEM: {e}")


def finish(log: Log, log_path: Path, code: str, message: str, outputs: dict | None = None, warnings: list[str] | None = None) -> int:
    """Write the log next to the record, then end with the result line."""
    log(f"log: {log_path}")
    rc = emit(code, message, outputs, warnings, log=log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log.text())
    return rc


# --- check ---------------------------------------------------------------------------------------


def write_check(model_dir: Path, size: str, check: dict) -> Path:
    path = record.check_path(model_dir, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"ok": True, "size": size, "protocol": protocol.PROTOCOL_VERSION, **record.fingerprint(model_dir),
                                **check, "checked": datetime.now().astimezone().isoformat(timespec="seconds")}, indent=2) + "\n")
    return path


def cmd_check(argv: list[str], log: Log) -> int:
    p = Parser(prog="mtr check", description="Load the checkpoint, make a few calls and check the adapter. Pauses nothing and writes no record.")
    model_args(p, input_required=False)
    p.add_argument("--task", help="Task id, to check to_prediction() (default: the adapter's TASK)")
    args = p.parse_args(argv)
    check_poll(args)
    model_dir = check_paths(args)
    model_id = args.model_id or model_dir.name
    llm = ad.kind_of_file(model_dir / "adapter.py") == "llm"
    log_path = record.check_path(model_dir, args.size).with_suffix(".log")
    warnings: list[str] = []

    log(f"mtr check {model_id} / {args.size} (not a measurement: nothing is paused, no record is written)")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    try:
        if llm:
            from . import llmcli

            a = llmcli.load(model_dir, args.size)
            spec, weights, binary = llmcli.spec(a, model_id, args.size, record.check_path(model_dir, args.size).parent, check=True, say=log, gpu=args.gpu)
            module = "mtr_measure.llm"
            what = f"{spec['warmup']} warm-up and {spec['timed']} timed requests of {spec['prompt_tokens']} + {spec['gen_tokens']} tokens"
        else:
            inputs = [f.resolve() for f in args.inputs] if args.inputs else default_inputs(model_dir, timed=False)
            task = args.task or read_task(model_dir)
            for f in inputs:
                log(f"input: {f}")
            spec = {"adapter": str(model_dir / "adapter.py"), "size": args.size, "inputs": [str(f) for f in inputs],
                    "warmup": protocol.CHECK_WARMUP, "timed": protocol.CHECK_TIMED, "mode": "check", "task": task}
            module = "mtr_measure.worker"
            what = f"{protocol.CHECK_WARMUP} warm-up and {protocol.CHECK_TIMED} timed predict() calls"
        gpu = Gpu(args.gpu)
        try:
            window = cpu.Window().start()
            before = sample_for(gpu, BASELINE_SECONDS, args.poll_ms)
            free_mib = (gpu.total_bytes() - before.median_bytes) / MIB
            log(f"gpu: {gpu.name()}, {before.median_bytes / MIB:.0f} MiB in use, {free_mib:.0f} MiB free; "
                f"other programs use {window.others():.0%} of the CPU")
            log(f"running the worker: load, {what}")
            worker, during = spawn_worker(spec, gpu, args.poll_ms, args.gpu, module)
        finally:
            gpu.close()
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))

    approx_vram = max(0, round((during.peak_bytes - before.median_bytes) / MIB))
    if llm:
        s = record.summarize_llm_run(worker, before.median_bytes, during.peak_bytes)
        log(f"engine: llama.cpp {llmcli.engine.LLAMA_CPP['build']}, quant {a.WEIGHTS[args.size]['quant']}, "
            f"{worker['gpu_layers']} layers on the GPU, load {worker['load_s']:.1f} s")
        log(f"sample output: {worker['sample']!r}")
        log(f"approximate: {s['decode_tps']} tokens/s decode, {s['prefill_tps']} tokens/s prefill, first token {s['ttft_ms']} ms, "
            f"{approx_vram} MiB of GPU memory at {protocol.LLM_CONTEXT} tokens of context (not measurements: other programs share the GPU)")
        check = {"model": model_id, "kind": "llm", "quant": a.WEIGHTS[args.size]["quant"], "load_s": round(worker["load_s"], 2),
                 "approx_decode_tps": s["decode_tps"], "approx_vram_mib": approx_vram, "gpu_layers": worker["gpu_layers"],
                 # one run: load, requests, and the server starting and stopping
                 "approx_run_s": round(worker["load_s"] + 3 + (protocol.LLM_WARMUP + protocol.LLM_TIMED) * s["e2e_ms"] / 1000, 1),
                 "weight_files": [str(p) for p in weights]}
        summary = f"about {s['decode_tps']} tokens/s and {approx_vram} MiB"
        numbers = {"approx_decode_tps": s["decode_tps"], "approx_vram_mib": approx_vram}
    else:
        approx_ms = round(sorted(worker["times_ms"])[len(worker["times_ms"]) // 2], 1) if worker["times_ms"] else None
        for line in worker.get("checks", []):
            log(line)
        log(f"package: {worker['env']['package']}, precision {worker['precision']}, input {worker['input']}, load {worker['load_s']:.1f} s")
        log(f"output of predict(): {worker['output']}")
        log(f"approximate: {approx_ms} ms per call, {approx_vram} MiB of GPU memory (not measurements: other programs share the GPU)")
        if not task:
            warnings.append("to_prediction() was not checked: the adapter declares no TASK and there was no --task")
        check = {"model": model_id, "kind": "image", "precision": worker["precision"], "input": worker["input"],
                 "package": worker["env"]["package"], "load_s": round(worker["load_s"], 2), "approx_ms": approx_ms,
                 "approx_vram_mib": approx_vram, "task": task,
                 # Where the weights are, resolved and as listed (often a cache symlink), for `mtr clean`.
                 "weight_files": sorted(set(worker["weights"]) | set(worker.get("weight_links", [])))}
        summary = f"about {approx_ms} ms and {approx_vram} MiB"
        numbers = {"approx_ms": approx_ms, "approx_vram_mib": approx_vram}
    path = write_check(model_dir, args.size, check)
    log(f"wrote {path}")
    log("next: `mtr run` (or `mtr queue add`) for the timed run, and `mtr eval` for accuracy")
    return finish(log, log_path, "OK", f"{model_id} / {args.size} works: load {worker['load_s']:.1f} s, {summary}",
                  {"check": str(path), "load_s": check["load_s"], **numbers}, warnings)


# --- run -----------------------------------------------------------------------------------------


def cmd_run(argv: list[str], log: Log) -> int:
    p = Parser(prog="mtr run", description="Measure one checkpoint by the protocol and write models/<id>/runs/<size>.json. Runs the pause hooks in machine.toml, if any.")
    model_args(p, input_required=False)
    p.add_argument("--out", type=Path, help="Run record path (default: <model_dir>/runs/<size>.json)")
    p.add_argument("--dry-run", action="store_true", help="Measure and print the record, but don't write it")
    p.add_argument("--force", action="store_true", help="Run even if a check fails, without pausing anything. Implies --dry-run. `check` is usually what you want")
    p.add_argument("--keep-desktop", action="store_true", help="Don't run the pause hooks (the checks then refuse to run while their programs are open)")
    args = p.parse_args(argv)
    check_poll(args)
    model_dir = check_paths(args)
    model_id = args.model_id or model_dir.name
    out = (args.out or model_dir / "runs" / f"{args.size}.json").resolve()
    dry = args.dry_run or args.force
    warnings: list[str] = []
    llm = ad.kind_of_file(model_dir / "adapter.py") == "llm"
    failed_log = out.with_name(f"{args.size}.failed.log")

    log(f"mtr run {model_id} / {args.size}, protocol {protocol.PROTOCOL_VERSION}")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    try:
        if llm:
            from . import llmcli

            a = llmcli.load(model_dir, args.size)
            # Never downloads or builds: that would happen inside the pause.
            spec, weight_paths, binary = llmcli.spec(a, model_id, args.size, out.parent, check=False, say=log, gpu=args.gpu)
            inputs = None
            module, summarize, spread_key = "mtr_measure.llm", record.summarize_llm_run, "decode_tps"
            what = f"{protocol.LLM_WARMUP} warm-up and {protocol.LLM_TIMED} timed requests of {protocol.LLM_PROMPT_TOKENS} + {protocol.LLM_GEN_TOKENS} tokens"
        else:
            inputs = [f.resolve() for f in args.inputs] if args.inputs else default_inputs(model_dir, timed=True)
            for f in inputs:
                log(f"input: {f} sha256 {record.sha256_file(f)}")
            spec = {"adapter": str(model_dir / "adapter.py"), "size": args.size, "inputs": [str(f) for f in inputs],
                    "warmup": protocol.WARMUP, "timed": protocol.TIMED}
            module, summarize, spread_key = "mtr_measure.worker", record.summarize_run, "median"
            what = f"{protocol.WARMUP} warm-up and {protocol.TIMED} timed predict() calls"
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, failed_log, e.code, str(e))
    log(f"{protocol.RUNS} runs, each a fresh process: load, {what}")

    clean = True
    runs: list[dict] = []
    # The pause hooks (machine.toml) hold from before the first baseline until after the last one (desktop.py).
    pause = contextlib.nullcontext() if args.force or args.keep_desktop else desktop.paused(log)
    try:
        with pause:
            gpu = Gpu(args.gpu)
            try:
                gpu_name, driver = gpu.name(), gpu.driver()
                facts = machine.facts(gpu)
                log(f"gpu: {gpu_name}, driver {driver}, {facts['gpu_memory_mib']} MiB, compute {facts['compute_capability']}, "
                    f"power limit {facts['power_limit_w']} W of {facts['power_default_w']} W")
                log(f"cpu: {facts['cpu']}; os: {facts['os']}")
                for i in range(protocol.RUNS):
                    window = cpu.Window().start()
                    before = sample_for(gpu, BASELINE_SECONDS, args.poll_ms)
                    pre = checks.before_run(gpu.processes(), before, window.others())
                    report(log, pre)
                    warnings += pre.warnings
                    if not pre.ok():
                        clean = False
                        if not args.force:
                            log("Stopped before loading the model. Fix the problems above, or try `mtr check`, which pauses nothing.")
                            raise MeasureError(pre.codes[0], pre.errors[0])
                    log(f"run {i + 1}/{protocol.RUNS}")
                    window = cpu.Window().start()
                    worker, during = spawn_worker(spec, gpu, args.poll_ms, args.gpu, module)
                    cpu_run = window.others()
                    time.sleep(0.5)
                    after = sample_for(gpu, BASELINE_SECONDS, args.poll_ms)
                    post = checks.after_run(before, after, during, cpu_run)
                    report(log, post)
                    warnings += post.warnings
                    if not post.ok():
                        clean = False
                        if not args.force:
                            raise MeasureError(post.codes[0], post.errors[0])
                    s = summarize(worker, before.median_bytes, during.peak_bytes)
                    log(f"run {i + 1}: " + ", ".join(f"{k} {v}" for k, v in s.items()) + f", load {worker['load_s']:.1f} s, other cpu {cpu_run:.1%}")
                    if "times_ms" in worker:
                        log(f"run {i + 1} timed iterations (ms): " + " ".join(f"{t:.3f}" for t in worker["times_ms"]), echo=False)
                    else:
                        log(f"run {i + 1} requests: {json.dumps(worker['requests'])}", echo=False)
                    runs.append({"worker": worker, "baseline_bytes": before.median_bytes, "peak_bytes": during.peak_bytes})
            finally:
                gpu.close()
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, failed_log, e.code, str(e), warnings=warnings)

    measured = datetime.now().astimezone().isoformat(timespec="seconds")
    worker = runs[-1]["worker"]
    try:
        weights = record.weight_entries(weight_paths if llm else [Path(p) for p in worker["weights"]])
    except FileNotFoundError as e:
        return finish(log, failed_log, "WEIGHTS_MISSING", str(e))
    except ValueError as e:
        return finish(log, failed_log, "ADAPTER_INTERFACE", str(e))
    if llm:
        rec = record.build_llm(
            model=model_id, size=args.size, weights=weights, runs=runs, quant=a.WEIGHTS[args.size]["quant"],
            engine=llmcli.engine.describe(binary), server_args=spec["server_args"], gen_tokens=protocol.LLM_GEN_TOKENS,
            context=protocol.LLM_CONTEXT, gpu_name=gpu_name, driver=driver, warmup=protocol.LLM_WARMUP, measured=measured,
            machine=facts, provenance=machine.provenance(model_dir, inputs),
        )
        m = rec["measurement"]
        spread = record.spread(m["runs"], spread_key)
        log(f"engine: {m['engine']['name']} {m['engine']['build']} (CUDA {m['engine']['cuda']}), quant {m['quant']}, {m['gpu_layers']} layers on the GPU")
        log(f"decode: {m['decode_tps']} tokens/s, prefill {m['prefill_tps']} tokens/s, first token {m['ttft_ms']['median']} ms, "
            f"{m['gen_tokens']} tokens in {m['e2e_ms']['median']} ms (medians of {len(runs)} runs; decode spread {spread:.1%}, "
            f"the protocol allows {protocol.MAX_SPREAD:.0%})")
        log(f"vram peak: {m['vram_peak_mib']} MiB at {m['context']} tokens of context, over a {m['env']['baseline_mib']} MiB baseline")
        log(f"sample output: {worker['sample']!r}")
        numbers = {"decode_tps": m["decode_tps"], "prefill_tps": m["prefill_tps"], "ttft_ms": m["ttft_ms"]["median"],
                   "vram_peak_mib": m["vram_peak_mib"], "spread": round(spread, 4)}
        headline = f"{m['decode_tps']} tokens/s decode, VRAM peak {m['vram_peak_mib']} MiB"
        schema = "llm-run-record"
    else:
        rec = record.build(
            model=model_id, size=args.size, weights=weights, runs=runs, gpu_name=gpu_name, driver=driver,
            image_digest=os.environ.get("MTR_IMAGE_DIGEST") or None, warmup=protocol.WARMUP, measured=measured,
            machine=facts, provenance=machine.provenance(model_dir, inputs),
        )
        m = rec["measurement"]
        spread = record.spread(m["runs"], spread_key)
        log(f"worker device: {worker['device']}")
        log(f"output of predict(): {worker['output']}")
        log(f"precision {m['precision']}, input {m['input']}")
        log(f"latency: median {m['latency_ms']['median']} ms, p90 {m['latency_ms']['p90']} ms "
            f"(medians of {len(runs)} runs; spread {spread:.1%}, the protocol allows {protocol.MAX_SPREAD:.0%})")
        log(f"vram peak: {m['vram_peak_mib']} MiB over a {m['env']['baseline_mib']} MiB baseline (torch max reserved {m['torch_max_reserved_mib']} MiB)")
        numbers = {"latency_median_ms": m["latency_ms"]["median"], "latency_p90_ms": m["latency_ms"]["p90"],
                   "vram_peak_mib": m["vram_peak_mib"], "spread": round(spread, 4)}
        headline = f"median {m['latency_ms']['median']} ms, VRAM peak {m['vram_peak_mib']} MiB"
        schema = "run-record"
    for w in weights:
        log(f"weights: {w['file']} {w['bytes'] / MIB:.1f} MiB sha256 {w['sha256']}")

    code, message = "OK", ""
    problems = record.validate(rec, schema)
    for prob in problems:
        log(f"SCHEMA: {prob}")
    if problems:
        code, message = "SCHEMA", f"The record fails the schema: {problems[0]}"
    elif spread > protocol.MAX_SPREAD:
        code, message = "VARIANCE", f"The {len(runs)} runs spread by {spread:.1%}, above the {protocol.MAX_SPREAD:.0%} the protocol allows"
    elif not clean:
        code, message = "FAILED", "A check failed, so this run can't be a record"

    if dry or code != "OK":
        why = message or ("--force" if args.force else "--dry-run")
        log(f"No run record written ({why}). The record would have been:")
        log(json.dumps(rec, indent=2))
        if code == "OK":
            return finish(log, out.with_name(f"{args.size}.dry-run.log"), "OK", f"Dry run of {model_id} / {args.size}: no record written", numbers, warnings)
        return finish(log, failed_log, code, message, numbers, warnings)

    times = record.times(runs, llm)
    problems = record.validate(times, "run-times")
    if problems:
        return finish(log, failed_log, "SCHEMA", f"The timings fail their schema: {problems[0]}", numbers, warnings)
    if rec["provenance"]["tool"]["dirty"]:
        warnings.append("tools/measure or the model directory has uncommitted changes, so this record can't be submitted")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2) + "\n")
    times_path = out.with_name(f"{args.size}.times.json")
    times_path.write_text(json.dumps(times) + "\n")
    log(f"wrote {out} and {times_path.name}")
    # Not .json, so it is never read as a record. It stays next to the record as evidence of the run.
    return finish(log, out.with_suffix(".log"), "OK", f"{model_id} / {args.size}: {headline}",
                  {"record": str(out), "times": str(times_path), **numbers}, warnings)


# --- entry ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv else 2
    if argv[0] not in SUBCOMMANDS:
        argv.insert(0, "run")  # the oldest form of the command: <model> --size … without `run`
    cmd, rest = argv[0], argv[1:]
    log = Log()
    try:
        if cmd == "check":
            return cmd_check(rest, log)
        if cmd == "run":
            return cmd_run(rest, log)
        if cmd == "queue":
            from . import runqueue as queue

            return queue.main(rest, log)
        if cmd == "eval":
            from . import eval as evaluate

            return evaluate.main(rest, log)
        from . import manage

        return manage.main(cmd, rest, log)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return emit(e.code, str(e), log=log)


if __name__ == "__main__":
    raise SystemExit(main())
