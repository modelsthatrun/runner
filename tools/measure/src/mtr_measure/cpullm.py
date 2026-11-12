"""A language model's CPU row (runtime llamacpp-cpu, docs/protocol.md "CPU rows"): the pinned llama.cpp build the GPU
row runs, with the GPU hidden and no device, on CPU_THREADS threads. The workload is the GPU row's: a 512-token
prompt, 256 generated tokens, an 8K context. Its check compares the next token the CPU predicts, given the prompt and
the GPU's greedy answer so far, with the GPU's at eight positions (llm.next_token_match). Memory is llama-server's peak
resident set, the mapped GGUF file included.
"""

from __future__ import annotations

import statistics
import time
from datetime import datetime
from pathlib import Path

from . import checks, cpu, hostmem, llmcli, machine, protocol, record
from .cpurun import BASELINE_S, OverLimit, WorkerFailed, open_gpu, spawn, threads, write
from .errors import MeasureError


def server_args(a, n: int) -> list[str]:
    return [*protocol.LLM_CPU_SERVER_ARGS, "-t", str(n), "-tb", str(n), *getattr(a, "SERVER_ARGS", [])]


def reference_path(model_dir: Path, size: str) -> Path:
    return record.check_path(model_dir, size).with_name(f"{size}.reference.json")


def make_spec(a, model_id: str, size: str, run_dir: Path, check: bool, log, gpu_index: int) -> tuple[dict, list[Path], Path]:
    spec, weights, binary = llmcli.spec(a, model_id, size, run_dir, check=check, say=log, gpu=gpu_index)
    n = threads()
    spec.update(cpu=True, server_args=server_args(a, n), threads=n,
                warmup=protocol.LLM_CHECK_WARMUP if check else protocol.CPU_LLM_WARMUP,
                timed=protocol.LLM_CHECK_TIMED if check else protocol.CPU_LLM_TIMED,
                server_log=str(run_dir / f"{size}.llamacpp-cpu.server.log"))
    return spec, weights, binary


def cmd_check(args, model_dir: Path, model_id: str, log, finish, write_check, log_path: Path) -> int:
    from .runqueue import read_check

    runtime = "llamacpp-cpu"
    try:
        a = llmcli.load(model_dir, args.size)
        spec, weights, binary = make_spec(a, model_id, args.size, record.check_path(model_dir, args.size).parent, True, log, args.gpu)
        ref = reference_path(model_dir, args.size)
        matched = False
        try:
            read_check(model_dir, args.size)
            matched = ref.is_file()
        except MeasureError:
            pass
        if matched:
            spec["reference_in"] = str(ref)
        log(f"engine: llama.cpp {llmcli.engine.LLAMA_CPP['build']} with the GPU hidden, {spec['threads']} threads; args {' '.join(spec['server_args'])}")
        gpu = open_gpu(args.gpu)
        try:
            worker, _ = spawn(spec, "mtr_measure.llm", gpu, log)
        finally:
            if gpu is not None:
                gpu.close()
    except OverLimit as e:
        log(f"OVER_LIMIT: {e}")
        path = write_check(model_dir, args.size, {"model": model_id, "kind": "llm", "runtime": runtime, "over_limit": {"call_s": round(e.call_s, 1)},
                                                  "weight_files": [str(p) for p in weights]}, runtime)
        return finish(log, log_path, "OK", f"{model_id} / {args.size} on {runtime}: one request ran past {protocol.CPU_CALL_LIMIT_S:.0f} s",
                      {"check": str(path), "over_limit": True}, [str(e)])
    except (WorkerFailed, MeasureError) as e:
        log(f"FAILED: {e}")
        return finish(log, log_path, e.code, str(e))
    s = record.summarize_llm_run(worker, 0, 0)
    ram = hostmem.peak_mib(worker["memory"])
    m = worker.get("match")
    log(f"layers on the GPU: {worker['gpu_layers']}; load {worker['load_s']:.1f} s; sample output: {worker['sample']!r}")
    log(f"approximate: {s['decode_tps']} tokens/s decode, {s['prefill_tps']} tokens/s prefill, first token {s['ttft_ms']} ms, "
        f"{ram} MiB of memory at its peak (not measurements: other programs share the CPU)")
    warnings: list[str] = []
    if m:
        log(f"match with the GPU: top-1 agrees at {m['metrics']['top1_agree']:.0%} of {int(m['metrics']['positions'])} positions "
            f"(needs {m['tolerance']['top1_agree']:.0%}), top-5 at {m['metrics']['top5_agree']:.0%} (needs {m['tolerance']['top5_agree']:.0%})")
        if m["metrics"]["top1_agree"] < m["tolerance"]["top1_agree"] or m["metrics"]["top5_agree"] < m["tolerance"]["top5_agree"]:
            return finish(log, log_path, "NOT_MATCHED", "The CPU's next tokens differ from the GPU's answer beyond the tolerance")
    else:
        warnings.append("No match with the GPU yet: run `mtr check` on the GPU for this size (it saves the reference), then this check again")
    check = {"model": model_id, "kind": "llm", "runtime": runtime, "quant": a.WEIGHTS[args.size]["quant"], "load_s": round(worker["load_s"], 2),
             "approx_decode_tps": s["decode_tps"], "approx_ram_mib": ram, "gpu_layers": worker["gpu_layers"], "threads": spec["threads"],
             "approx_run_s": round(worker["load_s"] + 3 + (protocol.CPU_LLM_WARMUP + protocol.CPU_LLM_TIMED) * s["e2e_ms"] / 1000, 1),
             "weight_files": [str(p) for p in weights], "match": m}
    path = write_check(model_dir, args.size, check, runtime)
    log(f"wrote {path}")
    return finish(log, log_path, "OK", f"{model_id} / {args.size} works on {runtime}: about {s['decode_tps']} tokens/s and {ram} MiB",
                  {"check": str(path), "approx_decode_tps": s["decode_tps"], "approx_ram_mib": ram, "matched": bool(m)}, warnings)


def cmd_run(args, model_dir: Path, model_id: str, log, finish, out: Path, failed_log: Path, dry: bool) -> int:
    from .cpurun import not_timed, require_ready
    from .runqueue import read_check

    runtime = "llamacpp-cpu"
    warnings: list[str] = []
    try:
        check = read_check(model_dir, args.size, runtime, allow_not_timed=True)
        m = require_ready(model_dir, args.size, runtime, check)
        a = llmcli.load(model_dir, args.size)
        spec, weight_paths, binary = make_spec(a, model_id, args.size, out.parent, False, log, args.gpu)
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, failed_log, e.code, str(e))
    facts = machine.cpu_facts()
    log(f"cpu: {facts['cpu']}, {facts['cpu_cores']} cores, {spec['threads']} threads, {facts['ram_total_mib']} MiB of memory")
    log(f"{protocol.RUNS} runs, each a fresh server: {spec['warmup']} warm-up and {spec['timed']} timed requests of "
        f"{protocol.LLM_PROMPT_TOKENS} + {protocol.LLM_GEN_TOKENS} tokens")
    need = int(check.get("approx_ram_mib") or 0)
    runs: list[dict] = []
    clean = True
    gpu = open_gpu(args.gpu)
    try:
        for i in range(protocol.RUNS):
            window = cpu.Window().start()
            time.sleep(BASELINE_S)
            pre = checks.before_cpu_run(window.others(), hostmem.available_mib(), need)
            from .cli import report

            report(log, pre)
            warnings += pre.warnings
            if not pre.ok():
                clean = False
                if not args.force:
                    raise MeasureError(pre.codes[0], pre.errors[0])
            log(f"run {i + 1}/{protocol.RUNS}")
            throttled = hostmem.throttle_count()
            window = cpu.Window().start()
            try:
                worker, _ = spawn(spec, "mtr_measure.llm", gpu, log)
            except OverLimit as e:
                others = window.others()
                if others <= checks.CPU_BUSY_LIMIT_CPU and pre.ok():
                    e.beat = {**e.beat, "weights": [str(p) for p in weight_paths]}
                    return not_timed(args, model_dir, model_id, "llm", runtime, e, {**check, "runtime_version": llmcli.engine.LLAMA_CPP["build"]},
                                     facts, None, out, failed_log, dry, log, finish, warnings)
                raise MeasureError("CPU_BUSY", f"{e} while other programs used {others:.0%} of the CPU, so it isn't recorded") from e
            others = window.others()
            post = checks.after_cpu_run(others, throttled, hostmem.throttle_count(), worker["memory"], None)
            report(log, post)
            warnings += post.warnings
            if not post.ok():
                clean = False
                if not args.force:
                    raise MeasureError(post.codes[0], post.errors[0])
            s = record.summarize_llm_run(worker, 0, 0)
            log(f"run {i + 1}: decode {s['decode_tps']} tokens/s, prefill {s['prefill_tps']} tokens/s, first token {s['ttft_ms']} ms, "
                f"ram peak {hostmem.peak_mib(worker['memory'])} MiB, other cpu {others:.1%}")
            runs.append({"worker": worker})
    except MeasureError as e:
        log(f"FAILED: {e}")
        return finish(log, failed_log, e.code, str(e), warnings=warnings)
    finally:
        if gpu is not None:
            gpu.close()
    if not m:
        return finish(log, failed_log, "NOT_MATCHED", "Every request ran within the limit this time, but the check never matched the GPU's "
                                                      "answer. Run `mtr check --runtime llamacpp-cpu` again on a quiet machine", warnings=warnings)
    weights = record.weight_entries(weight_paths)
    engine = llmcli.engine.describe(binary)
    rt = {"name": runtime, "version": engine["build"], "status": "runs-as-is", "threads": spec["threads"], "match": m}
    rec = record.build_cpu_llm(model=model_id, size=args.size, weights=weights, runs=runs, runtime=rt, quant=a.WEIGHTS[args.size]["quant"],
                               engine=engine, server_args=spec["server_args"], warmup=spec["warmup"],
                               measured=datetime.now().astimezone().isoformat(timespec="seconds"), machine=facts,
                               provenance=machine.provenance(model_dir, None))
    meas = rec["measurement"]
    spread = record.spread(meas["runs"], "decode_tps")
    log(f"decode: {meas['decode_tps']} tokens/s, prefill {meas['prefill_tps']} tokens/s, first token {meas['ttft_ms']['median']} ms "
        f"(medians of {len(runs)} runs; decode spread {spread:.1%}, the protocol allows {protocol.MAX_SPREAD:.0%}); ram peak {meas['ram_peak_mib']} MiB")
    return write(rec, runs, "llm", spread, clean, dry, args, out, failed_log, log, finish, warnings,
                 {"decode_tps": meas["decode_tps"], "prefill_tps": meas["prefill_tps"], "ttft_ms": meas["ttft_ms"]["median"],
                  "ram_peak_mib": meas["ram_peak_mib"], "spread": round(spread, 4)},
                 f"{meas['decode_tps']} tokens/s decode on the CPU, RAM peak {meas['ram_peak_mib']} MiB")


def median(xs):
    return statistics.median(xs)
