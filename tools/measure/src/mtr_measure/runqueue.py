"""The queue of timed runs: add to it any time, then measure everything in it in one pause.

    mtr queue add <model> --size <size> [--input <file>]    needs a passed `mtr check`
    mtr queue list                                          items and the expected time
    mtr queue run                                           measure them all
    mtr queue drop <n|all>

A timed run needs a quiet machine, and the pause hooks in machine.toml (desktop.py) stop programs such as
a streaming host for its length. The queue batches runs so that happens once: `queue run` pauses once,
pushes the hooks' fallback timer back before each item (desktop.extend_fallback) and runs `mtr run` for
each one, in that model's own uv environment. Items that fail stay in the queue with the reason.

The queue lives in tools/measure/queue.json, which is gitignored.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import desktop, paths, protocol, record
from .cli import Log, Parser
from .errors import MeasureError, emit, parse_result

# Running the pause hooks and starting their programs again, once per batch.
PAUSE_OVERHEAD_S = 60
# Starting a worker process and importing torch, per run.
WORKER_START_S = 6
BASELINE_S = 2.5


def queue_path() -> Path:
    return paths.tool_dir() / "queue.json"


def load() -> list[dict]:
    path = queue_path()
    if not path.is_file():
        return []
    return json.loads(path.read_text())


def save(items: list[dict]) -> None:
    path = queue_path()
    path.write_text(json.dumps(items, indent=2) + "\n")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(paths.root()))
    except ValueError:
        return str(path.resolve())


def read_check(model_dir: Path, size: str) -> dict:
    """The passed check for this checkpoint, if it still matches the adapter and the lock."""
    path = record.check_path(model_dir, size)
    if not path.is_file():
        raise MeasureError("NOT_CHECKED", f"No passed check for {model_dir.name} / {size}. Run `mtr check {rel(model_dir)} --size {size}` first.")
    check = json.loads(path.read_text())
    if not check.get("ok"):
        raise MeasureError("NOT_CHECKED", f"The last check of {model_dir.name} / {size} failed. Fix it and check again.")
    now = record.fingerprint(model_dir)
    if {k: check.get(k) for k in now} != now:
        raise MeasureError("NOT_CHECKED", f"adapter.py or uv.lock changed since {model_dir.name} / {size} was checked. Run `mtr check` again.")
    if check.get("protocol") != protocol.PROTOCOL_VERSION:
        raise MeasureError("NOT_CHECKED", f"{model_dir.name} / {size} was checked under protocol {check.get('protocol')}. Check it again.")
    return check


def estimate_s(check: dict) -> float:
    if check.get("kind") == "llm":
        return protocol.RUNS * (float(check.get("approx_run_s") or 60) + BASELINE_S)
    calls = protocol.WARMUP + protocol.TIMED
    per_run = WORKER_START_S + float(check.get("load_s") or 0) + calls * float(check.get("approx_ms") or 0) / 1000 + BASELINE_S
    return protocol.RUNS * per_run


def minutes(seconds: float) -> str:
    return f"{seconds / 60:.1f} min"


def cmd_add(argv: list[str], log: Log) -> int:
    p = Parser(prog="mtr queue add", description="Queue one timed run. The checkpoint must have passed `mtr check`.")
    p.add_argument("model_dir", type=paths.model_dir, help="The model: an id such as rf-detr, or its directory")
    p.add_argument("--size", required=True)
    p.add_argument("--input", dest="inputs", type=Path, action="append", default=[],
                   help="Image models: the input file (default: the task's timing image). Repeat for models that take several")
    p.add_argument("--note", default="", help="Why, for whoever runs the queue, e.g. 'new adapter for RF-DETR medium'")
    args = p.parse_args(argv)
    model_dir = args.model_dir.resolve()
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No adapter.py in {model_dir}")
    for f in args.inputs:
        if not f.is_file():
            raise MeasureError("USAGE", f"Input file not found: {f}")
    check = read_check(model_dir, args.size)
    items = load()
    item = {"model_dir": rel(model_dir), "size": args.size, "inputs": [rel(f) for f in args.inputs]}
    if any({k: i[k] for k in item} == item for i in items):
        return emit("OK", f"{item['model_dir']} / {args.size} is already queued", {"queued": len(items)}, log=log)
    n = max((i["n"] for i in items), default=0) + 1
    items.append({"n": n, **item, "note": args.note, "estimate_s": round(estimate_s(check)),
                  "added": datetime.now().astimezone().isoformat(timespec="seconds"), "last_error": None})
    save(items)
    total = PAUSE_OVERHEAD_S + sum(i["estimate_s"] for i in items)
    log(f"queued #{n}: {item['model_dir']} / {args.size}")
    log(f"{len(items)} run(s) waiting, about {minutes(total)} with the pause. Run them with `mtr queue run` when the machine can be quiet.")
    return emit("OK", f"Queued #{n}. {len(items)} run(s) waiting, about {minutes(total)}", {"queued": len(items), "estimate_s": total}, log=log)


def cmd_list(argv: list[str], log: Log) -> int:
    Parser(prog="mtr queue list").parse_args(argv)
    items = load()
    if not items:
        return emit("OK", "The queue is empty", {"queued": 0}, log=log)
    for i in items:
        err = f"  last failure: {i['last_error']['code']}: {i['last_error']['message']}" if i.get("last_error") else ""
        note = f"  ({i['note']})" if i.get("note") else ""
        log(f"#{i['n']} {i['model_dir']} / {i['size']}  input {', '.join(i['inputs']) or 'timing image'}  ~{minutes(i['estimate_s'])}{note}{err}")
    total = PAUSE_OVERHEAD_S + sum(i["estimate_s"] for i in items)
    return emit("OK", f"{len(items)} run(s) waiting, about {minutes(total)} with the pause",
                {"queued": len(items), "estimate_s": total, "items": items}, log=log)


def cmd_drop(argv: list[str], log: Log) -> int:
    p = Parser(prog="mtr queue drop")
    p.add_argument("which", help="Item number from `queue list`, or all")
    args = p.parse_args(argv)
    items = load()
    if args.which == "all":
        kept = []
    else:
        if not args.which.isdigit():
            raise MeasureError("USAGE", "Pass an item number from `mtr queue list`, or all")
        kept = [i for i in items if i["n"] != int(args.which)]
        if len(kept) == len(items):
            raise MeasureError("QUEUE", f"No item #{args.which} in the queue")
    save(kept)
    return emit("OK", f"Dropped {len(items) - len(kept)} item(s). {len(kept)} left", {"queued": len(kept)}, log=log)


def run_one(item: dict) -> dict:
    """`mtr run` for one item, in its own uv environment. Its output streams through."""
    root = paths.root()
    cmd = [sys.executable, "-m", "mtr_measure.main", "run", item["model_dir"], "--size", item["size"]]
    for f in item["inputs"]:
        cmd += ["--input", f]
    proc = subprocess.Popen(cmd, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: list[str] = []
    assert proc.stdout
    for line in proc.stdout:
        sys.stdout.write(line)
        out.append(line)
    proc.wait()
    result = parse_result("".join(out))
    return result or {"status": "fail", "code": "FAILED", "message": f"`mtr run` exited with code {proc.returncode} without a result"}


def cmd_run(argv: list[str], log: Log) -> int:
    Parser(prog="mtr queue run", description="Measure everything in the queue, running the pause hooks once.").parse_args(argv)
    items = load()
    if not items:
        return emit("OK", "The queue is empty", {"queued": 0}, log=log)
    total = PAUSE_OVERHEAD_S + sum(i["estimate_s"] for i in items)
    log(f"measuring {len(items)} queued run(s), about {minutes(total)}")
    results: list[dict] = []
    with desktop.paused(log):
        for i in items:
            desktop.extend_fallback()
            log(f"--- #{i['n']} {i['model_dir']} / {i['size']}")
            try:
                read_check(paths.root() / i["model_dir"], i["size"])
                res = run_one(i)
            except MeasureError as e:
                res = {"status": "fail", "code": e.code, "message": str(e)}
            results.append({"n": i["n"], "model_dir": i["model_dir"], "size": i["size"], **{k: res.get(k) for k in ("status", "code", "message", "outputs")}})

    done = {r["n"] for r in results if r["status"] == "ok"}
    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    failed = {r["n"]: r for r in results if r["status"] != "ok"}
    left = []
    for i in load():
        if i["n"] in done:
            continue
        if i["n"] in failed:
            i["last_error"] = {"code": failed[i["n"]]["code"], "message": failed[i["n"]]["message"], "at": stamp}
        left.append(i)
    save(left)

    log("--- summary")
    for r in results:
        log(f"#{r['n']} {r['model_dir']} / {r['size']}: {r['code']}  {r['message']}")
    if failed:
        return emit("QUEUE", f"{len(done)} of {len(results)} measured; {len(failed)} failed and stay in the queue", {"results": results}, log=log)
    return emit("OK", f"All {len(results)} queued run(s) measured", {"results": results}, log=log)


def main(argv: list[str], log: Log) -> int:
    if not argv:
        raise MeasureError("USAGE", "mtr queue add|list|run|drop")
    cmd, rest = argv[0], argv[1:]
    commands = {"add": cmd_add, "list": cmd_list, "run": cmd_run, "drop": cmd_drop}
    if cmd not in commands:
        raise MeasureError("USAGE", f"Unknown queue command {cmd!r}. Use add, list, run or drop")
    return commands[cmd](rest, log)
