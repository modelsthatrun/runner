"""mtr: measure open models on your GPU by one protocol, and send the results to ModelsThatRun.

    mtr doctor                                   is this machine ready? What would a record say about it?
    mtr models                                   the models you can measure, and what you measured
    mtr check <model> --size <size>              does the adapter work here? Pauses nothing, writes no record
    mtr run <model> --size <size>                one timed measurement: models/<id>/runs/<size>.json
    mtr bench <model> [--size <size>…]           check, then run, every size (or the ones named)
    mtr eval <model> --size <size> --benchmark <name>    accuracy on a standard benchmark
    mtr results                                  your records, and which are sent
    mtr login | logout | whoami                  sign in to the inbox with GitHub
    mtr submit <model> [--size <size>…]          send your records; they show on modelsthatrun.dev once reviewed
    mtr submissions                              what you sent, and what the review said

    mtr queue add|list|run|drop                  batch timed runs into one pause
    mtr new <id> [--kind image|llm]              start an adapter for a new model
    mtr clean <model> [--yes]                    free the disk a model used
    mtr outputs <model> --size <size> --images <dir>     draw outputs on test images (maintainers)

<model> is an id such as rf-detr (`mtr models` lists them) or a directory with adapter.py. Every command
ends with one line, `mtr-result: {json}`, with a status, a code, a message and a hint (errors.py).

check, run, eval and outputs load the model, so they run in the model's own uv environment, from its
committed uv.lock. Everything else runs in the tool's environment.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import adapter as ad
from . import paths, record, timing
from .errors import MeasureError, emit, parse_result

MODEL_COMMANDS = ("check", "run", "eval", "outputs")
TOOL_COMMANDS = ("queue", "new", "clean")


def _uv() -> str:
    uv = os.environ.get("UV") or shutil.which("uv")
    if not uv:
        raise MeasureError("UNSUPPORTED", "uv isn't on PATH. Install it: https://docs.astral.sh/uv/")
    return uv


def project_for(model_dir: Path) -> Path:
    """The uv project the model runs in (record.project_dir), refusing a model that has none."""
    project = record.project_dir(model_dir)
    if not (project / "pyproject.toml").is_file():
        raise MeasureError("USAGE", f"No pyproject.toml in {model_dir} or its parent. Start the model with `mtr new <id>`.")
    return project


def lock_is_current(project: Path) -> bool:
    """Whether uv.lock still matches pyproject.toml. Resolves from uv's cache; changes nothing."""
    proc = subprocess.run([_uv(), "lock", "--check", "--project", str(project)], capture_output=True, text=True)
    return proc.returncode == 0


def model_command(cmd: str, model_dir: Path, rest: list[str]) -> list[str]:
    """The command that runs `cmd` for this model inside its uv environment."""
    project = project_for(model_dir)
    lock = project / "uv.lock"
    uv = [_uv(), "run", "--project", str(project)]
    # From the committed lock, never re-resolved: a newer uv must not rewrite it, which would change
    # lock_sha256 and mark the checkout as modified. A model's first check creates its lock.
    if lock.is_file():
        uv.append("--frozen")
    module = "mtr_measure.outputs" if cmd == "outputs" else "mtr_measure"
    args = [str(model_dir)] + rest
    return [*uv, "python", "-m", module, *([] if cmd == "outputs" else [cmd]), *args]


def run_in_model(cmd: str, model_dir: Path, rest: list[str]) -> int:
    project = project_for(model_dir)
    if cmd == "check" and (project / "uv.lock").is_file() and project != paths.tool_dir() and not lock_is_current(project):
        raise MeasureError("LOCK_STALE", f"{project / 'uv.lock'} doesn't match {project / 'pyproject.toml'}")
    env = dict(os.environ)
    env.pop("VIRTUAL_ENV", None)  # the tool's own environment; uv would only warn about it
    argv = model_command(cmd, model_dir, rest)
    sys.stdout.flush()
    os.execvpe(argv[0], argv, env)
    return 0  # not reached


def split_model(rest: list[str]) -> tuple[Path, list[str]]:
    """The model argument (the first one that isn't an option) and everything else."""
    for i, a in enumerate(rest):
        if not a.startswith("-"):
            return paths.model_dir(a), rest[:i] + rest[i + 1:]
        if a in ("-h", "--help"):
            break
    raise MeasureError("USAGE", "Name the model: an id such as rf-detr (`mtr models` lists them), or its directory")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    try:
        if cmd in MODEL_COMMANDS:
            if rest[:1] in (["-h"], ["--help"]):
                from . import cli

                return cli.main([cmd, "--help"]) if cmd != "outputs" else _outputs_help()
            model_dir, others = split_model(rest)
            if not (model_dir / "adapter.py").is_file():
                raise MeasureError("USAGE", f"No model {str(model_dir)!r}. `mtr models` lists them; `mtr new <id>` starts one.")
            return run_in_model(cmd, model_dir, others)
        if cmd in TOOL_COMMANDS:
            from . import cli

            return cli.main(argv)
        if cmd == "doctor":
            from . import doctor

            return doctor.main(rest)
        if cmd == "models":
            return cmd_models(rest)
        if cmd == "bench":
            return cmd_bench(rest)
        if cmd in ("results", "login", "logout", "whoami", "submit", "submissions"):
            from . import submit

            return submit.main(cmd, rest)
        raise MeasureError("USAGE", f"Unknown command {cmd!r}. `mtr --help` lists them")
    except MeasureError as e:
        print(f"FAILED: {e}", flush=True)
        return emit(e.code, str(e))


def _outputs_help() -> int:
    from . import outputs

    outputs.parse_args(["--help"])
    return 0


# --- models --------------------------------------------------------------------------------------


def adapters() -> list[Path]:
    root = paths.root() / "models"
    return sorted(d for d in root.iterdir() if (d / "adapter.py").is_file() and not d.name.startswith("_"))


def local_runs(model_dir: Path) -> list[str]:
    runs = model_dir / "runs"
    return sorted(f.name[: -len(".json")] for f in runs.glob("*.json") if f.name.count(".") == 1) if runs.is_dir() else []


def cmd_models(argv: list[str]) -> int:
    from .cli import Parser

    Parser(prog="mtr models", description="List the models this checkout can measure. The result line holds the same as JSON.").parse_args(argv)
    rows = []
    for d in adapters():
        f = d / "adapter.py"
        sizes = ad.sizes_of_file(f)
        measured = sorted(local_runs(d), key=lambda s: sizes.index(s) if sizes and s in sizes else len(sizes or []))
        kind, task = ad.kind_of_file(f), ad.task_of_file(f)
        rows.append({"id": d.name, "kind": kind, "task": task, "sizes": sizes, "measured": measured,
                     "timing_sha256": timing.sha_for(task) if kind == "image" else None})
    width = max(len(r["id"]) for r in rows) + 2
    for r in rows:
        sizes = ", ".join(r["sizes"] or ["see adapter.py"])
        mine = f"   measured here: {', '.join(r['measured'])}" if r["measured"] else ""
        print(f"{r['id']:<{width}}{r['kind']:<7}{(r['task'] or '?'):<23}{sizes}{mine}")
    print("\nnext: `mtr bench <id> --size <size>` measures one checkpoint on this GPU")
    return emit("OK", f"{len(rows)} models", {"models": rows})


# --- bench ---------------------------------------------------------------------------------------


def mtr_self() -> list[str]:
    """This command, to run a step as its own process."""
    return [sys.executable, "-m", "mtr_measure.main"]


def step(args: list[str]) -> dict:
    proc = subprocess.Popen([*mtr_self(), *args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out: list[str] = []
    assert proc.stdout
    for line in proc.stdout:
        sys.stdout.write(line)
        out.append(line)
    proc.wait()
    return parse_result("".join(out)) or {"status": "fail", "code": "FAILED", "message": f"exited with code {proc.returncode}"}


def cmd_bench(argv: list[str]) -> int:
    from .cli import Parser

    p = Parser(prog="mtr bench", description="Check, then measure, each size of a model on this GPU. Timed runs need a quiet machine.")
    p.add_argument("model_dir", type=paths.model_dir, help="The model: an id such as rf-detr, or its directory")
    p.add_argument("--size", action="append", dest="sizes", help="A size to measure. Repeat for several (default: every size)")
    p.add_argument("--gpu", type=int, default=0, help="NVML device index (default 0)")
    args = p.parse_args(argv)
    model_dir = args.model_dir
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No model {str(model_dir)!r}. `mtr models` lists them.")
    sizes = args.sizes or ad.sizes_of_file(model_dir / "adapter.py")
    if not sizes:
        raise MeasureError("USAGE", "Couldn't read the adapter's SIZES. Name them with --size")
    results = []
    for size in sizes:
        print(f"--- {model_dir.name} / {size}: check", flush=True)
        res = step(["check", str(model_dir), "--size", size, "--gpu", str(args.gpu)])
        if res["status"] == "ok":
            print(f"--- {model_dir.name} / {size}: timed run", flush=True)
            res = step(["run", str(model_dir), "--size", size, "--gpu", str(args.gpu)])
        results.append({"size": size, **{k: res.get(k) for k in ("status", "code", "message", "outputs")}})
    print("--- summary", flush=True)
    for r in results:
        print(f"{model_dir.name} / {r['size']}: {r['code']}  {r['message']}")
    done = [r for r in results if r["status"] == "ok"]
    if done:
        print(f"\nnext: `mtr submit {model_dir.name}` sends {'it' if len(done) == 1 else 'them'} for review")
    if len(done) < len(results):
        return emit("FAILED", f"{len(done)} of {len(results)} size(s) measured; read each failure above", {"results": results})
    return emit("OK", f"All {len(results)} size(s) measured", {"results": results})


if __name__ == "__main__":
    raise SystemExit(main())
