"""Housekeeping for model directories: start one from the template, and free its disk space.

    mtr new <id> [--kind image|llm]   models/<id>/ from models/_template/ or _template-llm/
    mtr clean <model> [--yes]         remove its uv environment and weights, then `uv cache prune`

Both run in the tool's own environment, so `clean` never has to build the model's environment
just to delete it. It learns where the weights are from the model's passed checks
(.check/<size>.json), which list the files that load() read.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

from . import paths, record
from .cli import Log, Parser
from .errors import MeasureError, emit

SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TEMPLATES = {"image": "_template", "llm": "_template-llm"}


def cmd_new(argv: list[str], log: Log) -> int:
    p = Parser(prog="mtr new", description="Start models/<id>/ from the template for its kind.")
    p.add_argument("id", help="The model family id, a lowercase-hyphen slug such as rf-detr")
    p.add_argument("--kind", choices=sorted(TEMPLATES), default="image", help="image (default) or llm (a GGUF through llama.cpp)")
    args = p.parse_args(argv)
    if not SLUG.match(args.id):
        raise MeasureError("USAGE", f"{args.id!r} is not a slug: use lowercase letters and digits joined by single hyphens")
    models = paths.root() / "models"
    dest = models / args.id
    if dest.exists():
        raise MeasureError("USAGE", f"{dest} already exists. Edit it instead.")
    shutil.copytree(models / TEMPLATES[args.kind], dest)
    for name, old, new in (("pyproject.toml", "mtr-model-template", f"mtr-model-{args.id}"),
                           ("meta.toml", 'id = "model-id"', f'id = "{args.id}"')):
        f = dest / name
        if f.is_file():
            f.write_text(f.read_text().replace(old, new))
    for f in sorted(dest.rglob("*")):
        log(f"created {f.relative_to(paths.root())}")
    todo = "name the GGUF files in adapter.py" if args.kind == "llm" else "follow the official README to fill in adapter.py and pyproject.toml"
    log(f"next: {todo}, then `mtr check {args.id} --size <size>` (docs/adapters.md)")
    return emit("OK", f"Created models/{args.id}/ ({args.kind})", {"dir": str(dest)}, log=log)


def tree_bytes(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        return path.lstat().st_size
    return sum(f.lstat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())


def remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def cmd_clean(argv: list[str], log: Log) -> int:
    p = Parser(prog="mtr clean", description="Remove a model's uv environment and weights to free disk space, then prune uv's cache.")
    p.add_argument("model_dir", type=paths.model_dir, help="The model: an id such as rf-detr, or its directory")
    p.add_argument("--yes", action="store_true", help="Delete. Without it, only list what would go")
    args = p.parse_args(argv)
    model_dir = args.model_dir.resolve()
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No adapter.py in {model_dir}")

    targets: list[Path] = []
    checked: list[str] = []
    for f in sorted((model_dir / ".check").glob("*.json")):
        check = json.loads(f.read_text())
        checked.append(f.stem)
        targets += [Path(w) for w in check.get("weight_files", [])]
    venv = model_dir / ".venv"
    if record.project_dir(model_dir) == model_dir and venv.is_dir():
        targets.append(venv)
    elif record.project_dir(model_dir) != model_dir:
        log(f"keeping {record.project_dir(model_dir) / '.venv'}: other models share it")
    existing = [t for t in dict.fromkeys(targets) if t.exists() or t.is_symlink()]
    total = sum(tree_bytes(t) for t in existing)
    for t in existing:
        log(f"{'removing' if args.yes else 'would remove'} {t} ({tree_bytes(t) / 2**20:.0f} MiB)")
    log(f"weights are known for the checked size(s): {', '.join(checked) or 'none'}. Others' weights, if any, aren't listed.")
    if not args.yes:
        return emit("OK", f"Would free about {total / 2**30:.2f} GiB. Pass --yes to delete", {"bytes": total, "paths": [str(t) for t in existing]}, log=log)

    for t in existing:
        remove(t)
    pruned = subprocess.run(["uv", "cache", "prune"], capture_output=True, text=True)
    log((pruned.stderr or pruned.stdout).strip() or "uv cache prune: done")
    free = shutil.disk_usage(paths.root()).free
    return emit("OK", f"Freed about {total / 2**30:.2f} GiB, plus what uv cache prune removed. {free / 2**30:.0f} GiB free now",
                {"bytes": total, "free_bytes": free}, log=log)


def main(cmd: str, argv: list[str], log: Log) -> int:
    if cmd == "new":
        return cmd_new(argv, log)
    if cmd == "clean":
        return cmd_clean(argv, log)
    raise MeasureError("USAGE", f"Unknown command {cmd!r}")
