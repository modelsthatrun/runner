"""The fixed input each task is timed on (timing/timing.toml). Part of the protocol: every model of a
task is timed on the same file, and the file must match its sha256. Image tasks have an image; text-to-image
has a prompt, a UTF-8 text file."""

from __future__ import annotations

from pathlib import Path

from . import paths, record
from .errors import MeasureError


def _table() -> dict:
    path = paths.root() / "timing" / "timing.toml"
    if not path.is_file():
        raise MeasureError("FAILED", f"{path} is missing. The checkout is incomplete")
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10
        raise MeasureError("USAGE", "Reading timing/timing.toml needs Python 3.11 or newer. Pass --input instead") from None
    return tomllib.loads(path.read_text())


def _entry(t: dict, task: str | None) -> dict | None:
    """The timing input of a task: an image ([images.<name>]), or for text-to-image a prompt ([prompts.<name>])."""
    if not task:
        return None
    name = t.get("tasks", {}).get(task)
    if not name:
        return None
    for table in ("images", "prompts"):
        if name in t.get(table, {}):
            return t[table][name]
    raise MeasureError("FAILED", f'timing/timing.toml names "{name}" for {task}, but has no [images.{name}] or [prompts.{name}]')


def checked(info: dict) -> Path:
    f = paths.root() / "timing" / info["file"]
    if not f.is_file() or record.sha256_file(f) != info["sha256"]:
        raise MeasureError("FAILED", f"{f} is missing or doesn't match its sha256 in timing/timing.toml. Restore it from git")
    return f


def input_for(task: str | None) -> Path | None:
    """The task's timing input, checked against its sha256, or None when the task has none."""
    info = _entry(_table(), task)
    return checked(info) if info else None


def check_all() -> int:
    """Check every timing input against its sha256 (`mtr doctor`). Returns how many there are."""
    t = _table()
    infos = [*t.get("images", {}).values(), *t.get("prompts", {}).values()]
    for info in infos:
        checked(info)
    return len(infos)


def any_image() -> Path:
    """An image to try an adapter on, for `check` of a task without a timing image of its own."""
    t = _table()
    info = next(iter(t["images"].values()))
    return paths.root() / "timing" / info["file"]


def sha_for(task: str | None) -> str | None:
    """The sha256 the task's timing input must have, from timing/timing.toml, or None when it has none."""
    info = _entry(_table(), task)
    return info["sha256"] if info else None
