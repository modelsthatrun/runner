"""The fixed input each task is timed on (timing/timing.toml). Part of the protocol: every model of a
task is timed on the same file, and the file must match its sha256."""

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


def image_for(task: str | None) -> Path | None:
    """The task's timing image, checked against its sha256, or None when the task has none."""
    if not task:
        return None
    t = _table()
    name = t.get("tasks", {}).get(task)
    if not name:
        return None
    info = t["images"][name]
    f = paths.root() / "timing" / info["file"]
    if not f.is_file() or record.sha256_file(f) != info["sha256"]:
        raise MeasureError("FAILED", f"{f} is missing or doesn't match its sha256 in timing/timing.toml. Restore it from git")
    return f


def any_image() -> Path:
    """An image to try an adapter on, for `check` of a task without a timing image of its own."""
    t = _table()
    info = next(iter(t["images"].values()))
    return paths.root() / "timing" / info["file"]


def sha_for(task: str | None) -> str | None:
    """The sha256 the task's timing image must have, from timing/timing.toml, or None when it has none."""
    if not task:
        return None
    t = _table()
    name = t.get("tasks", {}).get(task)
    return t["images"][name]["sha256"] if name else None
