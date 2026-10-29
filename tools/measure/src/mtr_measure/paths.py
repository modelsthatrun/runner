"""Where things are in a checkout of the measuring tool: models/, timing/ and tools/measure/ under one root.

The tool always runs from a checkout (adapters and their locks live in it), so the root is found from
this file's own location, never from the working directory.
"""

from __future__ import annotations

from pathlib import Path

from .errors import MeasureError


def tool_dir() -> Path:
    """tools/measure/, the uv project of the tool itself."""
    return Path(__file__).resolve().parents[2]


def root() -> Path:
    """The checkout: the directory that holds models/ and tools/measure/."""
    d = tool_dir().parents[1]
    if not (d / "models").is_dir():
        raise MeasureError("FAILED", f"Can't find models/ next to tools/measure/ in {d}. Run the tool from a checkout of its repository")
    return d


def model_dir(arg: str | Path) -> Path:
    """A model directory from what the user typed: a path (models/rf-detr) or a bare id (rf-detr)."""
    p = Path(arg)
    if (p / "adapter.py").is_file() or len(p.parts) > 1:
        return p
    candidate = root() / "models" / p.name
    return candidate if (candidate / "adapter.py").is_file() else p
