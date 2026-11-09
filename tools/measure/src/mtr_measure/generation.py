"""Text-to-image models (KIND = "generation"): what the tool passes predict(), and what it checks of the image.

The tool, not the adapter, makes each call's input: the prompt, a fixed seed, and the official steps, guidance
and resolution that the adapter declares in GENERATION. So the record says exactly what every call was asked
to do. The adapter turns it into the official pipeline call and returns the image (docs/adapters.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import MeasureError

LATENCY_SCOPE = "predict(): one image from the prompt: text encoding, every denoising step and the VAE decode, to a PIL image; prompt read beforehand"
GENERATORS = ("cpu", "cuda")


@dataclass(frozen=True)
class Generate:
    """One call's input. `seed` seeds a torch.Generator on `generator` ("cpu" or "cuda"), as the model card does."""

    prompt: str
    seed: int
    steps: int
    guidance: float
    width: int
    height: int
    generator: str

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def settings(a: Any, size: str) -> dict:
    """The adapter's GENERATION entry for a size, checked: steps, guidance, resolution and generator."""
    from .adapter import AdapterError

    g = getattr(a, "GENERATION", {}).get(size)
    if not isinstance(g, dict):
        raise AdapterError(f'GENERATION has no entry for size "{size}"')
    lacking = [k for k in ("steps", "guidance", "resolution", "generator") if k not in g]
    if lacking:
        raise AdapterError(f'GENERATION["{size}"] is missing {", ".join(lacking)}')
    if not isinstance(g["steps"], int) or g["steps"] < 1:
        raise AdapterError(f'GENERATION["{size}"]["steps"] must be a positive integer')
    if not isinstance(g["guidance"], (int, float)) or g["guidance"] < 0:
        raise AdapterError(f'GENERATION["{size}"]["guidance"] must be a number, 0 or more')
    if not isinstance(g["resolution"], str) or not re.fullmatch(r"\d+x\d+", g["resolution"]):
        raise AdapterError(f'GENERATION["{size}"]["resolution"] must be "WxH", got {g["resolution"]!r}')
    if g["generator"] not in GENERATORS:
        raise AdapterError(f'GENERATION["{size}"]["generator"] must be one of {GENERATORS}: the device the model card seeds its generator on')
    return g


def make(a: Any, size: str, prompt: str, seed: int) -> Generate:
    g = settings(a, size)
    w, h = (int(v) for v in g["resolution"].split("x"))
    return Generate(prompt=prompt, seed=seed, steps=g["steps"], guidance=float(g["guidance"]), width=w, height=h, generator=g["generator"])


def read_prompt(paths: list[Path]) -> str:
    """The timing prompt: one UTF-8 text file, stripped."""
    if len(paths) != 1:
        raise MeasureError("USAGE", "A text-to-image model takes one input file: the prompt, as UTF-8 text")
    text = paths[0].read_text(encoding="utf-8").strip()
    if not text:
        raise MeasureError("USAGE", f"{paths[0]} is empty")
    return text


def scheduler(a: Any, model: Any) -> str:
    """The pipeline's scheduler class, for the record: the adapter's SCHEDULER, or the class of `model.scheduler`."""
    if getattr(a, "SCHEDULER", None):
        return str(a.SCHEDULER)
    s = getattr(model, "scheduler", None)
    if s is None:
        from .adapter import AdapterError

        raise AdapterError("load() returned no pipeline with a .scheduler. Set SCHEDULER to the sampler's name")
    return type(s).__name__


def recorded(x: Generate, sched: str) -> dict:
    """The record's `generation` block (generationSettings in the schema)."""
    return {"steps": x.steps, "guidance": x.guidance, "seed": x.seed, "generator": x.generator, "scheduler": sched}


def check_image(out: Any, x: Generate) -> Any:
    """predict() must return one RGB PIL image at the requested resolution."""
    from PIL import Image

    from .adapter import AdapterError

    if not isinstance(out, Image.Image):
        raise AdapterError(f"predict() must return a PIL image for a text-to-image model, got {type(out).__name__}")
    if out.size != (x.width, x.height):
        raise AdapterError(f"predict() returned a {out.size[0]}x{out.size[1]} image; GENERATION asks for {x.resolution}")
    return out.convert("RGB")


def check_interface(a: Any) -> None:
    from .adapter import KNOWN_PRECISIONS, AdapterError

    missing = [n for n in ("TASK", "SIZES", "PRECISIONS", "GENERATION", "load", "predict", "weight_files") if not hasattr(a, n)]
    if not hasattr(a, "package_version") and not hasattr(a, "PACKAGE"):
        missing.append("PACKAGE (or package_version())")
    if missing:
        raise AdapterError(f"The adapter is missing {', '.join(missing)}")
    unknown = [p for p in a.PRECISIONS if p not in KNOWN_PRECISIONS]
    if unknown or not a.PRECISIONS:
        raise AdapterError(f"PRECISIONS must list some of {KNOWN_PRECISIONS}, got {a.PRECISIONS}")
    for size in a.SIZES:
        settings(a, size)
