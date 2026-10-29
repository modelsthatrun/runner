"""Adapter for <model name>: how the shared tool loads and runs this model (docs/adapters.md).

`mtr check <id> --size <size>` tries it, and `mtr run <id> --size <size>` times it and writes
runs/<size>.json (docs/protocol.md). Timing needs only what is below; `mtr eval` and `mtr outputs`
also need to_prediction().
Follow the official README: its default input size, its loading code and its inference call.
See tools/measure/examples/ and the other models/ for working adapters.
"""

from pathlib import Path

import torch

# from mtr_measure import weights_dir   # for packages that download weights into the working directory
# from mtr_measure.predictions import Boxes, Masks   # for to_prediction()

TASK = "object-detection"  # the task it solves; picks the timing image (timing/timing.toml)
SIZES = ["medium"]  # checkpoint sizes, named as the model's own README names them
PRECISIONS = ["fp16", "fp32"]  # what the official code supports; the tool uses fp16 if listed, else fp32
PACKAGE = "package-name"  # the installed distribution that runs the model; its version is recorded
INPUT = {"medium": "640x640"}  # the official default input resolution per size, WxH


def load(size: str, precision: str):
    """Load the checkpoint onto the GPU at this precision and return whatever predict() needs."""
    raise NotImplementedError


@torch.inference_mode()
def predict(model, x):
    """One call from input to output, as a user's code would make it: pre-processing, the forward
    pass and post-processing. Return the output as the official API gives it (usually on the CPU).
    `x` is the input file decoded as an RGB PIL image, unless prepare_input() says otherwise."""
    raise NotImplementedError


def weight_files(size: str) -> list[Path]:
    """The files load() read, main checkpoint first. Records carry their sha256, which must match the published checkpoint."""
    raise NotImplementedError


def to_prediction(model, x, out):
    """For `mtr eval`, `mtr outputs` and `check`: predict()'s output `out` as the task's standard prediction type,
    such as Boxes for detection or Masks for segmentation (mtr_measure/predictions.py).
    Coordinates and masks are in pixels of the input image. Labels are class names, not ids.
    Return every score: the tool applies the task's fixed threshold (docs/protocol.md).
    e.g. return Boxes(xyxy=..., scores=..., labels=[names[i] for i in class_ids])"""
    raise NotImplementedError


# Optional:
# def prepare_input(paths: list[Path]): ...     # several input files, or an input that isn't one image
# def input_size(size: str, x) -> str: ...      # instead of INPUT, when the resolution depends on the input
# def package_version() -> str: ...             # instead of PACKAGE, e.g. "owner/repo@<commit>" for a git checkout
# LATENCY_SCOPE = "..."                         # if one predict() call covers something other than the default
