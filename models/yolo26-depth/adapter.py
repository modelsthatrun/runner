"""Adapter for Ultralytics YOLO26-depth, monocular metric depth.

The docs (docs/en/tasks/depth.md) load `YOLO("yolo26n-depth.pt")` and call the model on an image with no other
argument. The checkpoint's own arguments then set imgsz 768, the resolution the released weights were trained at,
and the depth predictor stretches the image to 768x768 instead of letterboxing it (DepthPredictor.scale_fill).
quantize=16 is the official FP16 switch. verbose=False only stops the per-image log line.

The head predicts a log-depth field, and a two-parameter scale fitted on the pretraining validation mix, stored in
the released checkpoints, turns it into meters. The predictor upsamples the map to the input image's size, so
`Results.depth` is (H, W) in meters.

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine. It also turns Ultralytics'
auto-install off (YOLO_AUTOINSTALL), so a missing package fails the run instead of changing the environment.

Its ONNX Runtime row on the CPU is the docs' export: `model.export(format="onnx")` with the export's defaults
(a static 768x768 input), then `YOLO("yolo26n-depth.onnx")` called like the .pt model, which runs ONNX Runtime with
the CPU provider on a CPU (ultralytics/nn/backends/onnx.py).
"""

import os
from pathlib import Path

import torch

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Depth

WEIGHTS = weights_dir("yolo26-depth")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update({"sync": False})

from ultralytics import YOLO  # noqa: E402

TASK = "depth-estimation"
SIZES = ["n", "s", "m", "l", "x"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "ultralytics"
IMGSZ = 768  # the checkpoints' own imgsz (train_args), which predict() uses when none is given
# Ultralytics names the first GPU 0 and the CPU "cpu".
DEVICE = 0 if device() == "cuda" else "cpu"
ONNXRUNTIME = {"status": "runs-as-is", "input": f"{IMGSZ}x{IMGSZ}"}
ONNX_EXPORT = {"format": "onnx", "imgsz": IMGSZ}


def input_size(size, image):
    # DepthPredictor stretches every image to imgsz x imgsz (scale_fill), with no padding.
    return f"{IMGSZ}x{IMGSZ}"


def load(size, precision):
    model = YOLO(str(weight_files(size)[0]))
    model.to(device())
    return {"model": model, "quantize": 16 if precision == "fp16" else 32}


def precision_args(m):
    # An ONNX file's precision was set when it was exported, and the docs call it with no precision argument.
    return {"quantize": m["quantize"]} if "quantize" in m else {}


@torch.inference_mode()
def predict(m, x):
    return m["model"].predict(x, imgsz=IMGSZ, device=DEVICE, verbose=False, **precision_args(m))[0]


def phase_times(m, out):
    # Ultralytics times its own phases per image, each closed by a synchronize (ultralytics/utils/ops.py, Profile).
    return {"preprocess": out.speed["preprocess"], "forward": out.speed["inference"], "postprocess": out.speed["postprocess"]}


def weight_files(size):
    return [WEIGHTS / f"yolo26{size}-depth.pt"]


def onnx_file(size, out):
    link = out / f"yolo26{size}-depth.pt"
    link.symlink_to(weight_files(size)[0])
    onnx = Path(YOLO(str(link)).export(**ONNX_EXPORT))
    return {"onnx": onnx, "exporter": f"{PACKAGE} YOLO.export", "args": [f"{k}={v}" for k, v in ONNX_EXPORT.items()]}


def load_onnx(size, onnx):
    return {"model": YOLO(str(onnx), task="depth")}


def onnx_session(m):
    # AutoBackend makes the session on the first call (ultralytics/nn/backends/onnx.py).
    return m["model"].predictor.model.session


def to_prediction(m, x, out):
    # Meters at the input image's size.
    return Depth(out.depth.data.float().cpu().numpy(), inverse=False)
