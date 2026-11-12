"""Adapter for Ultralytics YOLO26-seg, real-time instance segmentation on COCO's 80 classes.

The docs load `YOLO("yolo26n-seg.pt")` and call the model on an image: imgsz 640, conf 0.25,
letterboxed to the smallest multiple of 32. quantize=16 is the official FP16 switch (it replaced
the deprecated half=True, which now logs a warning on every call). Prediction and
validation default to the one-to-many head with NMS (the docs' "mAPval 50-95" column); the NMS-free
head (nms=False) is what the docs' speed column and "(e2e)" column use. verbose=False only stops
the per-image log line. predict() returns masks at the letterboxed input size, as Ultralytics
gives them; to_prediction() scales them to the image the way its validator does.

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine.

Its ONNX Runtime row on the CPU is the docs' ONNX integration: `model.export(format="onnx")` with the export's defaults
(a static 640x640 input, onnxslim, NMS outside the graph), then `YOLO("yolo26n-seg.onnx")` called like the .pt model,
which runs ONNX Runtime with the CPU provider on a CPU (ultralytics/nn/backends/onnx.py). A static model is letterboxed
to the full 640x640.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Masks

WEIGHTS = weights_dir("yolo26-seg")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))
# A missing package fails the run instead of being installed into the environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update({"sync": False})

from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import ops  # noqa: E402

TASK = "instance-segmentation"
SIZES = ["n", "s", "m", "l", "x"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "ultralytics"
IMGSZ = 640
# Ultralytics names the first GPU 0 and the CPU "cpu".
DEVICE = 0 if device() == "cuda" else "cpu"
ONNXRUNTIME = {"status": "runs-as-is", "input": f"{IMGSZ}x{IMGSZ}"}
ONNX_EXPORT = {"format": "onnx", "imgsz": IMGSZ}
STRIDE = 32


def input_size(size, image):
    # LetterBox(auto=True): the longer side to 640, the shorter padded up to a multiple of 32.
    w, h = image.size
    r = min(IMGSZ / w, IMGSZ / h)
    nw, nh = round(w * r), round(h * r)
    return f"{nw + (IMGSZ - nw) % STRIDE}x{nh + (IMGSZ - nh) % STRIDE}"


def load(size, precision):
    model = YOLO(str(weight_files(size)[0]))
    model.to(device())
    return {"model": model, "quantize": 16 if precision == "fp16" else 32}


def precision_args(m):
    # An ONNX file's precision was set when it was exported, and the docs call it with no precision argument.
    return {"quantize": m["quantize"]} if "quantize" in m else {}


@torch.inference_mode()
def predict(m, x):
    return m["model"].predict(x, imgsz=IMGSZ, device=DEVICE, **precision_args(m), verbose=False)[0]


def phase_times(m, out):
    # Ultralytics times its own phases per image, each closed by a synchronize (ultralytics/utils/ops.py, Profile).
    return {"preprocess": out.speed["preprocess"], "forward": out.speed["inference"], "postprocess": out.speed["postprocess"]}


@torch.inference_mode()
def eval_predict(m, x):
    # The docs' mask AP comes from `yolo segment val data=coco.yaml nms=False`: the NMS-free head,
    # conf 0.001 and up to 300 detections.
    return m["model"].predict(x, imgsz=IMGSZ, device=DEVICE, **precision_args(m), conf=0.001, max_det=300, nms=False,
                              verbose=False)[0]


def weight_files(size):
    return [WEIGHTS / f"yolo26{size}-seg.pt"]


def onnx_file(size, out):
    link = out / f"yolo26{size}-seg.pt"
    link.symlink_to(weight_files(size)[0])
    onnx = Path(YOLO(str(link)).export(**ONNX_EXPORT))
    return {"onnx": onnx, "exporter": f"{PACKAGE} YOLO.export", "args": [f"{k}={v}" for k, v in ONNX_EXPORT.items()]}


def load_onnx(size, onnx):
    return {"model": YOLO(str(onnx), task="segment")}


def onnx_session(m):
    # AutoBackend makes the session on the first call (ultralytics/nn/backends/onnx.py).
    return m["model"].predictor.model.session


def to_prediction(m, x, out):
    names = out.names
    labels = [names[int(c)] for c in out.boxes.cls.tolist()]
    if out.masks is None:
        return Masks(np.zeros((0, x.size[1], x.size[0]), bool), [], np.zeros(0))
    # SegmentationValidator.scale_preds: undo the letterbox, then threshold.
    masks = ops.scale_masks(out.masks.data[None].float(), out.orig_shape)[0] > 0.5
    return Masks(masks.cpu().numpy(), labels, out.boxes.conf.cpu().numpy())
