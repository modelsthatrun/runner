"""Adapter for Ultralytics YOLO26, real-time object detection on COCO's 80 classes.

The README loads `YOLO("yolo26n.pt")` and calls the model on an image: imgsz 640, conf 0.25,
letterboxed to the smallest multiple of 32. quantize=16 is the official FP16 switch (it replaced
the deprecated half=True, which now logs a warning on every call). Prediction and
validation default to the one-to-many head with NMS (the docs' "mAPval 50-95" column); the NMS-free
head (nms=False) is what the docs' speed column and "(e2e)" column use. verbose=False only stops
the per-image log line.

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine.
"""

import math
import os
from pathlib import Path

import torch

from mtr_measure import weights_dir
from mtr_measure.predictions import Boxes

WEIGHTS = weights_dir("yolo26")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))

from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update({"sync": False})

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.detect import DetectionPredictor  # noqa: E402
from ultralytics.utils import nms, ops  # noqa: E402

TASK = "object-detection"
SIZES = ["n", "s", "m", "l", "x"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "ultralytics"
IMGSZ = 640
STRIDE = 32


def input_size(size, image):
    # LetterBox(auto=True): the longer side to 640, the shorter padded up to a multiple of 32.
    w, h = image.size
    r = min(IMGSZ / w, IMGSZ / h)
    nw, nh = round(w * r), round(h * r)
    return f"{nw + (IMGSZ - nw) % STRIDE}x{nh + (IMGSZ - nh) % STRIDE}"


def load(size, precision):
    model = YOLO(str(weight_files(size)[0]))
    model.to("cuda")
    return {"model": model, "quantize": 16 if precision == "fp16" else 32}


@torch.inference_mode()
def predict(m, x):
    return m["model"].predict(x, imgsz=IMGSZ, quantize=m["quantize"], device=0, verbose=False)[0]


class ValNMSPredictor(DetectionPredictor):
    """DetectionPredictor with DetectionValidator's NMS, which keeps every class above conf for a box
    (multi_label=True). Prediction keeps only the best class."""

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        preds = nms.non_max_suppression(
            preds, self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms,
            max_det=self.args.max_det, nc=0, multi_label=True, end2end=getattr(self.model, "end2end", False))
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(preds, img, orig_imgs, **kwargs)


@torch.inference_mode()
def eval_predict(m, x):
    # model.val() on COCO: conf 0.001, up to 300 detections, multi-label NMS (DetectionValidator.postprocess).
    return m["model"].predict(x, imgsz=IMGSZ, quantize=m["quantize"], device=0, conf=0.001, max_det=300, verbose=False,
                              predictor=ValNMSPredictor)[0]


def weight_files(size):
    return [WEIGHTS / f"yolo26{size}.pt"]


def to_prediction(m, x, out):
    b = out.boxes
    names = out.names
    # At conf 0.001 a box can come back with a negative width or height (one on COCO image 143068
    # for x, scoring 0.0014, in fp16 and fp32 alike). Collapse it to zero area: it still counts as
    # a false positive in COCO's evaluation, and never as a match.
    xyxy = b.xyxy.cpu().clone()
    xyxy[:, 2:] = xyxy[:, 2:].maximum(xyxy[:, :2])
    return Boxes(xyxy.numpy(), b.conf.cpu().numpy(), [names[int(c)] for c in b.cls.tolist()])
