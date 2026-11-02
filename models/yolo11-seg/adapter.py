"""Adapter for Ultralytics YOLO11-seg, real-time instance segmentation on COCO's 80 classes.

The docs load `YOLO("yolo11n-seg.pt")` and call the model on an image: imgsz 640, conf 0.25, letterboxed to
the smallest multiple of 32. quantize=16 is the official FP16 switch (it replaced the deprecated
half=True, which now logs a warning on every call). The model has one head, with NMS. verbose=False
only stops the per-image log line.
predict() returns masks at the letterboxed input size, as Ultralytics gives them; to_prediction()
scales them to the image the way its validator does.

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine.
"""

import math
import os
from pathlib import Path

import numpy as np
import torch

from mtr_measure import weights_dir
from mtr_measure.predictions import Masks

WEIGHTS = weights_dir("yolo11-seg")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))

from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update({"sync": False})

from ultralytics import YOLO  # noqa: E402
from ultralytics.models.yolo.segment import SegmentationPredictor  # noqa: E402
from ultralytics.utils import nms, ops  # noqa: E402

TASK = "instance-segmentation"
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


class ValNMSPredictor(SegmentationPredictor):
    """SegmentationPredictor with DetectionValidator's NMS, which keeps every class above conf for a box
    (multi_label=True). Prediction keeps only the best class."""

    def postprocess(self, preds, img, orig_imgs):
        protos = preds[0][1] if isinstance(preds[0], tuple) else preds[1]
        results = nms.non_max_suppression(
            preds[0], self.args.conf, self.args.iou, self.args.classes, self.args.agnostic_nms,
            max_det=self.args.max_det, nc=len(self.model.names), multi_label=True, end2end=False)
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(results, img, orig_imgs, protos=protos)


@torch.inference_mode()
def eval_predict(m, x):
    # model.val() on COCO: conf 0.001, up to 300 detections, multi-label NMS (DetectionValidator.postprocess).
    return m["model"].predict(x, imgsz=IMGSZ, quantize=m["quantize"], device=0, conf=0.001, max_det=300, verbose=False,
                              predictor=ValNMSPredictor)[0]


def weight_files(size):
    return [WEIGHTS / f"yolo11{size}-seg.pt"]


def to_prediction(m, x, out):
    names = out.names
    labels = [names[int(c)] for c in out.boxes.cls.tolist()]
    if out.masks is None:
        return Masks(np.zeros((0, x.size[1], x.size[0]), bool), [], np.zeros(0))
    # SegmentationValidator.scale_preds: undo the letterbox, then threshold.
    masks = ops.scale_masks(out.masks.data[None].float(), out.orig_shape)[0] > 0.5
    return Masks(masks.cpu().numpy(), labels, out.boxes.conf.cpu().numpy())
