"""Example adapter: torchvision Faster R-CNN with a MobileNetV3-Large FPN, object detection.

Not a catalog model. It declares fp32 only, so it also shows the precision rule's fallback, and
its input resolution depends on the input image. to_prediction() gives the boxes to `mtr outputs`.
"""

import math
from pathlib import Path

import torch
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, fasterrcnn_mobilenet_v3_large_fpn

from mtr_measure.predictions import Boxes

WEIGHTS = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1

TASK = "object-detection"
SIZES = ["default"]
PRECISIONS = ["fp32"]
PACKAGE = "torchvision"


def input_size(size, image):
    # GeneralizedRCNNTransform: shorter side to 800, longer side at most 1333.
    w, h = image.size
    scale = min(800 / min(w, h), 1333 / max(w, h))
    return f"{math.floor(w * scale)}x{math.floor(h * scale)}"


def load(size, precision):
    net = fasterrcnn_mobilenet_v3_large_fpn(weights=WEIGHTS).eval().to("cuda")
    return {"net": net, "preprocess": WEIGHTS.transforms()}


@torch.inference_mode()
def predict(m, image):
    x = m["preprocess"](image).to("cuda")
    out = m["net"]([x])[0]
    return {k: v.cpu() for k, v in out.items()}


def weight_files(size):
    return [Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name]


def to_prediction(m, image, out):
    # Boxes come back in the input image's pixels, with every score above the model's own 0.05.
    names = WEIGHTS.meta["categories"]
    return Boxes(out["boxes"].numpy(), out["scores"].numpy(), [names[i] for i in out["labels"].tolist()])
