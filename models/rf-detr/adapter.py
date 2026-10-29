"""Adapter for RF-DETR (Roboflow), real-time object detection on COCO's 80 classes.

The README loads a size class and calls predict(image, threshold=0.5). Unoptimized, the package
warns that latency "may be higher than expected" and says to call model.inference(dtype=torch.float16)
for FP16, so load() does that, with its defaults (a TorchScript trace at the size's resolution).
Only the Apache-2.0 sizes are listed: XL and 2XL need the rfdetr_plus extension under PML 1.0.
"""

import os
from pathlib import Path

import torch

from mtr_measure import weights_dir
from mtr_measure.predictions import Boxes

# RF-DETR downloads weights into RF_HOME (default ~/.roboflow/models).
os.environ.setdefault("RF_HOME", str(weights_dir("rf-detr")))

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall  # noqa: E402
from rfdetr.assets.model_weights import get_model_cache_dir  # noqa: E402

CLASSES = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}

TASK = "object-detection"
SIZES = list(CLASSES)
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "rfdetr"
INPUT = {"nano": "384x384", "small": "512x512", "medium": "576x576", "large": "704x704"}
FILES = {"nano": "rf-detr-nano.pth", "small": "rf-detr-small.pth", "medium": "rf-detr-medium.pth", "large": "rf-detr-large-2026.pth"}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size, precision):
    model = CLASSES[size]()
    model.inference(dtype=DTYPES[precision])
    return model


def predict(model, x):
    # The README's call and display threshold.
    return model.predict(x, threshold=0.5)


def eval_predict(model, x):
    # rfdetr's own COCO evaluation scores the top num_select detections with no score threshold.
    return model.predict(x, threshold=0.0)


def weight_files(size):
    # The pretrain_weights default of each size's config (rfdetr/config.py).
    return [Path(get_model_cache_dir()) / FILES[size]]


def to_prediction(model, x, out):
    # predict() resolves COCO category ids to names in data["class_name"]; boxes are in input pixels.
    # The ten ids COCO leaves unused come back with an empty name, which COCO's evaluation ignores.
    return Boxes(out.xyxy, out.confidence, [name or "N/A" for name in out.data["class_name"]])
