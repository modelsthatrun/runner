"""Adapter for RF-DETR-Seg (Roboflow), real-time instance segmentation on COCO's 80 classes.

The same package and call as models/rf-detr: the README loads a size class and calls
predict(image, threshold=0.5), and the package says to call model.inference(dtype=torch.float16)
for FP16. Every segmentation size, nano to 2XL, is Apache-2.0.
"""

import os
from pathlib import Path

import torch

from mtr_measure import weights_dir
from mtr_measure.predictions import Masks

# RF-DETR downloads weights into RF_HOME (default ~/.roboflow/models).
os.environ.setdefault("RF_HOME", str(weights_dir("rf-detr-seg")))

from rfdetr import (  # noqa: E402
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
)
from rfdetr.assets.model_weights import get_model_cache_dir  # noqa: E402

CLASSES = {
    "nano": RFDETRSegNano, "small": RFDETRSegSmall, "medium": RFDETRSegMedium,
    "large": RFDETRSegLarge, "xlarge": RFDETRSegXLarge, "2xlarge": RFDETRSeg2XLarge,
}

TASK = "instance-segmentation"
SIZES = list(CLASSES)
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "rfdetr"
INPUT = {"nano": "312x312", "small": "384x384", "medium": "432x432", "large": "504x504", "xlarge": "624x624", "2xlarge": "768x768"}
# The pretrain_weights default of each size's config (rfdetr/config.py).
FILES = {
    "nano": "rf-detr-seg-nano.pt", "small": "rf-detr-seg-small.pt", "medium": "rf-detr-seg-medium.pt",
    "large": "rf-detr-seg-large.pt", "xlarge": "rf-detr-seg-xlarge.pt", "2xlarge": "rf-detr-seg-xxlarge.pt",
}
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
    return [Path(get_model_cache_dir()) / FILES[size]]


def to_prediction(model, x, out):
    # Masks come back at the input image's size. The ids COCO leaves unused have an empty name,
    # which COCO's evaluation ignores.
    return Masks(out.mask, [name or "N/A" for name in out.data["class_name"]], out.confidence)
