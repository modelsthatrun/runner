"""Adapter for RF-DETR Keypoint (Preview), multi-person pose estimation with COCO's 17 keypoints.

The same package and call as models/rf-detr: the README loads RFDETRKeypointPreview and calls
predict(image, threshold=0.5), and the package says to call model.inference(dtype=torch.float16)
for FP16. It is Apache-2.0 like the package (README); PML 1.0 covers only the rfdetr_plus models.
"""

import os
from pathlib import Path

import numpy as np
import torch

from mtr_measure import weights_dir
from mtr_measure.predictions import Keypoints

# RF-DETR downloads weights into RF_HOME (default ~/.roboflow/models).
os.environ.setdefault("RF_HOME", str(weights_dir("rf-detr-keypoint")))

from rfdetr import RFDETRKeypointPreview  # noqa: E402
from rfdetr.assets.model_weights import get_model_cache_dir  # noqa: E402

TASK = "pose-estimation"
SIZES = ["preview"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "rfdetr"
INPUT = {"preview": "576x576"}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
# COCO's keypoint names and skeleton (person_keypoints_val2017.json), 0-based.
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def load(size, precision):
    model = RFDETRKeypointPreview()
    model.inference(dtype=DTYPES[precision])
    return model


def predict(model, x):
    # The README's call and display threshold.
    return model.predict(x, threshold=0.5)


def weight_files(size):
    # The pretrain_weights default of RFDETRKeypointPreviewConfig (rfdetr/config.py).
    return [Path(get_model_cache_dir()) / "rf-detr-keypoint-preview-xlarge.pth"]


def to_prediction(model, x, out):
    # One row of 17 keypoints per person, in input pixels, scored by the detection confidence.
    points = np.asarray(out.xy, dtype=np.float32).reshape(-1, len(NAMES), 2)
    return Keypoints(points, np.asarray(out.detection_confidence, dtype=np.float32).reshape(-1), NAMES, SKELETON)
