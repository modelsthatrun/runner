"""Adapter for Ultralytics YOLO26-pose, real-time multi-person pose with COCO's 17 keypoints.

The docs load `YOLO("yolo26n-pose.pt")` and call the model on an image: imgsz 640, conf 0.25,
letterboxed to the smallest multiple of 32. quantize=16 is the official FP16 switch (it replaced
the deprecated half=True, which now logs a warning on every call). Prediction defaults to the
one-to-many head with NMS; the docs' speed column uses the NMS-free head (nms=False). verbose=False
only stops the per-image log line. There is no keypoint benchmark in the tool yet, so there is no
eval_predict().

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine.
"""

import os
from pathlib import Path

import torch

from mtr_measure import weights_dir
from mtr_measure.predictions import Keypoints

WEIGHTS = weights_dir("yolo26-pose")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))

from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update({"sync": False})

from ultralytics import YOLO  # noqa: E402

TASK = "pose-estimation"
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


def weight_files(size):
    return [WEIGHTS / f"yolo26{size}-pose.pt"]


# COCO's keypoint names and skeleton (person_keypoints_val2017.json), 0-based.
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def to_prediction(m, x, out):
    # Keypoints come back in the image's pixels, one row per person, scored by the box confidence.
    # Ultralytics' own drawing (Annotator.kpts) hides keypoints whose confidence is under 0.25.
    visible = out.keypoints.conf.cpu().numpy() >= 0.25
    return Keypoints(out.keypoints.xy.cpu().numpy(), out.boxes.conf.cpu().numpy(), NAMES, SKELETON, visible)
