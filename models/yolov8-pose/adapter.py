"""Adapter for Ultralytics YOLOv8-pose, real-time multi-person pose with COCO's 17 keypoints.

The docs load `YOLO("yolov8n-pose.pt")` and call the model on an image: imgsz 640, conf 0.25, letterboxed to
the smallest multiple of 32. quantize=16 is the official FP16 switch (it replaced the deprecated
half=True, which now logs a warning on every call). The model has one head, with NMS. verbose=False
only stops the per-image log line.
There is no keypoint benchmark in the tool yet, so there is no eval_predict().

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine.

Its ONNX Runtime row on the CPU is the docs' ONNX integration: `model.export(format="onnx")` with the export's defaults
(a static 640x640 input, onnxslim, NMS outside the graph), then `YOLO("yolov8n-pose.onnx")` called like the .pt model,
which runs ONNX Runtime with the CPU provider on a CPU (ultralytics/nn/backends/onnx.py). A static model is letterboxed
to the full 640x640.
"""

import os
from pathlib import Path

import torch

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Keypoints

WEIGHTS = weights_dir("yolov8-pose")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))
# A missing package fails the run instead of being installed into the environment.
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

from ultralytics.utils import SETTINGS  # noqa: E402

SETTINGS.update({"sync": False})

from ultralytics import YOLO  # noqa: E402

TASK = "pose-estimation"
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


def weight_files(size):
    return [WEIGHTS / f"yolov8{size}-pose.pt"]


def onnx_file(size, out):
    link = out / f"yolov8{size}-pose.pt"
    link.symlink_to(weight_files(size)[0])
    onnx = Path(YOLO(str(link)).export(**ONNX_EXPORT))
    return {"onnx": onnx, "exporter": f"{PACKAGE} YOLO.export", "args": [f"{k}={v}" for k, v in ONNX_EXPORT.items()]}


def load_onnx(size, onnx):
    return {"model": YOLO(str(onnx), task="pose")}


def onnx_session(m):
    # AutoBackend makes the session on the first call (ultralytics/nn/backends/onnx.py).
    return m["model"].predictor.model.session


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
