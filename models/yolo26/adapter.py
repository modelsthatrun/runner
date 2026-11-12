"""Adapter for Ultralytics YOLO26, real-time object detection on COCO's 80 classes.

The README loads `YOLO("yolo26n.pt")` and calls the model on an image: imgsz 640, conf 0.25,
letterboxed to the smallest multiple of 32. quantize=16 is the official FP16 switch (it replaced
the deprecated half=True, which now logs a warning on every call). Prediction and
validation default to the one-to-many head with NMS (the docs' "mAPval 50-95" column); the NMS-free
head (nms=False) is what the docs' speed column and "(e2e)" column use. verbose=False only stops
the per-image log line.

Ultralytics sends usage events when its `sync` setting is on. The adapter keeps its settings in the
model's weights directory and turns sync off, so nothing leaves this machine. It also turns Ultralytics'
auto-install off (YOLO_AUTOINSTALL), so a missing package fails the run instead of changing the environment.

The TensorRT row is the docs' TensorRT integration: `model.export(format="engine")` with quantize=16, then
`YOLO("yolo26n.engine")` called like the .pt model. The export writes the ONNX file and builds the engine itself with
TensorRT's Python API (ultralytics/utils/export/engine.py, onnx2engine): the FP16 flag, a static 640x640 input,
TensorRT's own workspace. nms stays at its default, so the engine holds the one-to-many head and NMS runs outside it,
as predict() and the docs' mAP column run it. A static engine is letterboxed to the full 640x640.

Its ONNX Runtime row on the CPU is the docs' ONNX integration: `model.export(format="onnx")` with the export's defaults
(a static 640x640 input, onnxslim, NMS outside the graph), then `YOLO("yolo26n.onnx")` called like the .pt model,
which runs ONNX Runtime with the CPU provider on a CPU (ultralytics/nn/backends/onnx.py). A static model is letterboxed
to the full 640x640.
"""

import math
import os
from pathlib import Path

import torch

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Boxes

WEIGHTS = weights_dir("yolo26")
os.environ.setdefault("YOLO_CONFIG_DIR", str(WEIGHTS / "config"))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")

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
TENSORRT = {"precision": "fp16", "builder": {"fp16": True}, "status": "runs-as-is", "input": f"{IMGSZ}x{IMGSZ}"}
ENGINE_EXPORT = {"format": "engine", "quantize": 16, "imgsz": IMGSZ, "device": 0}
# Ultralytics names the first GPU 0 and the CPU "cpu".
DEVICE = 0 if device() == "cuda" else "cpu"
ONNXRUNTIME = {"status": "runs-as-is", "input": f"{IMGSZ}x{IMGSZ}"}
ONNX_EXPORT = {"format": "onnx", "imgsz": IMGSZ}


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


def export_engine(size, out):
    link = out / f"yolo26{size}.pt"
    link.symlink_to(weight_files(size)[0])
    engine = Path(YOLO(str(link)).export(**ENGINE_EXPORT))
    return {"engine": engine, "onnx": engine.with_suffix(".onnx"), "exporter": f"{PACKAGE} YOLO.export",
            "args": [f"{k}={v}" for k, v in ENGINE_EXPORT.items()],
            "builder": {"input": f"1x3x{IMGSZ}x{IMGSZ}, static", "workspace": "TensorRT's default"}}


def load_engine(size, engine):
    # The engine file carries Ultralytics' metadata (names, stride, image size) ahead of TensorRT's plan.
    return {"model": YOLO(str(engine), task="detect"), "engine": True}


def engine_device(m):
    # A TensorRT engine runs on the GPU; the .pt model and the ONNX file on the tool's device.
    return 0 if m.get("engine") else DEVICE


def precision_args(m):
    # An engine's or an ONNX file's precision was set when it was made, and the docs call it with no precision argument.
    return {"quantize": m["quantize"]} if "quantize" in m else {}


@torch.inference_mode()
def predict(m, x):
    return m["model"].predict(x, imgsz=IMGSZ, device=engine_device(m), verbose=False, **precision_args(m))[0]


def phase_times(m, out):
    # Ultralytics times its own phases per image, each closed by a synchronize (ultralytics/utils/ops.py, Profile).
    return {"preprocess": out.speed["preprocess"], "forward": out.speed["inference"], "postprocess": out.speed["postprocess"]}


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
    return m["model"].predict(x, imgsz=IMGSZ, device=engine_device(m), conf=0.001, max_det=300, verbose=False,
                              predictor=ValNMSPredictor, **precision_args(m))[0]


def weight_files(size):
    return [WEIGHTS / f"yolo26{size}.pt"]


def onnx_file(size, out):
    link = out / f"yolo26{size}.pt"
    link.symlink_to(weight_files(size)[0])
    onnx = Path(YOLO(str(link)).export(**ONNX_EXPORT))
    return {"onnx": onnx, "exporter": f"{PACKAGE} YOLO.export", "args": [f"{k}={v}" for k, v in ONNX_EXPORT.items()]}


def load_onnx(size, onnx):
    return {"model": YOLO(str(onnx), task="detect")}


def onnx_session(m):
    # AutoBackend makes the session on the first call (ultralytics/nn/backends/onnx.py).
    return m["model"].predictor.model.session


def to_prediction(m, x, out):
    b = out.boxes
    names = out.names
    # At conf 0.001 a box can come back with a negative width or height (one on COCO image 143068
    # for x, scoring 0.0014, in fp16 and fp32 alike). Collapse it to zero area: it still counts as
    # a false positive in COCO's evaluation, and never as a match.
    xyxy = b.xyxy.cpu().clone()
    xyxy[:, 2:] = xyxy[:, 2:].maximum(xyxy[:, :2])
    return Boxes(xyxy.numpy(), b.conf.cpu().numpy(), [names[int(c)] for c in b.cls.tolist()])
