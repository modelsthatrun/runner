"""Adapter for YOLOX, real-time object detection on COCO's 80 classes.

The README installs YOLOX from source; the adapter runs a pinned checkout of the official repository,
the way the README's demo command runs it: `python tools/demo.py image -f exps/default/yolox_s.py -c
yolox_s.pth --conf 0.25 --nms 0.45 --tsize 640 --device gpu`. tools/demo.py's Predictor reads the image
with cv2 (BGR), letterboxes it into the experiment's test size (top-left, padded with 114, no
normalization: ValTransform with legacy=False), runs the model with its decoding head, keeps boxes
whose objectness times class score passes --conf, and applies class-agnostic NMS at --nms. Its
--fp16 flag halves the model and the input, so PRECISIONS lists fp16. The test size is 640 for s–x
and 416 for nano and tiny, as the README's tables and the experiments give it. The checkpoints are
the README's 0.1.1rc0 release links (the current, non-legacy weights).

COCO evaluation follows the README's `python -m yolox.tools.eval -n yolox-s -c yolox_s.pth --conf 0.001`:
the same pre-processing, confidence 0.001, the experiment's NMS threshold of 0.65, per class.

There is no ONNX Runtime row yet. The README's export, tools/export_onnx.py, calls torch.onnx._export,
which the pinned torch no longer has, so the official path doesn't run here without a fix.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

import torch

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Boxes

REPO_URL = "https://github.com/Megvii-BaseDetection/YOLOX.git"
COMMIT = "6ddff4824372906469a7fae2dc3206c7aa4bbaee"
WEIGHTS = weights_dir("yolox")
REPO = WEIGHTS / "YOLOX"
RELEASE = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0"

TASK = "object-detection"
SIZES = ["nano", "tiny", "s", "m", "l", "x"]
PRECISIONS = ["fp16", "fp32"]
INPUT = {s: ("416x416" if s in ("nano", "tiny") else "640x640") for s in SIZES}
# The README's demo command: --conf 0.25 --nms 0.45.
CONF = 0.25
NMS = 0.45
# The README's COCO evaluation: --conf 0.001, with the experiment's nmsthre.
EVAL_CONF = 0.001
EVAL_NMS = 0.65


def package_version():
    return f"Megvii-BaseDetection/YOLOX@{COMMIT}"


def checkout():
    if not (REPO / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def exp_file(size):
    return REPO / "exps" / "default" / f"yolox_{size}.py"


def download(size):
    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        part = ckpt.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE}/{ckpt.name}", part)
        part.rename(ckpt)
    return ckpt


def prepare_input(paths):
    """tools/demo.py's Predictor.inference() reads the image with cv2.imread(), so the model gets a BGR array."""
    import cv2

    if len(paths) != 1:
        raise ValueError("YOLOX takes one image")
    img = cv2.imread(str(paths[0]))
    if img is None:
        raise ValueError(f"cv2 could not read {paths[0]}")
    return img


def load(size, precision):
    checkout()
    from yolox.data.data_augment import ValTransform
    from yolox.data.datasets import COCO_CLASSES
    from yolox.exp import get_exp

    exp = get_exp(str(exp_file(size)), None)
    model = exp.get_model()
    model.to(device())
    if precision == "fp16":
        model.half()
    model.eval()
    ckpt = torch.load(download(size), map_location="cpu")
    model.load_state_dict(ckpt["model"])
    return {"model": model, "exp": exp, "preproc": ValTransform(legacy=False), "half": precision == "fp16",
            "size": exp.test_size, "names": list(COCO_CLASSES)}


# predict() in the three phases `mtr profile` times apart, as Predictor.inference() runs them.
@torch.inference_mode()
def preprocess(m, x):
    img, _ = m["preproc"](x, None, m["size"])
    img = torch.from_numpy(img).unsqueeze(0).float().to(device())
    return img.half() if m["half"] else img


@torch.inference_mode()
def forward(m, img):
    return m["model"](img)


def decode(m, x, raw, conf, nms, class_agnostic):
    from yolox.utils import postprocess as yolox_postprocess

    out = yolox_postprocess(raw, m["exp"].num_classes, conf, nms, class_agnostic=class_agnostic)[0]
    if out is None:
        return torch.zeros(0, dtype=torch.int64), torch.zeros(0, 4), torch.zeros(0)
    out = out.cpu().float()
    ratio = min(m["size"][0] / x.shape[0], m["size"][1] / x.shape[1])
    # Predictor.visual(): boxes back to the photo, score = objectness * class confidence.
    return out[:, 6].long(), out[:, 0:4] / ratio, out[:, 4] * out[:, 5]


@torch.inference_mode()
def postprocess(m, raw, x):
    return decode(m, x, raw, CONF, NMS, class_agnostic=True)


@torch.inference_mode()
def predict(m, x):
    return postprocess(m, forward(m, preprocess(m, x)), x)


@torch.inference_mode()
def eval_predict(m, x):
    return decode(m, x, forward(m, preprocess(m, x)), EVAL_CONF, EVAL_NMS, class_agnostic=False)


def weight_files(size):
    return [WEIGHTS / f"yolox_{size}.pth"]


def to_prediction(m, x, out):
    labels, boxes, scores = out
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
