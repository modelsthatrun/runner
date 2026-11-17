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

Its ONNX Runtime row on the CPU follows demo/ONNXRuntime/README.md: tools/export_onnx.py on the released
checkpoint (`--output-name yolox_s.onnx -f exps/default/yolox_s.py -c yolox_s.pth`), then
demo/ONNXRuntime/onnx_inference.py, which makes a default InferenceSession, letterboxes the image as
the PyTorch demo does, decodes the grid with demo_postprocess() and applies multiclass_nms() at
nms_thr 0.45 and score_thr 0.1.
"""

import importlib.util
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
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
# onnx_inference.py letterboxes to the same size as the PyTorch demo.
ONNXRUNTIME = {"status": "runs-as-is"}


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
    if "sess" in m:
        return ort_call(m, x, 0.1)
    return postprocess(m, forward(m, preprocess(m, x)), x)


@torch.inference_mode()
def eval_predict(m, x):
    return decode(m, x, forward(m, preprocess(m, x)), EVAL_CONF, EVAL_NMS, class_agnostic=False)


def onnx_file(size, out):
    """demo/ONNXRuntime/README.md's export, `python3 tools/export_onnx.py --output-name yolox_s.onnx -f
    exps/default/yolox_s.py -c yolox_s.pth`, run as it is on the released checkpoint, writing into `out`."""
    checkout()
    ckpt = download(size)
    out = Path(out)
    name = f"yolox_{size}.onnx"
    subprocess.run([sys.executable, str(REPO / "tools" / "export_onnx.py"), "--output-name", name, "-f", str(exp_file(size)),
                    "-c", str(ckpt)], cwd=out, check=True, env={**os.environ, "PYTHONPATH": str(REPO)})
    args = ["--output-name", name, "-f", f"exps/default/yolox_{size}.py", "-c", ckpt.name]
    return {"onnx": out / name, "exporter": f"Megvii-BaseDetection/YOLOX@{COMMIT} tools/export_onnx.py", "args": args}


def onnx_inference():
    checkout()
    spec = importlib.util.spec_from_file_location("yolox_onnx_inference", REPO / "demo" / "ONNXRuntime" / "onnx_inference.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_onnx(size, onnx):
    """The ONNX file in a session as onnx_inference.py makes it: onnxruntime.InferenceSession(args.model)."""
    checkout()
    from yolox.data.datasets import COCO_CLASSES
    from yolox.exp import get_exp

    module = onnx_inference()
    exp = get_exp(str(exp_file(size)), None)
    return {"sess": module.onnxruntime.InferenceSession(str(onnx)), "inf": module, "size": exp.test_size, "names": list(COCO_CLASSES)}


def onnx_session(m):
    return m["sess"]


def ort_call(m, x, score_thr):
    # onnx_inference.py's main(), step for step, with score_thr as given (0.1 in the demo).
    inf = m["inf"]
    img, ratio = inf.preprocess(x, m["size"])
    output = m["sess"].run(None, {m["sess"].get_inputs()[0].name: img[None, :, :, :]})
    predictions = inf.demo_postprocess(output[0], m["size"])[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio
    dets = inf.multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=score_thr)
    if dets is None:
        return torch.zeros(0, dtype=torch.int64), torch.zeros(0, 4), torch.zeros(0)
    return torch.from_numpy(dets[:, 5]).long(), torch.from_numpy(dets[:, :4]).float(), torch.from_numpy(dets[:, 4]).float()


def weight_files(size):
    return [WEIGHTS / f"yolox_{size}.pth"]


def to_prediction(m, x, out):
    labels, boxes, scores = out
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
