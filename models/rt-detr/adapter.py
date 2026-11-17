"""Adapter for RT-DETR, real-time object detection on COCO's 80 classes.

RT-DETR has no package. The adapter runs a pinned checkout of the official repository's
rtdetr_pytorch/, the way its tools/infer.py does without slicing: the config's model and
postprocessor in deploy mode, the image resized to 640x640 with torchvision, and the top 300 boxes
back in the input image's pixels. That script runs in fp32 (it uses autocast only for sliced
inference, which is off by default), so PRECISIONS is fp32. Sizes are named by backbone, as the
configs are. Only the COCO-trained checkpoints of the README's model zoo with GitHub release links
are listed. The README says they were converted from the PaddlePaddle models, and "may have slight
differences between this table and paper". RT-DETRv2, from the same repository, is its own family.

The configs build PResNet with `pretrained: True`, which downloads ImageNet weights that the
checkpoint then replaces. The adapter turns that off, so the model is the same and nothing extra is
downloaded.

rtdetr_pytorch/ has an ONNX export but no ONNX Runtime inference code (its README points to an issue),
so there is no ONNX Runtime row.
"""

import subprocess
import sys
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Boxes

REPO_URL = "https://github.com/lyuwenyu/RT-DETR.git"
COMMIT = "df2a96c8c3a098567a1149bf0248da6efe6d495d"
WEIGHTS = weights_dir("rt-detr")
CHECKOUT = WEIGHTS / "RT-DETR"
REPO = CHECKOUT / "rtdetr_pytorch"
RELEASE = "https://github.com/lyuwenyu/storage/releases/download"
# The README's model zoo: config and checkpoint per size.
CKPT = {
    "r18": ("rtdetr_r18vd_6x_coco.yml", "v0.1/rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
    "r34": ("rtdetr_r34vd_6x_coco.yml", "v0.1/rtdetr_r34vd_dec4_6x_coco_from_paddle.pth"),
    "r50-m": ("rtdetr_r50vd_m_6x_coco.yml", "v0.1/rtdetr_r50vd_m_6x_coco_from_paddle.pth"),
    "r50": ("rtdetr_r50vd_6x_coco.yml", "v0.1/rtdetr_r50vd_6x_coco_from_paddle.pth"),
    "r101": ("rtdetr_r101vd_6x_coco.yml", "v0.1/rtdetr_r101vd_6x_coco_from_paddle.pth"),
}

TASK = "object-detection"
SIZES = list(CKPT)
PRECISIONS = ["fp32"]
INPUT = {s: "640x640" for s in SIZES}
# tools/infer.py draws boxes above 0.6, over the task's 0.5, so predict() keeps them from 0.5 (docs/adapters.md).
THRESHOLD = 0.5


def package_version():
    return f"lyuwenyu/RT-DETR@{COMMIT} rtdetr_pytorch"


def checkout():
    if not (CHECKOUT / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(CHECKOUT)], check=True)
    head = subprocess.run(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(CHECKOUT), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(CHECKOUT), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def config(size):
    return REPO / "configs" / "rtdetr" / CKPT[size][0]


def download(size):
    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        part = ckpt.with_suffix(".part")
        urllib.request.urlretrieve(f"{RELEASE}/{CKPT[size][1]}", part)
        part.rename(ckpt)
    return ckpt


def names():
    from src.data.coco import mscoco_category2name, mscoco_label2category

    return [mscoco_category2name[mscoco_label2category[i]] for i in range(80)]


def load(size, precision):
    checkout()
    from src.core import YAMLConfig

    ckpt = download(size)
    cfg = YAMLConfig(str(config(size)), resume=str(ckpt))
    cfg.yaml_cfg["PResNet"]["pretrained"] = False
    checkpoint = torch.load(ckpt, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return {"net": Model().to(device()), "transforms": T.Compose([T.Resize((640, 640)), T.ToTensor()]), "names": names()}


@torch.inference_mode()
def eval_predict(m, x):
    # The top 300 boxes with every score, as the repository's COCO evaluation scores them.
    w, h = x.size
    labels, boxes, scores = m["net"](m["transforms"](x).unsqueeze(0).to(device()), torch.tensor([[w, h]], device=device()))
    return labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()


# predict() in the three phases `mtr profile` times apart: the tensor on the GPU, the network, and the decoded result.
@torch.inference_mode()
def preprocess(m, x):
    return m["transforms"](x).unsqueeze(0).to(device())


@torch.inference_mode()
def forward(m, images):
    return m["net"].model(images)


@torch.inference_mode()
def postprocess(m, raw, x):
    w, h = x.size
    labels, boxes, scores = m["net"].postprocessor(raw, torch.tensor([[w, h]], device=device()))
    labels, boxes, scores = labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()
    keep = scores > THRESHOLD
    return labels[keep], boxes[keep], scores[keep]


@torch.inference_mode()
def predict(m, x):
    return postprocess(m, forward(m, preprocess(m, x)), x)


def weight_files(size):
    return [WEIGHTS / Path(CKPT[size][1]).name]


def to_prediction(m, x, out):
    labels, boxes, scores = out
    # A low-scoring box can come back with a negative width or height. Collapse it to zero area: it still counts as a
    # false positive in COCO's evaluation.
    boxes = boxes.clone()
    boxes[:, 2:] = boxes[:, 2:].maximum(boxes[:, :2])
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
