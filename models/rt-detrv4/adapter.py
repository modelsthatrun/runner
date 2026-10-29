"""Adapter for RT-DETRv4, real-time object detection on COCO's 80 classes.

RT-DETRv4 has no package. The adapter runs a pinned checkout of the official repository, the way
its tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the
image resized to 640x640 with torchvision, and the top 300 boxes back in the input image's pixels.
That script runs in fp32 (FP16 only through TensorRT), so PRECISIONS is fp32. The checkpoints are
the README's Google Drive links. Its DINOv3 teacher is used only in training.

The checkout is pinned to the last main commit before pull request #20 (merged 2026-05-22). That
pull request stopped moving the encoder's fixed position embedding to the model's device, leaving
it for an "inference wrapper" the repository doesn't have, so torch_inf.py fails on CUDA at the
current main (checked 2026-10-27).
"""

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import weights_dir
from mtr_measure.predictions import Boxes

REPO_URL = "https://github.com/RT-DETRs/RT-DETRv4.git"
COMMIT = "df4412261c6162f62aa4fb84e90b2df3c1b5f223"
WEIGHTS = weights_dir("rt-detrv4")
REPO = WEIGHTS / "RT-DETRv4"
# Google Drive file ids from the README model zoo.
DRIVE = {"s": "1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z", "m": "1O-YpP4X-quuOXbi96y2TKkztbjroP5mX",
         "l": "1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8", "x": "19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3"}

TASK = "object-detection"
SIZES = ["s", "m", "l", "x"]
PRECISIONS = ["fp32"]
INPUT = {s: "640x640" for s in SIZES}
# The README's display threshold in tools/inference/torch_inf.py.
THRESHOLD = 0.4


def package_version():
    return f"RT-DETRs/RT-DETRv4@{COMMIT}"


def checkout():
    if not (REPO / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def load(size, precision):
    checkout()
    from engine.core import YAMLConfig
    from engine.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        import gdown

        gdown.download(id=DRIVE[size], output=str(ckpt), quiet=True)
    cfg = YAMLConfig(str(REPO / "configs" / "rtv4" / f"rtv4_hgnetv2_{size}_coco.yml"), resume=str(ckpt))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
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

    names = [mscoco_category2name[mscoco_label2category[i]] for i in range(80)]
    return {"net": Model().to("cuda"), "transforms": T.Compose([T.Resize((640, 640)), T.ToTensor()]), "names": names}


@torch.inference_mode()
def eval_predict(m, x):
    # The top 300 boxes with every score, as the repository's COCO evaluation scores them.
    w, h = x.size
    labels, boxes, scores = m["net"](m["transforms"](x).unsqueeze(0).to("cuda"), torch.tensor([[w, h]], device="cuda"))
    return labels[0].cpu(), boxes[0].cpu(), scores[0].cpu()


@torch.inference_mode()
def predict(m, x):
    labels, boxes, scores = eval_predict(m, x)
    keep = scores > THRESHOLD
    return labels[keep], boxes[keep], scores[keep]


def weight_files(size):
    return [WEIGHTS / f"rtv4_hgnetv2_{size}_coco.pth"]


def to_prediction(m, x, out):
    labels, boxes, scores = out
    # As in D-FINE, whose box head RT-DETRv4 keeps, a low-scoring box can come back with a negative
    # width or height. Collapse it to zero area: it still counts as a false positive in COCO's evaluation.
    boxes = boxes.clone()
    boxes[:, 2:] = boxes[:, 2:].maximum(boxes[:, :2])
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
