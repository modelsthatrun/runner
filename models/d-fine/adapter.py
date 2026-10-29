"""Adapter for D-FINE, real-time object detection on COCO's 80 classes.

D-FINE has no package. The adapter runs a pinned checkout of the official repository, the way its
tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the image
resized to 640x640 with torchvision, and the top 300 boxes back in the input image's pixels. That
script and the repository's COCO evaluation run in fp32; FP16 exists only through TensorRT, so
PRECISIONS is fp32. Only the COCO-trained checkpoints are listed: the README says the Objects365
ones may be bound by the Objects365 dataset terms.
"""

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import weights_dir
from mtr_measure.predictions import Boxes

REPO_URL = "https://github.com/Peterande/D-FINE.git"
COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
WEIGHTS = weights_dir("d-fine")
REPO = WEIGHTS / "D-FINE"
RELEASE = "https://github.com/Peterande/storage/releases/download/dfinev1.0"

TASK = "object-detection"
SIZES = ["n", "s", "m", "l", "x"]
PRECISIONS = ["fp32"]
INPUT = {s: "640x640" for s in SIZES}
# The README's display threshold in tools/inference/torch_inf.py.
THRESHOLD = 0.4


def package_version():
    return f"Peterande/D-FINE@{COMMIT}"


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
    from src.core import YAMLConfig
    from src.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        torch.hub.download_url_to_file(f"{RELEASE}/{ckpt.name}", str(ckpt))
    cfg = YAMLConfig(str(REPO / "configs" / "dfine" / f"dfine_hgnetv2_{size}_coco.yml"), resume=str(ckpt))
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
    return [WEIGHTS / f"dfine_{size}_coco.pth"]


def to_prediction(m, x, out):
    labels, boxes, scores = out
    # A few of the top 300 come back with a negative width or height (152 of 90,000 on 300 COCO
    # images for x, all scoring under 0.02). Collapse them to zero area: they still count as false
    # positives in COCO's evaluation, and never as matches.
    boxes = boxes.clone()
    boxes[:, 2:] = boxes[:, 2:].maximum(boxes[:, :2])
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
