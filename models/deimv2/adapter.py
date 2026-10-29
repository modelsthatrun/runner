"""Adapter for DEIMv2 (Intellindust), real-time object detection on COCO's 80 classes.

DEIMv2 has no package. The adapter runs a pinned checkout of the official repository, the way its
tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the image
resized to 640x640 and normalized with ImageNet statistics (the DINOv3 configs), and the top 300
boxes back in the input image's pixels. That script and the repository's COCO evaluation run in
fp32, so PRECISIONS is fp32. The weights are the authors' Hugging Face uploads (README section
2.0); they load into the config's model with only the two fixed decoder constants (`up`,
`reg_scale`) left at their config values.
"""

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import weights_dir
from mtr_measure.predictions import Boxes

REPO_URL = "https://github.com/Intellindust-AI-Lab/DEIMv2.git"
COMMIT = "f005f50ce565e63f6c298feccb51901a98c3d035"
REPO = weights_dir("deimv2") / "DEIMv2"
HF = {
    "s": ("Intellindust/DEIMv2_DINOv3_S_COCO", "cf0540f3f319bb8ecbe132624358a35a7beb86d5"),
    "m": ("Intellindust/DEIMv2_DINOv3_M_COCO", "c5db2fc6f4d9e59635cfc2effaf097bc9a293beb"),
    "l": ("Intellindust/DEIMv2_DINOv3_L_COCO", "6ce90ce18c14271c8ce145c35626d9a6544f881b"),
    "x": ("Intellindust/DEIMv2_DINOv3_X_COCO", "b1d81486dce5686b2afcd38a231d086d142c996d"),
}
# Parameters the config sets and the Hugging Face files leave out.
CONSTANTS = {"decoder.up", "decoder.reg_scale"}

TASK = "object-detection"
SIZES = list(HF)
PRECISIONS = ["fp32"]
INPUT = {s: "640x640" for s in SIZES}
# torch_inf.py's display threshold.
THRESHOLD = 0.45


def package_version():
    return f"Intellindust-AI-Lab/DEIMv2@{COMMIT}"


def checkout():
    if not (REPO / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = HF[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def load(size, precision):
    checkout()
    from safetensors.torch import load_file

    from engine.core import YAMLConfig
    from engine.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    cfg = YAMLConfig(str(REPO / "configs" / "deimv2" / f"deimv2_dinov3_{size}_coco.yml"))
    state = {k: v for k, v in load_file(weight_files(size)[0]).items() if not k.startswith("postprocessor.")}
    missing, unexpected = cfg.model.load_state_dict(state, strict=False)
    if set(missing) - CONSTANTS or unexpected:
        raise RuntimeError(f"weights don't match the config: missing {missing}, unexpected {unexpected}")

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    h, w = cfg.yaml_cfg["eval_spatial_size"]
    transforms = T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    names = [mscoco_category2name[mscoco_label2category[i]] for i in range(80)]
    return {"net": Model().to("cuda"), "transforms": transforms, "names": names}


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


def to_prediction(m, x, out):
    labels, boxes, scores = out
    # As in D-FINE, whose box head DEIMv2 keeps, a low-scoring box can come back with a negative
    # width or height. Collapse it to zero area: it still counts as a false positive in COCO's evaluation.
    boxes = boxes.clone()
    boxes[:, 2:] = boxes[:, 2:].maximum(boxes[:, :2])
    return Boxes(boxes.numpy(), scores.numpy(), [m["names"][int(i)] for i in labels])
