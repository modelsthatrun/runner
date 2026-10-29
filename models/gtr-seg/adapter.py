"""Adapter for GTR-Seg (Gated Token Recurrence), real-time instance segmentation on COCO's 80 classes.

GTR has no package. The adapter runs a pinned checkout of the official repository, the way its
tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the image
resized to the config's 640x640 and normalized with ImageNet statistics, and the top 300
instances back with mask logits at the head's resolution. Its latency script,
tools/benchmark/torch_speed.py, casts the model and input to fp16 (its default), so fp16 is the
official precision.

The README's latencies use an optional fused CUDA GLA kernel (engine/gtr/backbone/csrc), whose
build needs nvcc to match torch's cu128; the reference machine has CUDA 12.4. Without it the backbone uses
flash-linear-attention's Triton kernel, which the README names as the fallback.
"""

import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from mtr_measure import weights_dir
from mtr_measure.predictions import Masks

REPO_URL = "https://github.com/Intellindust-AI-Lab/GTR.git"
COMMIT = "f8ffd2444bfc2c1adfa70933e28afa37a0377830"
HF_REPO = "Phoenix8125/GTR"
HF_REVISION = "9fc62c8c2b2c976835d0f1c1ffc544dbc0f9e29f"
WEIGHTS = weights_dir("gtr")
REPO = WEIGHTS / "GTR"

TASK = "instance-segmentation"
SIZES = ["s", "m", "l", "x"]
PRECISIONS = ["fp16", "fp32"]
INPUT = {s: "640x640" for s in SIZES}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
# torch_inf.py's default display threshold.
THRESHOLD = 0.45


def package_version():
    return f"Intellindust-AI-Lab/GTR@{COMMIT}"


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
    return [WEIGHTS / "seg" / f"gtrseg_{size}_coco.pth"]


def load(size, precision):
    checkout()
    from engine.core import YAMLConfig
    from engine.data.dataset.coco_dataset import mscoco_category2name, mscoco_label2category

    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(HF_REPO, f"seg/{ckpt.name}", revision=HF_REVISION, local_dir=WEIGHTS)
    cfg = YAMLConfig(str(REPO / "configs" / "seg" / "coco_seg_finetune" / f"gtrseg_{size}.yml"), resume=str(ckpt))
    for backbone in ("ViTAdapter", "ViTAdapterSpatialSwiGLU"):
        if backbone in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone]["skip_weights_warning"] = True
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    dtype = DTYPES[precision]
    h, w = cfg.yaml_cfg["eval_spatial_size"]
    transforms = T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    names = [mscoco_category2name[mscoco_label2category[i]] for i in range(80)]
    return {"net": Model().to("cuda", dtype=dtype), "dtype": dtype, "transforms": transforms, "names": names}


@torch.inference_mode()
def eval_predict(m, x):
    # The top 300 instances with every score, as the repository's COCO evaluation scores them.
    w, h = x.size
    images = m["transforms"](x).unsqueeze(0).to("cuda", dtype=m["dtype"])
    labels, boxes, scores, masks = m["net"](images, torch.tensor([[w, h]], device="cuda"))
    return labels[0].cpu(), scores[0].float().cpu(), masks[0]


@torch.inference_mode()
def predict(m, x):
    labels, scores, masks = eval_predict(m, x)
    keep = scores > THRESHOLD
    return labels[keep], scores[keep], masks[keep.to(masks.device)].cpu()


def to_prediction(m, x, out):
    labels, scores, masks = out
    # The repository's evaluation and torch_inf.py: upsample the logits to the image, sigmoid > 0.5.
    masks = F.interpolate(masks[:, None].float(), size=(x.height, x.width), mode="bilinear", align_corners=False)[:, 0] > 0
    return Masks(masks.cpu().numpy(), [m["names"][int(i)] for i in labels], scores.numpy())
