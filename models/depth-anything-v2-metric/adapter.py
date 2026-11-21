"""Adapter for Depth Anything V2 Metric, monocular metric depth.

The authors fine-tuned Depth Anything V2 for metric depth on Hypersim (indoor, depth up to 20 m) and Virtual KITTI 2
(outdoor, up to 80 m) (metric_depth/README.md). The transformers ports under depth-anything/ on Hugging Face carry the
same checkpoints with their maximum depth in the config. The model card's usage is the depth-estimation pipeline:
AutoModelForDepthEstimation with its image processor (shorter side 518, both sides multiples of 14), then
predicted_depth interpolated back to the image's size. The output is depth in meters: larger is farther. It runs in
fp32, as the card and the repository's run.py do.

The repository's README licenses Small under Apache-2.0 and Base and Large under CC BY-NC 4.0, so Base and Large
are non-commercial whatever the ports' cards say.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from mtr_measure import device
from mtr_measure.predictions import Depth

REPOS = {
    "indoor-small": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "8078d68a9c75a972131914f6afd0c1723be0da7f"),
    "indoor-base": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf", "c6d9784685727bfc6d0a7b5452ce94afaee1e7f5"),
    "indoor-large": ("depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf", "d2fc6a93601aabb1139a3bf0ebfcb4e89c67817f"),
    "outdoor-small": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf", "fd2c22027eaf20374204f14099b8341e1925ad39"),
    "outdoor-base": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf", "5b575fded0346068343a6d96d0ce3ab6dbc9fb49"),
    "outdoor-large": ("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf", "4eab4cf1983c2801c515804005214de56a4b67cc"),
}

TASK = "depth-estimation"
SIZES = list(REPOS)
PRECISIONS = ["fp32"]
PACKAGE = "transformers"


def input_size(size, image):
    # The processor keeps the aspect ratio, with the shorter side at 518 and both sides multiples of 14.
    repo, revision = REPOS[size]
    pixels = AutoImageProcessor.from_pretrained(repo, revision=revision)(images=image, return_tensors="pt")["pixel_values"]
    return f"{pixels.shape[-1]}x{pixels.shape[-2]}"


def load(size, precision):
    repo, revision = REPOS[size]
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    model = AutoModelForDepthEstimation.from_pretrained(repo, revision=revision).to(device()).eval()
    return {"model": model, "processor": processor}


@torch.no_grad()
def predict(m, x):
    inputs = m["processor"](images=x, return_tensors="pt").to(device())
    depth = m["model"](**inputs).predicted_depth
    return F.interpolate(depth.unsqueeze(1), size=x.size[::-1], mode="bicubic", align_corners=False)[0, 0].cpu()


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(m, x, out):
    return Depth(out.numpy(), inverse=False)
