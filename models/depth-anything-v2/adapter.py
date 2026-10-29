"""Adapter for Depth Anything V2 (Small), relative monocular depth.

The model card loads AutoModelForDepthEstimation and its image processor from transformers, runs
the model, and interpolates predicted_depth back to the image's size (bicubic). The output is
relative inverse depth: larger is nearer. It runs in fp32, as the card and the original
repository's run.py do. Only Small is listed: Base and Large are CC BY-NC 4.0.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from mtr_measure.predictions import Depth

REPOS = {"small": ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99")}

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
    model = AutoModelForDepthEstimation.from_pretrained(repo, revision=revision).to("cuda").eval()
    return {"model": model, "processor": processor}


@torch.no_grad()
def predict(m, x):
    inputs = m["processor"](images=x, return_tensors="pt").to("cuda")
    depth = m["model"](**inputs).predicted_depth
    return F.interpolate(depth.unsqueeze(1), size=x.size[::-1], mode="bicubic", align_corners=False)[0, 0].cpu()


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(m, x, out):
    return Depth(out.numpy(), inverse=True)
