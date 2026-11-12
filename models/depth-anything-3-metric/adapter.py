"""Adapter for Depth Anything 3 Metric (ByteDance Seed), metric monocular depth.

The model card loads DepthAnything3 from Hugging Face and calls model.inference(images). forward()
autocasts to bf16 on a GPU that supports it, so bf16 is the official precision, and the weights
themselves stay fp32. inference() defaults to process_res 504 on the longer side and returns the
depth map at that size, not at the input's. So a call covers no upscaling, and to_prediction()
scales the map up only for drawing.

The output is canonical depth. The README turns it into meters with focal * output / 300, which
needs the camera's focal length in pixels. Drawing scales each map to itself, so it needs no focal.
"""

import os
from pathlib import Path

os.environ.setdefault("DA3_LOG_LEVEL", "WARN")  # inference() prints three INFO lines per call

import torch
import torch.nn.functional as F
from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.io.input_processor import InputProcessor

from mtr_measure import device
from mtr_measure.predictions import Depth

COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
REPOS = {"large": ("depth-anything/DA3METRIC-LARGE", "4010e39f3634a45bc60553321fb49fb760bd594e")}

TASK = "depth-estimation"
SIZES = list(REPOS)
PRECISIONS = ["bf16"]
LATENCY_SCOPE = "predict(): pre-processing, forward pass and post-processing, with the depth map at the processing size; input file decoded beforehand"


def package_version():
    return f"ByteDance-Seed/Depth-Anything-3@{COMMIT}"


def input_size(size, image):
    # The processor scales the longer side to 504, then rounds both sides to multiples of 14.
    pixels = InputProcessor()([image], num_workers=1)[0]
    return f"{pixels.shape[-1]}x{pixels.shape[-2]}"


def load(size, precision):
    repo, revision = REPOS[size]
    return DepthAnything3.from_pretrained(repo, revision=revision).to(device()).eval()


def predict(model, x):
    return model.inference([x])


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(model, x, out):
    # Bilinear, because bicubic overshoots at the sky's edge and can go below zero. Canonical depth, not meters: drawing doesn't need the scale.
    depth = torch.from_numpy(out.depth[0])[None, None]
    return Depth(F.interpolate(depth, size=x.size[::-1], mode="bilinear", align_corners=False)[0, 0].numpy(), inverse=False)
