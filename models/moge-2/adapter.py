"""Adapter for MoGe-2 (Microsoft), metric monocular geometry: depth, points and normals.

The README loads MoGeModel (moge.model.v2) from Hugging Face, turns the image into a float tensor
in [0, 1], and calls model.infer(image). infer() defaults to resolution_level 9 and use_fp16=True
(fp16 autocast for the network, fp32 for the geometry), so fp16 is the official precision.

The package is pinned to the last commit before 2026-08-18, when MoGe-3 added FlexGEMM, whose
CUDA kernels must be compiled. MoGe-2 itself doesn't use them.
"""

from pathlib import Path

import numpy as np
import torch
from moge.model.v2 import MoGeModel

from mtr_measure.predictions import Depth

REPOS = {
    "vits": ("Ruicheng/moge-2-vits-normal", "26b477f41595707c5db6770294c0d1721e8ed4ed"),
    "vitb": ("Ruicheng/moge-2-vitb-normal", "ca5f0e07ff01d3e5a364c1d954ed12ee1814b368"),
    "vitl": ("Ruicheng/moge-2-vitl-normal", "cb0e8bbd6b1e243589717c78e750b1ba4c093acf"),
}
COMMIT = "925b8ed835a7a9cdb7578ba15c658a0afc969030"
RESOLUTION_LEVEL = 9  # infer()'s default
PATCH = 14
_tokens_range = {}

TASK = "depth-estimation"
SIZES = list(REPOS)
PRECISIONS = ["fp16", "fp32"]


def package_version():
    return f"microsoft/MoGe@{COMMIT}"


def input_size(size, image):
    # infer(): resolution_level 9 picks the top of the model's token range; forward() splits it by aspect ratio.
    lo, hi = _tokens_range[size]
    tokens = int(lo + (RESOLUTION_LEVEL / 9) * (hi - lo))
    aspect = image.width / image.height
    return f"{round((tokens * aspect) ** 0.5) * PATCH}x{round((tokens / aspect) ** 0.5) * PATCH}"


def load(size, precision):
    model = MoGeModel.from_pretrained(weight_files(size)[0]).to("cuda").eval()
    _tokens_range[size] = model.num_tokens_range
    return {"model": model, "fp16": precision == "fp16"}


@torch.no_grad()
def predict(m, x):
    image = torch.tensor(np.asarray(x) / 255, dtype=torch.float32, device="cuda").permute(2, 0, 1)
    return m["model"].infer(image, use_fp16=m["fp16"])


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.pt", revision=revision))]


def to_prediction(m, x, out):
    # Metric depth in meters; pixels outside the predicted mask are infinite.
    return Depth(out["depth"].cpu().numpy(), inverse=False)
