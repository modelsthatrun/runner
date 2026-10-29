"""Example adapter: torchvision ResNet-50, image classification. Not a catalog model."""

from pathlib import Path

import torch
from torchvision.models import ResNet50_Weights, resnet50

WEIGHTS = ResNet50_Weights.IMAGENET1K_V2

TASK = "image-classification"
SIZES = ["default"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "torchvision"
INPUT = {"default": "224x224"}  # the weights' transforms: resize to 232, center crop 224


def load(size, precision):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    net = resnet50(weights=WEIGHTS).eval().to("cuda", dtype)
    return {"net": net, "dtype": dtype, "preprocess": WEIGHTS.transforms()}


@torch.inference_mode()
def predict(m, image):
    batch = m["preprocess"](image).unsqueeze(0).to("cuda", m["dtype"])
    top = m["net"](batch).float().softmax(-1).topk(5)
    return top.values.cpu(), top.indices.cpu()


def weight_files(size):
    return [Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name]
