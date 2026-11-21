"""Adapter for BiRefNet (ZhengPeng7), background removal: one foreground mask per image.

The model card loads AutoModelForImageSegmentation with trust_remote_code, sets
torch.set_float32_matmul_precision("high"), calls .half(), resizes the image to the checkpoint's
square size with ImageNet normalization, takes the last output's sigmoid, and resizes that mask
back to the image with PIL. predict() is that whole extract_object() call, without pasting the
alpha onto the image. Model and input are cast with .half(), so fp16 is the official precision.

The portrait and matting checkpoints use the general card's code at 1024, the size they were trained at, and
HR-matting its own card's 2048. BiRefNet_dynamic isn't listed: its card leaves the resize out, and the model then
fails on an image whose sides aren't multiples of its patch grid (1600x1067 here).
"""

from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

from mtr_measure import device
from mtr_measure.predictions import Alpha

REPOS = {
    "general": ("ZhengPeng7/BiRefNet", "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4", 1024),
    "hr": ("ZhengPeng7/BiRefNet_HR", "a7a562f6fd16021180f2f4348f4de003a2d3d1e1", 2048),
    "lite": ("ZhengPeng7/BiRefNet_lite", "aa62cd87eafb9cc43056d08ef3615a14628b831d", 1024),
    "portrait": ("ZhengPeng7/BiRefNet-portrait", "b6561965a70070d9143fd9e558f6ca3c481510db", 1024),
    "matting": ("ZhengPeng7/BiRefNet-matting", "eccde0a8cbdce7ac5fecfeb06340fe7b949e85d9", 1024),
    "hr-matting": ("ZhengPeng7/BiRefNet_HR-matting", "5d6b6f8adcb5b417c871b1d84ceaae9871355b7f", 2048),
}

TASK = "image-matting"
SIZES = list(REPOS)
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "transformers"
INPUT = {s: f"{r[2]}x{r[2]}" for s, r in REPOS.items()}


def load(size, precision):
    torch.set_float32_matmul_precision("high")
    repo, revision, side = REPOS[size]
    model = AutoModelForImageSegmentation.from_pretrained(repo, revision=revision, trust_remote_code=True)
    model.to(device()).eval()
    dtype = torch.float16 if precision == "fp16" else torch.float32
    model.to(dtype)
    transform = transforms.Compose([
        transforms.Resize((side, side)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return {"model": model, "transform": transform, "dtype": dtype}


@torch.no_grad()
def predict(m, x):
    images = m["transform"](x).unsqueeze(0).to(device(), dtype=m["dtype"])
    pred = m["model"](images)[-1].sigmoid().cpu()[0].squeeze()
    return transforms.ToPILImage()(pred.float()).resize(x.size)


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision, _ = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(m, x, out):
    return Alpha(np.asarray(out, dtype=np.float32) / 255)
