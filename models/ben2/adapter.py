"""Adapter for BEN2 Base (Prama LLC), background removal: one foreground mask per image.

The model card loads BEN_Base.from_pretrained("PramaLLC/BEN2") from the ben2 package (installed
from its git repository) and calls model.inference(image, refine_foreground=False), which resizes
to 1024x1024, runs the network and returns the image with the predicted alpha. On CUDA the package
always runs in fp16 (an fp16 input and an autocast decorator on forward()), so PRECISIONS is fp16.

inference() puts the alpha onto the image it was given, in place. predict() hands it a copy, so
every timed call starts from the same RGB image, as a call on a freshly opened file would.
"""

from pathlib import Path

import numpy as np
import torch
from ben2 import BEN_Base

from mtr_measure.predictions import Alpha

REPO, REVISION = "PramaLLC/BEN2", "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"
COMMIT = "2c99a5da477b5523585bfa5c893888a6e818a8f6"

TASK = "image-matting"
SIZES = ["base"]
PRECISIONS = ["fp16"]
INPUT = {"base": "1024x1024"}


def package_version():
    return f"PramaLLC/BEN2@{COMMIT}"


def load(size, precision):
    model = BEN_Base.from_pretrained(REPO, revision=REVISION)
    return model.to("cuda").eval()


@torch.no_grad()
def predict(model, x):
    return model.inference(x.copy(), refine_foreground=False)


def weight_files(size):
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(REPO, "model.safetensors", revision=REVISION))]


def to_prediction(model, x, out):
    return Alpha(np.asarray(out.getchannel("A"), dtype=np.float32) / 255)
