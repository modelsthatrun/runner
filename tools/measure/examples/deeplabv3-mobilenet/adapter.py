"""Example adapter: torchvision DeepLabV3 with a MobileNetV3-Large backbone, semantic segmentation.

Not a catalog model. predict() returns a class map, so the output is a mask rather than boxes
or labels. to_prediction() turns it into one mask per class for `mtr outputs`.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

from mtr_measure.predictions import Masks

WEIGHTS = DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1

TASK = "semantic-segmentation"
SIZES = ["default"]
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "torchvision"


def input_size(size, image):
    # The weights' transforms resize the shorter side to 520.
    w, h = image.size
    if w <= h:
        return f"520x{int(520 * h / w)}"
    return f"{int(520 * w / h)}x520"


def load(size, precision):
    dtype = torch.float16 if precision == "fp16" else torch.float32
    net = deeplabv3_mobilenet_v3_large(weights=WEIGHTS).eval().to("cuda", dtype)
    return {"net": net, "dtype": dtype, "preprocess": WEIGHTS.transforms()}


@torch.inference_mode()
def predict(m, image):
    x = m["preprocess"](image).unsqueeze(0).to("cuda", m["dtype"])
    return m["net"](x)["out"].argmax(1)[0].to(torch.uint8).cpu()


def weight_files(size):
    return [Path(torch.hub.get_dir()) / "checkpoints" / Path(WEIGHTS.url).name]


def to_prediction(m, image, out):
    # The class map is at the resized input (shorter side 520). Bring it back to the image's size.
    class_map = np.asarray(Image.fromarray(out.numpy()).resize(image.size, Image.Resampling.NEAREST))
    return Masks.from_class_map(class_map, WEIGHTS.meta["categories"], background=0)
