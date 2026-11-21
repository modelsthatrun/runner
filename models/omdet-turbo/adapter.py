"""Adapter for OmDet-Turbo (Om AI Lab), real-time open-vocabulary object detection from class names.

The model card runs OmDetTurboForObjectDetection from transformers on an image and a list of class
names, then post_process_grounded_object_detection with threshold 0.3 and class-wise NMS at 0.3. The
processor resizes the image to 640x640 without padding. It runs in fp32, as the card does. The class
names here are always COCO's 80, so every open-vocabulary model does the same work and can be scored
zero-shot on COCO.

For COCO, eval_predict() makes the same call with no score threshold, keeping the card's NMS, so every
candidate is scored; the tool then keeps the 100 best per image.
"""

import torch
from transformers import AutoProcessor, OmDetTurboForObjectDetection

from mtr_measure import device
from mtr_measure.predictions import Boxes

REPOS = {"tiny": ("omlab/omdet-turbo-swin-tiny-hf", "7fe93cecfb770c4d76cf71163956221249cab566")}
# The model card's thresholds.
THRESHOLD, NMS_THRESHOLD = 0.3, 0.3
COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

TASK = "object-detection"
SIZES = list(REPOS)
PRECISIONS = ["fp32"]
PACKAGE = "transformers"


def input_size(size, image):
    # The image processor resizes every image to 640x640 (preprocessor_config.json), with no padding.
    return "640x640"


def load(size, precision):
    repo, revision = REPOS[size]
    processor = AutoProcessor.from_pretrained(repo, revision=revision)
    model = OmDetTurboForObjectDetection.from_pretrained(repo, revision=revision).to(device()).eval()
    return {"model": model, "processor": processor}


def run(m, x, threshold):
    inputs = m["processor"](x, text=COCO_NAMES, return_tensors="pt").to(device())
    outputs = m["model"](**inputs)
    return m["processor"].post_process_grounded_object_detection(
        outputs, text_labels=COCO_NAMES, target_sizes=[(x.height, x.width)], threshold=threshold, nms_threshold=NMS_THRESHOLD)[0]


@torch.inference_mode()
def predict(m, x):
    return run(m, x, THRESHOLD)


@torch.inference_mode()
def eval_predict(m, x):
    return run(m, x, 0.0)


def weight_files(size):
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(m, x, out):
    return Boxes(out["boxes"].float().cpu().numpy(), out["scores"].float().cpu().numpy(), [COCO_NAMES[int(i)] for i in out["labels"].tolist()])
