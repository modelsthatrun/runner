"""Adapter for OWLv2 (Google), open-vocabulary object detection from text queries.

The model card runs Owlv2ForObjectDetection from transformers on an image and a list of text queries
written as "a photo of a <name>", then post_process_object_detection with threshold 0.1 and the image's
own size (the processor pads the image to a square, and post-processing scales boxes back to it). It
runs in fp32, as the card does. The queries here are always COCO's 80 class names in the card's
phrasing, so every open-vocabulary model does the same work and can be scored zero-shot on COCO.

Post-processing gives each predicted box its best-scoring query. For COCO, eval_predict() makes the
same call with no threshold, so every box is scored; the tool then keeps the 100 best per image.
"""

import torch
from transformers import Owlv2ForObjectDetection, Owlv2Processor

from mtr_measure import device
from mtr_measure.predictions import Boxes

REPOS = {
    "base": ("google/owlv2-base-patch16-ensemble", "cfd3195ba4ea9592eec887ded089f4c08eff231d"),
    "large": ("google/owlv2-large-patch14-ensemble", "95e26936e865f87db1742128404b3c035d47d89d"),
}
# The model card's threshold.
THRESHOLD = 0.1
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
QUERIES = [f"a photo of a {name}" for name in COCO_NAMES]

TASK = "object-detection"
SIZES = list(REPOS)
PRECISIONS = ["fp32"]
PACKAGE = "transformers"


def input_size(size, image):
    # The processor pads the image to a square at its longer side, then resizes it to the model's input.
    side = Owlv2Processor.from_pretrained(REPOS[size][0], revision=REPOS[size][1]).image_processor.size["height"]
    return f"{side}x{side}"


def load(size, precision):
    repo, revision = REPOS[size]
    processor = Owlv2Processor.from_pretrained(repo, revision=revision)
    model = Owlv2ForObjectDetection.from_pretrained(repo, revision=revision).to(device()).eval()
    return {"model": model, "processor": processor}


def run(m, x, threshold):
    inputs = m["processor"](text=[QUERIES], images=x, return_tensors="pt").to(device())
    outputs = m["model"](**inputs)
    # transformers 5 moved the card's processor.post_process_object_detection to the image processor.
    return m["processor"].image_processor.post_process_object_detection(outputs=outputs, target_sizes=[(x.height, x.width)], threshold=threshold)[0]


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
