"""Adapter for Grounding DINO (IDEA Research), open-vocabulary object detection from a text prompt.

The model card runs AutoModelForZeroShotObjectDetection from transformers on an image and a list of
labels, then post_process_grounded_object_detection with threshold 0.4 and text_threshold 0.3. It
runs in fp32, as the original repository does. The prompt here is always COCO's 80 class names, so
every open-vocabulary model does the same work and can be scored zero-shot on COCO.

The original repository's COCO evaluation (demo/test_ap_on_coco.py) scores each class as the mean
probability of its tokens and keeps the top 300 (query, class) pairs; eval_predict() does the same.
"""

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.image_transforms import get_size_with_aspect_ratio

from mtr_measure.predictions import Boxes

REPOS = {
    "tiny": ("IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71"),
    "base": ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"),
}
# The model card's thresholds.
THRESHOLD, TEXT_THRESHOLD = 0.4, 0.3
NUM_SELECT = 300
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
    # The image processor's own rule: shorter side 800, longer side at most 1333.
    oh, ow = get_size_with_aspect_ratio((image.height, image.width), 800, 1333)
    return f"{ow}x{oh}"


def positive_map(processor, names, length):
    """(classes, text length): 1/n on the n tokens of each class name in the merged prompt."""
    caption = ". ".join(names) + "."
    offsets = processor.tokenizer(caption, return_offsets_mapping=True)["offset_mapping"]
    pmap = torch.zeros(len(names), length)
    start = 0
    for j, name in enumerate(names):
        end = start + len(name)
        for t, (b, e) in enumerate(offsets):
            if e > b and b >= start and e <= end:
                pmap[j, t] = 1
        start = end + 2
    return pmap / pmap.sum(-1, keepdim=True)


def load(size, precision):
    repo, revision = REPOS[size]
    processor = AutoProcessor.from_pretrained(repo, revision=revision)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(repo, revision=revision).to("cuda").eval()
    pmap = positive_map(processor, COCO_NAMES, model.config.max_text_len).to("cuda")
    return {"model": model, "processor": processor, "pmap": pmap}


def forward(m, x):
    inputs = m["processor"](images=x, text=[COCO_NAMES], return_tensors="pt").to("cuda")
    return inputs, m["model"](**inputs)


@torch.inference_mode()
def predict(m, x):
    inputs, outputs = forward(m, x)
    return m["processor"].post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(x.height, x.width)])[0]


@torch.inference_mode()
def eval_predict(m, x):
    _, outputs = forward(m, x)
    prob = outputs.logits[0].sigmoid() @ m["pmap"].T
    scores, index = prob.flatten().topk(NUM_SELECT)
    labels = index % prob.shape[1]
    cx, cy, w, h = outputs.pred_boxes[0][index // prob.shape[1]].unbind(-1)
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) * torch.tensor([x.width, x.height] * 2, device="cuda")
    return Boxes(boxes.cpu().numpy(), scores.cpu().numpy(), [COCO_NAMES[int(i)] for i in labels])


def weight_files(size):
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(m, x, out):
    if isinstance(out, Boxes):
        return out
    # The labels are the prompt phrases each box matched. With no box, post-processing still returns
    # one empty label ([""]), so the labels follow the boxes.
    labels = list(out["text_labels"]) if len(out["scores"]) else []
    return Boxes(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), labels)
