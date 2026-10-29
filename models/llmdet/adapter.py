"""Adapter for LLMDet (iSEE Laboratory), open-vocabulary object detection from a text prompt.

LLMDet is MM-Grounding-DINO trained with a large language model's captions; at inference the
language model isn't used. The model card runs AutoModelForZeroShotObjectDetection from
transformers on an image and a list of labels, then post_process_grounded_object_detection with
threshold 0.4. It runs in fp32. The prompt here is always COCO's 80 class names, so every
open-vocabulary model does the same work and can be scored zero-shot on COCO.

Its evaluation, built on mmdetection, scores each class as the mean probability of its tokens and
keeps the top 300 (query, class) pairs; eval_predict() does the same.

Transformers is pinned to 4.57.1: from 5.0.0 on, the MM-Grounding-DINO architecture returns broken
boxes (thin vertical strips running out of the image) while its scores stay right; 4.57.1 gives
the expected boxes on the same image (checked 2026-10-27).
"""

import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from transformers.image_transforms import get_size_with_aspect_ratio

from mtr_measure.predictions import Boxes

REPOS = {
    "tiny": ("iSEE-Laboratory/llmdet_tiny", "d05199165a19320a9236396e20fca0a5e065189c"),
    "base": ("iSEE-Laboratory/llmdet_base", "3537869413e7dca01cfd86d6ab10553b72e4fe29"),
    "large": ("iSEE-Laboratory/llmdet_large", "bec37f296f05b22f6c6b39bc05a6c611239f4e31"),
}
# The model card's threshold; it leaves text_threshold at the processor's default.
THRESHOLD, TEXT_THRESHOLD = 0.4, 0.25
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
    # The labels are the prompt phrases each box matched.
    return Boxes(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy(), list(out["text_labels"]))
