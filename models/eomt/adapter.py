"""Adapter for EoMT (Encoder-only Mask Transformer), instance segmentation on COCO's 80 classes.

The model card loads EomtForUniversalSegmentation and its image processor from transformers, runs
the model and calls post_process_instance_segmentation (threshold 0.8). The official repository
(tue-mps/eomt) trains and evaluates with Lightning's "16-mixed" precision and
torch.set_float32_matmul_precision("medium"), so the forward pass runs under fp16 autocast here.
Its COCO evaluation (training/mask_classification_instance.py) keeps the top 100 (query, class)
pairs with no threshold, scored as class probability times mean mask probability; eval_predict()
does the same. Only the DINOv2 checkpoints are listed: the DINOv3 ones are deltas that need access
to Meta's manually gated DINOv3 weights.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, EomtForUniversalSegmentation

from mtr_measure.predictions import Masks

REPOS = {
    "large-640": ("tue-mps/coco_instance_eomt_large_640", "a43666870d47a7010bfea2c4554f7ca829714558"),
    "large-1280": ("tue-mps/coco_instance_eomt_large_1280", "057a34b54f9f1cf5395de72edaeaefca43e66494"),
}

TASK = "instance-segmentation"
SIZES = list(REPOS)
PRECISIONS = ["fp16", "fp32"]
PACKAGE = "transformers"
INPUT = {"large-640": "640x640", "large-1280": "1280x1280"}
TOP_K = 100


def load(size, precision):
    torch.set_float32_matmul_precision("medium")
    repo, revision = REPOS[size]
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision)
    model = EomtForUniversalSegmentation.from_pretrained(repo, revision=revision).to("cuda").eval()
    return {"model": model, "processor": processor, "fp16": precision == "fp16", "names": model.config.id2label}


def forward(m, x):
    inputs = m["processor"](images=x, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.float16, enabled=m["fp16"]):
        return m["model"](**inputs)


@torch.inference_mode()
def predict(m, x):
    outputs = forward(m, x)
    # The model card's call, with the score cut-off lowered from its default 0.8 to the task's fixed
    # threshold (docs/protocol.md), so the tool's threshold decides what is drawn, as for every model.
    return m["processor"].post_process_instance_segmentation(outputs, target_sizes=[(x.height, x.width)], threshold=0.5)[0]


@torch.inference_mode()
def eval_predict(m, x):
    outputs = forward(m, x)
    processor = m["processor"]
    size = processor.size
    mask_logits = F.interpolate(outputs.masks_queries_logits.float(), size=(size["longest_edge"],) * 2, mode="bilinear")
    mask_logits = processor.unpad_image(mask_logits, [(x.height, x.width)], size)[0]
    scores = outputs.class_queries_logits[0].float().softmax(dim=-1)[:, :-1]
    top_scores, top = scores.flatten(0, 1).topk(TOP_K, sorted=False)
    labels = top % scores.shape[-1]
    mask_logits = mask_logits[top // scores.shape[-1]]
    masks = mask_logits > 0
    mask_scores = (mask_logits.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
    names = m["names"]
    return Masks(masks.cpu().numpy(), [names[int(i)] for i in labels], (top_scores * mask_scores).cpu().numpy())


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "model.safetensors", revision=revision))]


def to_prediction(m, x, out):
    if isinstance(out, Masks):
        return out
    # post_process_instance_segmentation returns one map of segment ids and the segments' labels. A
    # segment whose pixels all went to higher-scoring segments has no mask left, so it is left out.
    seg = out["segmentation"].cpu().numpy()
    info = [s for s in out["segments_info"] if (seg == s["id"]).any()]
    masks = np.stack([seg == s["id"] for s in info]) if info else np.zeros((0, *seg.shape), bool)
    return Masks(masks, [m["names"][s["label_id"]] for s in info], np.array([s["score"] for s in info], np.float32))
