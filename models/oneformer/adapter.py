"""Adapter for OneFormer (SHI Labs), instance segmentation on COCO's 80 thing classes.

OneFormer is one model for semantic, instance and panoptic segmentation, trained once on COCO panoptic (133
classes, the first 80 of them COCO's things). The model card runs OneFormerForUniversalSegmentation from
transformers with the processor and task_inputs=["instance"]. The processor resizes the image to a shorter side of
800 and a longer side of at most 1333, as the authors' COCO configs test it. Instances are read the way the authors'
repository does (oneformer/oneformer_model.py, instance_inference()): mask logits upsampled to the padded input,
cropped to the image and resized to its own size, the top 100 (query, class) pairs by class probability
(TEST.DETECTIONS_PER_IMAGE), then only "thing" classes kept, since the model was trained on panoptic labels. Each
mask is the pixels with a positive logit, scored as class probability times the mask's mean probability inside it.
It runs in fp32, as the card and the repository's test do. predict() returns every kept instance, as the
repository's predictor does; the tool's fixed threshold decides what is drawn. Only the Swin-L checkpoint is listed:
the DiNAT-L one needs NATTEN's compiled kernels.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

from mtr_measure import device
from mtr_measure.predictions import Masks

REPOS = {"swin-large": ("shi-labs/oneformer_coco_swin_large", "3a263017ca5c75adbea145f25f81b118243d4394")}
TOP_K = 100
THINGS = 80  # COCO panoptic's contiguous ids 0-79 are the things

TASK = "instance-segmentation"
SIZES = list(REPOS)
PRECISIONS = ["fp32"]
PACKAGE = "transformers"


def load(size, precision):
    repo, revision = REPOS[size]
    processor = OneFormerProcessor.from_pretrained(repo, revision=revision)
    model = OneFormerForUniversalSegmentation.from_pretrained(repo, revision=revision).to(device()).eval()
    return {"model": model, "processor": processor, "names": model.config.id2label}


def input_size(size, image):
    repo, revision = REPOS[size]
    pixels = OneFormerProcessor.from_pretrained(repo, revision=revision)(images=image, task_inputs=["instance"], return_tensors="pt")["pixel_values"]
    return f"{pixels.shape[-1]}x{pixels.shape[-2]}"


def instance_inference(inputs, outputs, x):
    scores = outputs.class_queries_logits[0].float().softmax(-1)[:, :-1]
    classes = scores.shape[-1]
    top_scores, top = scores.flatten().topk(TOP_K, sorted=False)
    labels = top % classes
    keep = labels < THINGS
    top_scores, top, labels = top_scores[keep], top[keep], labels[keep]
    logits = outputs.masks_queries_logits[0, top // classes].float()
    ph, pw = inputs["pixel_values"].shape[-2:]
    logits = F.interpolate(logits[None], size=(ph, pw), mode="bilinear", align_corners=False)[0]
    valid = inputs["pixel_mask"][0].bool()
    h, w = int(valid.any(1).sum()), int(valid.any(0).sum())
    logits = F.interpolate(logits[None, :, :h, :w], size=(x.height, x.width), mode="bilinear", align_corners=False)[0]
    masks = logits > 0
    mask_scores = (logits.sigmoid().flatten(1) * masks.flatten(1)).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
    return masks, labels, top_scores * mask_scores


@torch.inference_mode()
def predict(m, x):
    inputs = m["processor"](images=x, task_inputs=["instance"], return_tensors="pt").to(device())
    outputs = m["model"](pixel_values=inputs["pixel_values"], pixel_mask=inputs["pixel_mask"], task_inputs=inputs["task_inputs"])
    return instance_inference(inputs, outputs, x)


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, "pytorch_model.bin", revision=revision))]


def to_prediction(m, x, out):
    masks, labels, scores = out
    return Masks(masks.cpu().numpy(), [m["names"][int(i)] for i in labels], scores.cpu().numpy())
