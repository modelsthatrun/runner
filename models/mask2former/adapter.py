"""Adapter for Mask2Former (Meta AI), instance segmentation on COCO's 80 classes.

The authors' repository (facebookresearch/Mask2Former) runs on detectron2 with compiled CUDA ops. The adapter runs
the same checkpoints as converted to transformers (Mask2FormerForUniversalSegmentation, under facebook/ on Hugging
Face), the way the repository tests them:
- the image resized to a shorter side of 800 and a longer side of at most 1333 and padded to a multiple of 32
  (INPUT.MIN_SIZE_TEST, MAX_SIZE_TEST and SIZE_DIVISIBILITY in its COCO configs). The Hugging Face processor's own
  default is a 384x384 square, and its instance post-processing scales masks back from a fixed 384x384, so both are
  replaced by the repository's.
- instance_inference() from mask2former/maskformer_model.py: mask logits upsampled to the padded input, cropped to
  the image and resized to its own size, then the top 100 (query, class) pairs by class probability
  (TEST.DETECTIONS_PER_IMAGE), each mask the pixels with a positive logit, scored as class probability times the
  mask's mean probability inside it.
It runs in fp32, as detectron2's test does and as the model card does. predict() returns all 100 instances, as the
repository's predictor does; the tool's fixed threshold decides what is drawn.

Swin-B is the ImageNet-21k checkpoint (facebook/mask2former-swin-base-IN21k-coco-instance, MODEL_ZOO.md's "Swin-B
(IN21k)" row). The ImageNet-1k one, facebook/mask2former-swin-base-coco-instance, gives no usable output in either of
its files: its best class probability on the timing photo is 0.66 where the other sizes reach 1.0, and it scores 0.0
mask AP on COCO val2017.
"""

from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from mtr_measure import device
from mtr_measure.predictions import Masks

REPOS = {
    "swin-tiny": ("facebook/mask2former-swin-tiny-coco-instance", "22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6"),
    "swin-small": ("facebook/mask2former-swin-small-coco-instance", "1b7ac418a0f3cb5edfc5037e488d53587b8bb76b"),
    "swin-base-in21k": ("facebook/mask2former-swin-base-IN21k-coco-instance", "9cfc92da70c0ba48132317591f1753f7a694e498"),
    "swin-large": ("facebook/mask2former-swin-large-coco-instance", "d8f2bced294e26108e73d48bb7a7094c9bc7c71d"),
}
# The repository's COCO test settings (configs/coco/instance-segmentation/Base-COCO-InstanceSegmentation.yaml).
TEST_SIZE = {"shortest_edge": 800, "longest_edge": 1333}
SIZE_DIVISOR = 32
TOP_K = 100

TASK = "instance-segmentation"
SIZES = list(REPOS)
PRECISIONS = ["fp32"]
PACKAGE = "transformers"


def load(size, precision):
    repo, revision = REPOS[size]
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision, size=TEST_SIZE, size_divisor=SIZE_DIVISOR)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(repo, revision=revision).to(device()).eval()
    return {"model": model, "processor": processor, "names": model.config.id2label}


def input_size(size, image):
    repo, revision = REPOS[size]
    processor = AutoImageProcessor.from_pretrained(repo, revision=revision, size=TEST_SIZE, size_divisor=SIZE_DIVISOR)
    pixels = processor(images=image, return_tensors="pt")["pixel_values"]
    return f"{pixels.shape[-1]}x{pixels.shape[-2]}"


def instance_inference(inputs, outputs, x):
    """mask2former/maskformer_model.py: the top 100 (query, class) pairs, masks at the image's own size."""
    scores = outputs.class_queries_logits[0].float().softmax(-1)[:, :-1]
    classes = scores.shape[-1]
    top_scores, top = scores.flatten().topk(TOP_K, sorted=False)
    labels = top % classes
    logits = outputs.masks_queries_logits[0, top // classes].float()
    # Upsample to the padded input, crop to the resized image, then resize to the image's own size (sem_seg_postprocess).
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
    inputs = m["processor"](images=x, return_tensors="pt").to(device())
    outputs = m["model"](pixel_values=inputs["pixel_values"], pixel_mask=inputs["pixel_mask"])
    return instance_inference(inputs, outputs, x)


def weight_files(size):
    from huggingface_hub import hf_hub_download

    repo, revision = REPOS[size]
    # The IN21k repository has only pytorch_model.bin, which transformers loads when there is no model.safetensors.
    name = "pytorch_model.bin" if size == "swin-base-in21k" else "model.safetensors"
    return [Path(hf_hub_download(repo, name, revision=revision))]


def to_prediction(m, x, out):
    masks, labels, scores = out
    return Masks(masks.cpu().numpy(), [m["names"][int(i)] for i in labels], scores.cpu().numpy())
