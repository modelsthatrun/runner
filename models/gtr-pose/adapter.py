"""Adapter for GTR-Pose (Gated Token Recurrence), real-time multi-person pose with COCO's 17 keypoints.

GTR has no package. The adapter runs a pinned checkout of the official repository, the way its
tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the image
resized to the config's 640x640 and normalized with ImageNet statistics, and the top 60 people back
with keypoints in the input image's pixels. Its latency script, tools/benchmark/torch_speed.py,
casts the model and input to fp16 (its default), so fp16 is the official precision. torch_inf.py
draws people above its threshold (0.45; its help suggests about 0.3 for pose) and predict() keeps
the same 0.45. There is no keypoint benchmark in the tool yet, so there is no eval_predict().

The README's latencies use an optional fused CUDA GLA kernel (engine/gtr/backbone/csrc), whose
build needs nvcc to match torch's cu128; the reference machine has CUDA 12.4. Without it the backbone uses
flash-linear-attention's Triton kernel, which the README names as the fallback.
"""

import subprocess
import sys

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import weights_dir
from mtr_measure.predictions import Keypoints

REPO_URL = "https://github.com/Intellindust-AI-Lab/GTR.git"
COMMIT = "f8ffd2444bfc2c1adfa70933e28afa37a0377830"
HF_REPO = "Phoenix8125/GTR"
HF_REVISION = "9fc62c8c2b2c976835d0f1c1ffc544dbc0f9e29f"
WEIGHTS = weights_dir("gtr")
REPO = WEIGHTS / "GTR"

TASK = "pose-estimation"
SIZES = ["s", "m", "l", "x"]
PRECISIONS = ["fp16", "fp32"]
INPUT = {s: "640x640" for s in SIZES}
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}
# torch_inf.py's default display threshold.
THRESHOLD = 0.45


def package_version():
    return f"Intellindust-AI-Lab/GTR@{COMMIT}"


def checkout():
    if not (REPO / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def weight_files(size):
    return [WEIGHTS / "pose" / f"gtrpose_{size}_coco.pth"]


def load(size, precision):
    checkout()
    from engine.core import YAMLConfig

    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        from huggingface_hub import hf_hub_download

        hf_hub_download(HF_REPO, f"pose/{ckpt.name}", revision=HF_REVISION, local_dir=WEIGHTS)
    cfg = YAMLConfig(str(REPO / "configs" / "pose" / "coco_pose_finetune" / f"gtrpose_{size}.yml"), resume=str(ckpt))
    for backbone in ("ViTAdapter", "ViTAdapterSpatialSwiGLU"):
        if backbone in cfg.yaml_cfg:
            cfg.yaml_cfg[backbone]["skip_weights_warning"] = True
    checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    dtype = DTYPES[precision]
    h, w = cfg.yaml_cfg["eval_spatial_size"]
    transforms = T.Compose([T.Resize((h, w)), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return {"net": Model().to("cuda", dtype=dtype), "dtype": dtype, "transforms": transforms}


# COCO's keypoint names and skeleton (person_keypoints_val2017.json), 0-based.
NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


@torch.inference_mode()
def predict(m, x):
    w, h = x.size
    images = m["transforms"](x).unsqueeze(0).to("cuda", dtype=m["dtype"])
    scores, labels, keypoints = m["net"](images, torch.tensor([[w, h]], device="cuda"))
    keep = scores[0] > THRESHOLD
    return scores[0][keep].float().cpu(), keypoints[0][keep].float().cpu()


def to_prediction(m, x, out):
    scores, keypoints = out
    return Keypoints(keypoints.numpy(), scores.numpy(), NAMES, SKELETON)
