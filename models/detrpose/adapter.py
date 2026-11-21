"""Adapter for DETRPose, real-time multi-person 2D pose estimation with COCO's 17 keypoints.

DETRPose has no package. The adapter runs a pinned checkout of the official repository, the way its
tools/inference/torch_inf.py does: the config's model and postprocessor in deploy mode, the image resized to 640x640
with torchvision, and the top 60 people (PostProcess num_select) with their keypoints in the input image's pixels.
That script keeps people above a score of 0.5 when it draws (`thrh`). It runs in fp32: the repository's FP16 is
TensorRT's, so PRECISIONS is fp32. Only the COCO-trained checkpoints are listed, not the CrowdPose ones (14 keypoints).
The backbone's ImageNet weights are switched off, as the script does, because the checkpoint replaces them.

The model gives each person a score and no score per keypoint, so every keypoint is drawn, as the repository's
annotator draws them.

Its ONNX Runtime row on the CPU follows the README's "Export onnx" and "Inference" steps:
tools/deployment/export_onnx.py on the released checkpoint, then tools/inference/onnx_inf.py's process_image(): the same
640x640 resize, and a default InferenceSession.
"""

import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T

from mtr_measure import device, weights_dir
from mtr_measure.predictions import Keypoints

REPO_URL = "https://github.com/SebastianJanampa/DETRPose.git"
COMMIT = "d56a05012e98e83d1594e53d1826b9cf42a5db8a"
WEIGHTS = weights_dir("detrpose")
REPO = WEIGHTS / "DETRPose"
RELEASE = "https://github.com/SebastianJanampa/DETRPose/releases/download/model_weights"

TASK = "pose-estimation"
SIZES = ["n", "s", "m", "l", "x"]
PRECISIONS = ["fp32"]
INPUT = {s: "640x640" for s in SIZES}
# tools/inference/torch_inf.py and onnx_inf.py draw people above 0.5.
THRESHOLD = 0.5
ONNXRUNTIME = {"status": "runs-as-is", "input": "640x640"}

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow",
         "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
SKELETON = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
            (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]


def package_version():
    return f"SebastianJanampa/DETRPose@{COMMIT}"


def checkout():
    if not (REPO / ".git").is_dir():
        subprocess.run(["git", "clone", "-q", REPO_URL, str(REPO)], check=True)
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    if head != COMMIT:
        subprocess.run(["git", "-C", str(REPO), "fetch", "-q", "origin"], check=True)
        subprocess.run(["git", "-C", str(REPO), "checkout", "-q", COMMIT], check=True)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))


def config(size):
    return f"configs/detrpose/detrpose_hgnetv2_{size}.py"


def checkpoint_file(size):
    ckpt = weight_files(size)[0]
    if not ckpt.is_file():
        torch.hub.download_url_to_file(f"{RELEASE}/{ckpt.name}", str(ckpt))
    return ckpt


def load(size, precision):
    checkout()
    from src.core import LazyConfig, instantiate

    cfg = LazyConfig.load(str(REPO / config(size)))
    if hasattr(cfg.model.backbone, "pretrained"):
        cfg.model.backbone.pretrained = False
    model = instantiate(cfg.model)
    postprocessor = instantiate(cfg.postprocessor)
    checkpoint = torch.load(checkpoint_file(size), map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.deploy()
            self.postprocessor = postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return {"net": Model().to(device()), "transforms": T.Compose([T.Resize((640, 640)), T.ToTensor()])}


def keep(scores, keypoints):
    scores, keypoints = scores[0].cpu(), keypoints[0].cpu()
    k = scores > THRESHOLD
    return scores[k], keypoints[k]


# predict() in the three phases `mtr profile` times apart: the tensor on the GPU, the network, and the decoded result.
@torch.inference_mode()
def preprocess(m, x):
    return m["transforms"](x).unsqueeze(0).to(device())


@torch.inference_mode()
def forward(m, images):
    return m["net"].model(images)


@torch.inference_mode()
def postprocess(m, raw, x):
    w, h = x.size
    scores, labels, keypoints = m["net"].postprocessor(raw, torch.tensor([[w, h]], device=device()))
    return keep(scores, keypoints)


def export_onnx(size, out):
    """The README's export, run as it is on the released checkpoint. It writes onnx_engines/<config name>.onnx under the
    directory it runs in, so it runs in `out`, with the checkout's files on its path."""
    checkout()
    ckpt = checkpoint_file(size)
    args = ["--check", "-c", config(size), "-r", ckpt.name]
    out = Path(out)
    (out / "configs").symlink_to(REPO / "configs")
    subprocess.run([sys.executable, str(REPO / "tools" / "deployment" / "export_onnx.py"), *args[:-1], str(ckpt)], cwd=out, check=True)
    return {"onnx": out / "onnx_engines" / f"detrpose_hgnetv2_{size}.onnx",
            "exporter": f"SebastianJanampa/DETRPose@{COMMIT} tools/deployment/export_onnx.py", "args": args}


def load_onnx(size, onnx):
    """The ONNX file in a session as tools/inference/onnx_inf.py's main() makes it: ort.InferenceSession(args.onnx)."""
    import onnxruntime as ort

    return {"sess": ort.InferenceSession(str(onnx)), "transforms": T.Compose([T.Resize((640, 640)), T.ToTensor()])}


def onnx_session(m):
    return m["sess"]


@torch.inference_mode()
def predict(m, x):
    if "sess" in m:
        # onnx_inf.py's process_image(): the resized image and the photo's size into the session.
        w, h = x.size
        scores, labels, keypoints = m["sess"].run(output_names=None, input_feed={
            "images": m["transforms"](x).unsqueeze(0).numpy(), "orig_target_sizes": torch.tensor([[w, h]]).numpy()})
        return keep(torch.from_numpy(scores), torch.from_numpy(keypoints))
    return postprocess(m, forward(m, preprocess(m, x)), x)


def weight_files(size):
    return [WEIGHTS / f"detrpose_hgnetv2_{size}.pth"]


def to_prediction(m, x, out):
    scores, keypoints = out
    return Keypoints(keypoints.numpy(), scores.numpy(), NAMES, SKELETON)
