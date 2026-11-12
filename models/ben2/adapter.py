"""Adapter for BEN2 Base (Prama LLC), background removal: one foreground mask per image.

The model card loads BEN_Base.from_pretrained("PramaLLC/BEN2") from the ben2 package (installed
from its git repository) and calls model.inference(image, refine_foreground=False), which resizes
to 1024x1024, runs the network and returns the image with the predicted alpha. On CUDA the package
always runs in fp16 (an fp16 input and an autocast decorator on forward()), so PRECISIONS is fp16.

inference() puts the alpha onto the image it was given, in place. predict() hands it a copy, so
every timed call starts from the same RGB image, as a call on a freshly opened file would.

Its ONNX Runtime row on the CPU runs the ONNX file the authors publish in the same repository at the same revision,
BEN2_Base.onnx, the way their onnx_run.py there does: a session with the CUDA provider and then the CPU provider (only
the CPU's exists here), the image resized to 1024x1024 with ToTensor and no normalization, the output resized back,
scaled to 0-1 by its own minimum and maximum, and put on the image as its alpha. onnx_run.py runs an example when it
is imported, so its steps are copied here. Nothing ties that file to model.safetensors by hash, so the row stands only
if its output on the timing photo matches the PyTorch row's.
"""

from pathlib import Path

import numpy as np
import torch
from ben2 import BEN_Base

from mtr_measure import device
from mtr_measure.predictions import Alpha

REPO, REVISION = "PramaLLC/BEN2", "e48a20765fb421d19dcdb0bf3cc61e802ca5ec8f"
COMMIT = "2c99a5da477b5523585bfa5c893888a6e818a8f6"

TASK = "image-matting"
SIZES = ["base"]
PRECISIONS = ["fp16"]
INPUT = {"base": "1024x1024"}
ONNXRUNTIME = {"status": "runs-as-is"}


def package_version():
    return f"PramaLLC/BEN2@{COMMIT}"


def load(size, precision):
    model = BEN_Base.from_pretrained(REPO, revision=REVISION)
    return model.to(device()).eval()


@torch.no_grad()
def predict(model, x):
    if isinstance(model, dict):
        return ort_inference(model, x.copy())
    return model.inference(x.copy(), refine_foreground=False)


def onnx_file(size, out):
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(REPO, "BEN2_Base.onnx", revision=REVISION))
    return {"onnx": path, "exporter": f"huggingface.co/{REPO}@{REVISION} BEN2_Base.onnx, published by the authors", "args": []}


def load_onnx(size, onnx):
    import onnxruntime

    # onnx_run.py's session.
    return {"sess": onnxruntime.InferenceSession(str(onnx), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])}


def onnx_session(m):
    return m["sess"]


def ort_inference(m, image):
    """onnx_run.py's run_inference(): preprocess_image(), the session, postprocess_image(), then the alpha on the image."""
    import torch.nn.functional as F
    import torchvision.transforms as transforms
    from PIL import Image

    w, h = image.size
    input_data = transforms.Compose([transforms.Resize((1024, 1024)), transforms.ToTensor()])(image).unsqueeze(0).numpy()
    outputs = m["sess"].run(None, {m["sess"].get_inputs()[0].name: input_data})
    result = torch.from_numpy(outputs[0])
    if len(result.shape) == 3:
        result = result.unsqueeze(0)
    result = torch.squeeze(F.interpolate(result, size=[w, h], mode="bilinear"), 0)
    ma, mi = torch.max(result), torch.min(result)
    result = (result - mi) / (ma - mi)
    alpha = np.squeeze((result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
    image.putalpha(Image.fromarray(alpha).resize((w, h)))
    return image


def weight_files(size):
    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(REPO, "model.safetensors", revision=REVISION))]


def to_prediction(model, x, out):
    return Alpha(np.asarray(out.getchannel("A"), dtype=np.float32) / 255)
