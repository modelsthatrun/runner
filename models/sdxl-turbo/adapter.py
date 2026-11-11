"""SDXL-Turbo by Stability AI, text-to-image, through diffusers. Size "1.0" is SDXL-Turbo 1.0.

Follows the model card (https://huggingface.co/stabilityai/sdxl-turbo): AutoPipelineForText2Image, float16 with
the fp16 variant of the weights, 1 step and guidance 0.0. The card sets no size, so the image is the pipeline's
default for this checkpoint, 512x512, which the card names as the size the model prefers. It seeds nothing, so
the generator is on the CPU.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["1.0"]
PRECISIONS = ["fp16"]
PACKAGE = "diffusers"
GENERATION = {"1.0": {"resolution": "512x512", "steps": 1, "guidance": 0.0, "generator": "cpu"}}
REPO = "stabilityai/sdxl-turbo"
REVISION = "71153311d3dbb46851df1931d3ca6e939de83304"
FILES = [
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "text_encoder/model.fp16.safetensors",
    "text_encoder_2/model.fp16.safetensors",
    "vae/diffusion_pytorch_model.fp16.safetensors",
]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import AutoPipelineForText2Image

    pipe = AutoPipelineForText2Image.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision], variant="fp16")
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
