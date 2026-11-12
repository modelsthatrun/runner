"""Stable Diffusion 1.5, text-to-image, through diffusers' StableDiffusionPipeline. Size "v1-5" is the v1-5 checkpoint.

Follows the model card (https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5), a mirror of the
removed runwayml/stable-diffusion-v1-5; the card's own id, sd-legacy/stable-diffusion-v1-5, resolves to it.
float16 and pipe(prompt) with the pipeline's defaults, which are 512x512, 50 steps and guidance 7.5. The card
seeds nothing, so the generator is on the CPU. The safety checker stays on, as the card's example loads it: it
is part of every call, and it returns a black image for a picture it flags.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from mtr_measure import device

KIND = "generation"
TASK = "text-to-image"
SIZES = ["v1-5"]
PRECISIONS = ["fp16"]
PACKAGE = "diffusers"
GENERATION = {"v1-5": {"resolution": "512x512", "steps": 50, "guidance": 7.5, "generator": "cpu"}}
REPO = "stable-diffusion-v1-5/stable-diffusion-v1-5"
REVISION = "451f4fe16113bff5a5d2269ed5ad43b0592e9a14"
FILES = [
    "unet/diffusion_pytorch_model.safetensors",
    "text_encoder/model.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "safety_checker/model.safetensors",
]
DTYPES = {"fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision])
    return pipe.to(device())


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
