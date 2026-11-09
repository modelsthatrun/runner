"""FLUX.2 [klein] 4B by Black Forest Labs, text-to-image, through diffusers' Flux2KleinPipeline.

Follows the model card (https://huggingface.co/black-forest-labs/FLUX.2-klein-4B): bfloat16, 1024x1024,
4 steps, guidance 1.0, a generator seeded on CUDA. The card also calls enable_model_cpu_offload() "to save
some VRAM"; every component fits on a 24 GB GPU, so the pipeline stays on the GPU, as protocol requires.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["4b"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {"4b": {"resolution": "1024x1024", "steps": 4, "guidance": 1.0, "generator": "cuda"}}
REPO = "black-forest-labs/FLUX.2-klein-4B"
REVISION = "e7b7dc27f91deacad38e78976d1f2b499d76a294"
FILES = [
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision])
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, guidance_scale=x.guidance, num_inference_steps=x.steps,
                 generator=torch.Generator(device=x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
