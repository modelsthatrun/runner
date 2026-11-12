"""FLUX.2 [klein] by Black Forest Labs, text-to-image, through diffusers' Flux2KleinPipeline.

Size "4b" is FLUX.2 [klein] 4B, distilled to 4 steps; "base-4b" is FLUX.2 [klein] base 4B, the undistilled model.
Follows the model cards (https://huggingface.co/black-forest-labs/FLUX.2-klein-4B and FLUX.2-klein-base-4B):
bfloat16, 1024x1024 and a generator seeded on CUDA, with 4 steps at guidance 1.0 for 4B and 50 steps at
guidance 4.0 for base 4B. The cards also call enable_model_cpu_offload() "to save some VRAM"; every component
fits on a 24 GB GPU, so the pipeline stays on the GPU, as protocol requires.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from mtr_measure import device

KIND = "generation"
TASK = "text-to-image"
SIZES = ["4b", "base-4b"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {
    "4b": {"resolution": "1024x1024", "steps": 4, "guidance": 1.0, "generator": "cuda"},
    "base-4b": {"resolution": "1024x1024", "steps": 50, "guidance": 4.0, "generator": "cuda"},
}
REPOS = {
    "4b": ("black-forest-labs/FLUX.2-klein-4B", "e7b7dc27f91deacad38e78976d1f2b499d76a294"),
    "base-4b": ("black-forest-labs/FLUX.2-klein-base-4B", "a3b4f4849157f664bdbc776fd7453c2783562f4d"),
}
FILES = [
    "transformer/diffusion_pytorch_model.safetensors",
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import Flux2KleinPipeline

    repo, revision = REPOS[size]
    pipe = Flux2KleinPipeline.from_pretrained(repo, revision=revision, torch_dtype=DTYPES[precision])
    return pipe.to(device())


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, guidance_scale=x.guidance, num_inference_steps=x.steps,
                 generator=torch.Generator(device=x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, f, revision=revision)) for f in FILES]
