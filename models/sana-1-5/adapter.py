"""Sana 1.5 by NVIDIA, text-to-image, through diffusers' SanaPipeline. Sizes "1.6b" and "4.8b" are its two transformers.

Follows the model cards (https://huggingface.co/Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers and
SANA1.5_4.8B_1024px_diffusers): bfloat16, the text encoder cast to bfloat16 again after the move to the GPU,
1024x1024, 20 steps and guidance 4.5. The cards seed nothing, so the generator is on the CPU. The pipeline's
default complex_human_instruction, a fixed instruction it puts before every prompt, stays as it is.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

from mtr_measure import device

KIND = "generation"
TASK = "text-to-image"
SIZES = ["1.6b", "4.8b"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {size: {"resolution": "1024x1024", "steps": 20, "guidance": 4.5, "generator": "cpu"} for size in SIZES}
REPOS = {
    "1.6b": ("Efficient-Large-Model/SANA1.5_1.6B_1024px_diffusers", "b77948f2b4eed5c728e9b828ccff07f7427b43cc"),
    "4.8b": ("Efficient-Large-Model/SANA1.5_4.8B_1024px_diffusers", "9468102c3cebb657f8c4b5f1e5a71e989a15f10d"),
}
TRANSFORMERS = {
    "1.6b": ["transformer/diffusion_pytorch_model.safetensors"],
    "4.8b": [f"transformer/diffusion_pytorch_model-0000{i}-of-00002.safetensors" for i in (1, 2)],
}
SHARED = [
    "text_encoder/model-00001-of-00002.safetensors",
    "text_encoder/model-00002-of-00002.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import SanaPipeline

    repo, revision = REPOS[size]
    pipe = SanaPipeline.from_pretrained(repo, revision=revision, torch_dtype=DTYPES[precision])
    pipe.to(device())
    pipe.text_encoder.to(DTYPES[precision])
    return pipe


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed)).images[0]


def weight_files(size: str) -> list[Path]:
    repo, revision = REPOS[size]
    return [Path(hf_hub_download(repo, f, revision=revision)) for f in (*TRANSFORMERS[size], *SHARED)]
