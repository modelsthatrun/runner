"""Lumina-Image 2.0 by Alpha-VLLM, text-to-image, through diffusers' Lumina2Pipeline. Size "2.0" is its one released checkpoint.

Follows the model card (https://huggingface.co/Alpha-VLLM/Lumina-Image-2.0): bfloat16, 1024x1024, 50 steps,
guidance 4.0, cfg_trunc_ratio 0.25 and cfg_normalization on, a generator seeded on the CPU. The card calls
enable_model_cpu_offload() to save VRAM and says to remove it with enough GPU memory; every component fits on
a 24 GB GPU, so the pipeline stays on the GPU, as protocol requires. The pipeline's default system prompt, a
fixed instruction it puts before every prompt, stays as it is. The weights are stored in float32 and loaded
as bfloat16, as the card does.
"""

from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

KIND = "generation"
TASK = "text-to-image"
SIZES = ["2.0"]
PRECISIONS = ["bf16"]
PACKAGE = "diffusers"
GENERATION = {"2.0": {"resolution": "1024x1024", "steps": 50, "guidance": 4.0, "generator": "cpu"}}
# The card's other two arguments, which GENERATION has no field for.
CFG = {"cfg_trunc_ratio": 0.25, "cfg_normalization": True}
REPO = "Alpha-VLLM/Lumina-Image-2.0"
REVISION = "53504abd8178b30685b6c4c7a4cd181ff78b73e9"
FILES = [
    *(f"transformer/diffusion_pytorch_model-0000{i}-of-00002.safetensors" for i in (1, 2)),
    *(f"text_encoder/model-0000{i}-of-00003.safetensors" for i in (1, 2, 3)),
    "vae/diffusion_pytorch_model.safetensors",
]
DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(size: str, precision: str):
    from diffusers import Lumina2Pipeline

    pipe = Lumina2Pipeline.from_pretrained(REPO, revision=REVISION, torch_dtype=DTYPES[precision])
    return pipe.to("cuda")


@torch.inference_mode()
def predict(model, x):
    return model(prompt=x.prompt, height=x.height, width=x.width, num_inference_steps=x.steps, guidance_scale=x.guidance,
                 generator=torch.Generator(x.generator).manual_seed(x.seed), **CFG).images[0]


def weight_files(size: str) -> list[Path]:
    return [Path(hf_hub_download(REPO, f, revision=REVISION)) for f in FILES]
