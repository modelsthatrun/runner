"""Tests for the text-to-image path that need no GPU: the adapter interface, the call's input, the record,
the timings file, the prompt set and the timing prompt."""

import types
from pathlib import Path

import pytest
from PIL import Image

from mtr_measure import adapter as ad
from mtr_measure import generation, machine, main, prompts, protocol, record, submit, timing

ROOT = Path(__file__).resolve().parents[3]


def fake(**over):
    a = types.SimpleNamespace(
        KIND="generation", TASK="text-to-image", SIZES=["4b"], PRECISIONS=["bf16"], PACKAGE="diffusers",
        GENERATION={"4b": {"resolution": "1024x768", "steps": 4, "guidance": 1.0, "generator": "cuda"}},
        load=lambda size, precision: None, predict=lambda model, x: None, weight_files=lambda size: [],
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_a_generation_adapter_declares_its_settings_per_size():
    ad.check_interface(fake())
    with pytest.raises(ad.AdapterError, match="GENERATION has no entry"):
        ad.check_interface(fake(SIZES=["4b", "9b"]))
    with pytest.raises(ad.AdapterError, match="steps"):
        ad.check_interface(fake(GENERATION={"4b": {"resolution": "1024x1024", "steps": 0, "guidance": 1.0, "generator": "cuda"}}))
    with pytest.raises(ad.AdapterError, match="generator"):
        ad.check_interface(fake(GENERATION={"4b": {"resolution": "1024x1024", "steps": 4, "guidance": 1.0, "generator": "mps"}}))
    no_settings = fake()
    del no_settings.GENERATION
    with pytest.raises(ad.AdapterError, match="missing GENERATION"):
        ad.check_interface(no_settings)


def test_the_tool_makes_each_calls_input_and_the_record_says_so():
    a = fake()
    x = generation.make(a, "4b", "a red bicycle", protocol.GEN_SEED)
    assert (x.width, x.height, x.steps, x.guidance, x.seed, x.generator) == (1024, 768, 4, 1.0, 0, "cuda")
    assert ad.input_size(a, "4b", x) == "1024x768"
    assert "denoising" in ad.latency_scope(a)
    model = types.SimpleNamespace(scheduler=type("FlowMatchEulerDiscreteScheduler", (), {})())
    assert generation.recorded(x, generation.scheduler(a, model)) == {
        "steps": 4, "guidance": 1.0, "seed": 0, "generator": "cuda", "scheduler": "FlowMatchEulerDiscreteScheduler"}
    assert generation.scheduler(fake(SCHEDULER="Euler"), object()) == "Euler"
    with pytest.raises(ad.AdapterError, match="SCHEDULER"):
        generation.scheduler(a, object())


def test_predict_must_return_one_image_at_the_requested_size():
    x = generation.make(fake(), "4b", "p", 0)
    assert generation.check_image(Image.new("RGBA", (1024, 768)), x).mode == "RGB"
    with pytest.raises(ad.AdapterError, match="1024x768"):
        generation.check_image(Image.new("RGB", (512, 512)), x)
    with pytest.raises(ad.AdapterError, match="PIL image"):
        generation.check_image([Image.new("RGB", (1024, 768))], x)


def gen_run(times, vram=20000):
    return {"worker": {"times_ms": times, "precision": "bf16", "input": "1024x1024", "latency_scope": generation.LATENCY_SCOPE,
                       "torch_max_reserved_bytes": 0, "weights": [],
                       "env": {"cuda": "13.0", "torch": "2.14.1", "python": "3.12.9", "package": "diffusers==0.42.1"},
                       "generation": {"steps": 4, "guidance": 1.0, "seed": 0, "generator": "cuda", "scheduler": "FlowMatchEulerDiscreteScheduler"}},
            "baseline_bytes": 800 * 2**20, "peak_bytes": (800 + vram) * 2**20}


def test_a_generation_record_and_its_timings_validate_and_can_be_submitted():
    runs = [gen_run([1370.0 + i for i in range(protocol.GEN_TIMED)]) for _ in range(protocol.RUNS)]
    rec = record.build(model="flux-2-klein", size="4b", weights=[{"file": "w.safetensors", "bytes": 1, "sha256": "a" * 64}], runs=runs,
                       gpu_name="NVIDIA GeForce RTX 4090", driver="580", image_digest=None, warmup=protocol.GEN_WARMUP,
                       measured="2026-11-10T10:00:00+09:00", kind="generation",
                       machine={"cpu": "CPU", "gpu_memory_mib": 24564, "compute_capability": "8.9", "os": "Linux", "power_limit_w": 450.0, "power_default_w": 450.0},
                       provenance={"tool": {"name": "mtr", "version": machine.TOOL_VERSION, "commit": "b" * 40, "dirty": False},
                                   "adapter_sha256": "c" * 64, "lock_sha256": "d" * 64, "input_sha256": [timing.sha_for("text-to-image")]})
    assert rec["kind"] == "generation" and rec["measurement"]["generation"]["steps"] == 4
    assert rec["measurement"]["iterations"] == {"warmup": protocol.GEN_WARMUP, "timed": protocol.GEN_TIMED}
    assert record.validate(rec, "gen-run-record") == []
    assert record.validate(rec, "run-record"), "an image run record has no generation block"
    times = record.times(runs, "generation")
    assert times["kind"] == "generation" and record.validate(times, "run-times") == []
    assert record.validate(submit.payload(rec, times, ""), "community-submission") == []
    del rec["measurement"]["generation"]
    assert record.validate(rec, "gen-run-record")


def test_runs_that_made_their_images_differently_cant_be_one_record():
    a, b = gen_run([1000.0] * 10), gen_run([1000.0] * 10)
    b["worker"]["generation"] = {**b["worker"]["generation"], "steps": 8}
    with pytest.raises(ValueError, match="different settings"):
        record.build(model="m", size="s", weights=[{"file": "w", "bytes": 1, "sha256": "a" * 64}], runs=[a, b], gpu_name="g", driver="d",
                     image_digest=None, warmup=2, measured="2026-11-10T10:00:00+09:00", kind="generation")


def test_the_text_to_image_prompt_set_and_timing_prompt():
    s = prompts.load("text-to-image")
    assert s.version >= 1 and s.seed == 0 and len(s.prompts) >= 10
    assert all(p.check for p in s.prompts)
    timing_prompt = timing.input_for("text-to-image").read_text().strip()
    assert timing_prompt not in {p.prompt for p in s.prompts}, "the timing prompt is in no prompt set"
    assert generation.read_prompt([timing.input_for("text-to-image")]) == timing_prompt


def test_generation_models_are_listed_with_their_timing_prompt():
    for d in main.adapters():
        f = d / "adapter.py"
        if ad.kind_of_file(f) == "generation":
            assert ad.task_of_file(f) == "text-to-image"
            assert timing.sha_for("text-to-image")
    assert (ROOT / "models" / "_template-gen" / "adapter.py").is_file()
