"""Tests for `mtr eval --benchmark geneval` that need no GPU and no GenEval environment: the images it makes and
resumes, how it reads GenEval's results, and the record it writes."""

import json
import types
from pathlib import Path

import pytest

from mtr_measure import geneval, protocol, record
from mtr_measure.cli import Log

TAG_PROMPTS = [
    {"tag": "single_object", "include": [{"class": "bench", "count": 1}], "prompt": "a photo of a bench"},
    {"tag": "two_object", "include": [{"class": "toilet", "count": 1}, {"class": "bicycle", "count": 1}], "prompt": "a photo of a toilet and a bicycle"},
    {"tag": "counting", "include": [{"class": "clock", "count": 2}], "exclude": [{"class": "clock", "count": 3}], "prompt": "a photo of two clocks"},
    {"tag": "colors", "include": [{"class": "car", "count": 1, "color": "red"}], "prompt": "a photo of a red car"},
    {"tag": "position", "include": [{"class": "dog", "count": 1}, {"class": "cat", "count": 1, "position": ["left of", 0]}], "prompt": "a photo of a cat left of a dog"},
    {"tag": "color_attr", "include": [{"class": "bus", "count": 1, "color": "blue"}, {"class": "cup", "count": 1, "color": "yellow"}], "prompt": "a photo of a blue bus and a yellow cup"},
]

ADAPTER = '''
from pathlib import Path
from PIL import Image

KIND = "generation"
TASK = "text-to-image"
SIZES = ["s"]
PRECISIONS = ["fp32"]
PACKAGE = "pillow"
GENERATION = {"s": {"resolution": "64x48", "steps": STEPS, "guidance": 1.0, "generator": "cpu"}}
SCHEDULER = "Euler"
CALLS = []


def load(size, precision):
    return object()


def predict(model, x):
    CALLS.append((x.prompt, x.seed))
    return Image.new("RGB", (x.width, x.height), (x.seed, len(x.prompt), 0))


def weight_files(size):
    return [Path(__file__).with_name("w.safetensors")]
'''


def model(tmp_path: Path, steps: int = 4) -> Path:
    d = tmp_path / "models" / "gen"
    d.mkdir(parents=True, exist_ok=True)
    (d / "adapter.py").write_text(ADAPTER.replace("STEPS", str(steps)))
    (d / "w.safetensors").write_bytes(b"weights")
    return d


def results_for(images: Path, rows: list[dict], correct) -> list[dict]:
    return [{"filename": str(geneval.sample_path(images, i, k)), "tag": r["tag"], "prompt": r["prompt"], "correct": correct(i, k),
             "reason": "", "metadata": json.dumps(r), "details": "{}"} for i, r in enumerate(rows) for k in range(4)]


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """geneval.main with GenEval's environment, the GPU and the downloads replaced."""
    base = tmp_path / "geneval-files"
    (base / "prompts").mkdir(parents=True)
    (base / "prompts" / "evaluation_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in TAG_PROMPTS))
    calls = {"scored": 0, "made": 0}
    monkeypatch.setattr(geneval, "ensure_files", lambda say: base)
    monkeypatch.setattr(geneval, "ensure_weights", lambda say: tmp_path / "detector")
    monkeypatch.setattr(geneval, "evaluator_info", lambda log_file: {
        "python": "3.9.21", "torch": "1.12.1+cu113", "cuda": "11.3", "mmcv-full": "1.7.1", "mmdet": "2.28.2",
        "open-clip-torch": "2.20.0", "clip-benchmark": "1.4.0", "nvrtc": "11.8.89 (preloaded)", "arch": "sm_89",
        "config": "mask2former.py", "preload": "libnvrtc.so.11.2"})
    monkeypatch.setattr(geneval, "read_check", lambda d, s: {"approx_vram_mib": 100})
    monkeypatch.setattr(geneval, "free_vram_mib", lambda index: (20000.0, 1000.0))
    monkeypatch.setattr(geneval, "cuda_ready", lambda precision: None)
    monkeypatch.setattr(geneval, "release", lambda: None)
    monkeypatch.setattr(geneval, "torch_env", lambda: {"cuda": "13.0", "torch": "2.14.1"})
    monkeypatch.setattr(geneval, "Gpu", lambda index: types.SimpleNamespace(name=lambda: "NVIDIA GeForce RTX 4090", driver=lambda: "580", close=lambda: None))

    def fake_score(base_, images, results, detector_dir, info, gpu, log_file, log):
        calls["scored"] += 1
        # Every single-object image is right, every other task gets the first two of its four images right.
        n = sum(1 for p in images.iterdir() if p.is_dir())
        rows = [json.loads((images / f"{i:05d}" / "metadata.jsonl").read_text()) for i in range(n)]
        rows_out = results_for(images, rows, lambda i, k: rows[i]["tag"] == "single_object" or k < 2)
        results.write_text("".join(json.dumps(r) + "\n" for r in rows_out))

    monkeypatch.setattr(geneval, "score", fake_score)
    summary = {"off_by": 0.0}

    def fake_summary(base_, results, log_file):
        rows = [json.loads(line) for line in results.read_text().splitlines()]
        return geneval.overall_of_summary(f"Overall score (avg. over tasks): {geneval.summarize(rows)['overall'] + summary['off_by']:.5f}")

    monkeypatch.setattr(geneval, "summary_overall", fake_summary)
    return types.SimpleNamespace(base=base, calls=calls, summary=summary)


def test_geneval_makes_four_seeded_images_per_prompt_scores_them_and_writes_a_record(tmp_path, offline):
    d = model(tmp_path)
    assert geneval.main([str(d), "--size", "s", "--benchmark", "geneval"], Log()) == 0
    rec = json.loads((d / "evals" / "s" / "geneval.json").read_text())
    assert record.validate(rec, "eval-record") == []
    assert rec["kind"] == "generation" and rec["benchmark"]["id"] == "geneval"
    assert rec["benchmark"]["n"] == len(TAG_PROMPTS) * 4 and rec["benchmark"]["data_sha256"] == geneval.FILES["prompts/evaluation_metadata.jsonl"][1]
    # (1 + 5 x 0.5) / 6, on GenEval's own 0-1 scale.
    assert rec["value"] == round(3.5 / 6, 4) and rec["metrics"]["single_object"] == 1.0 and rec["metrics"]["counting"] == 0.5
    s = rec["settings"]
    assert s["seeds"] == [42, 43, 44, 45] and s["images_per_prompt"] == 4
    assert (s["steps"], s["guidance"], s["resolution"], s["generator"], s["scheduler"]) == (4, 1.0, "64x48", "cpu", "Euler")
    assert s["prompt_rewriter"].startswith("none") and s["negative_prompt"] == "none"
    assert s["evaluator"] == f"GenEval {protocol.GENEVAL_COMMIT[:7]}" and s["evaluator_env"]["mmdet"] == "2.28.2"
    assert s["evaluator_env"]["nvrtc"] == "11.8.89 (preloaded)" and "preload" not in s["evaluator_env"]
    images = d / "evals" / "s" / "geneval"
    assert sorted(p.name for p in (images / "00000" / "samples").iterdir()) == ["0000.png", "0001.png", "0002.png", "0003.png"]
    assert json.loads((images / "00003" / "metadata.jsonl").read_text()) == TAG_PROMPTS[3]
    listing = json.loads((d / "evals" / "s" / "geneval.images.json").read_text())
    assert len(listing) == 24 and s["images_sha256"] == record.sha256_file(d / "evals" / "s" / "geneval.images.json")


def test_a_stopped_run_resumes_and_other_settings_are_refused(tmp_path, offline, monkeypatch):
    d = model(tmp_path)
    assert geneval.main([str(d), "--size", "s", "--benchmark", "geneval"], Log()) == 0
    images = d / "evals" / "s" / "geneval"
    geneval.sample_path(images, 2, 3).unlink()
    (images / "00002" / "samples" / ".0001.png.tmp").write_bytes(b"half")
    made = []
    real = geneval.save_png
    monkeypatch.setattr(geneval, "save_png", lambda img, dest: (made.append(dest), real(img, dest)))
    assert geneval.main([str(d), "--size", "s", "--benchmark", "geneval"], Log()) == 0
    assert made == [geneval.sample_path(images, 2, 3)], "only the missing image is made again"
    assert not (images / "00002" / "samples" / ".0001.png.tmp").exists()

    d2 = model(tmp_path, steps=8)
    assert d2 == d
    log = Log()
    assert geneval.main([str(d), "--size", "s", "--benchmark", "geneval"], log) == 2
    assert "made with other settings" in log.text() and "steps" in log.text()


def test_the_tool_and_summary_scores_must_agree(tmp_path, offline):
    offline.summary["off_by"] = 0.01
    log = Log()
    assert geneval.main([str(model(tmp_path)), "--size", "s", "--benchmark", "geneval"], log) == 42
    assert "summary_scores.py says 0.59333, the tool computed 0.58333" in log.text()


def test_a_limited_run_writes_no_record_and_keeps_its_images_apart(tmp_path, offline):
    d = model(tmp_path)
    assert geneval.main([str(d), "--size", "s", "--benchmark", "geneval", "--limit", "2"], Log()) == 0
    assert not (d / "evals" / "s" / "geneval.json").exists()
    assert sorted(p.name for p in (d / "evals" / "s" / "geneval.limit").iterdir() if p.is_dir()) == ["00000", "00001"]
    assert not (d / "evals" / "s" / "geneval").exists()


def test_summarize_is_genevals_mean_over_tasks():
    rows = []
    for tag, correct in (("single_object", [1, 1, 1, 1]), ("two_object", [1, 0, 0, 0]), ("counting", [0, 0, 0, 0]),
                         ("colors", [1, 1, 0, 0]), ("position", [0, 0, 0, 1]), ("color_attr", [1, 1, 1, 0])):
        rows += [{"tag": tag, "correct": bool(c), "metadata": tag} for c in correct]
    s = geneval.summarize(rows)
    assert s["tasks"]["two_object"] == 0.25 and s["overall"] == pytest.approx((1 + 0.25 + 0 + 0.5 + 0.25 + 0.75) / 6)
    assert s["images_correct"] == pytest.approx(11 / 24) and s["prompts_correct"] == pytest.approx(5 / 6)
    with pytest.raises(geneval.MeasureError, match="tags"):
        geneval.summarize([{"tag": "style", "correct": True, "metadata": "x"}])
    assert geneval.overall_of_summary("Task breakdown\n...\nOverall score (avg. over tasks): 0.55123\n") == 0.55123


def test_results_must_cover_every_image_once(tmp_path):
    images = tmp_path / "geneval"
    rows = TAG_PROMPTS[:2]
    good = results_for(images, rows, lambda i, k: True)
    f = tmp_path / "r.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in good))
    assert len(geneval.read_results(f, images, rows, 4)) == 8
    f.write_text("".join(json.dumps(r) + "\n" for r in good[:-1] + [good[0]]))
    with pytest.raises(geneval.MeasureError, match="exactly once"):
        geneval.read_results(f, images, rows, 4)


def test_genevals_files_and_settings_are_pinned():
    assert set(geneval.FILES) >= {"evaluation/evaluate_images.py", "evaluation/object_names.txt", "prompts/evaluation_metadata.jsonl", "LICENSE"}
    assert all(len(sha) == 64 for _, sha in geneval.FILES.values())
    assert geneval.DETECTOR.url.endswith(f"-{geneval.DETECTOR.sha256[:8]}.pth"), "MMDetection names a checkpoint by its sha256"
    assert f"/{geneval.CLIP.sha256}/" in geneval.CLIP.url, "open_clip checks the sha256 the URL names"
    b = protocol.GEN_BENCHMARKS["geneval"]
    assert b["seeds"] == [42, 43, 44, 45] and b["images_per_prompt"] == 4
    lock = (geneval.project() / "uv.lock").read_text()
    for pin in ('name = "torch"\nversion = "1.12.1+cu113"', 'name = "mmcv-full"\nversion = "1.7.1"', 'name = "mmdet"\nversion = "2.28.2"',
                'name = "open-clip-torch"\nversion = "2.20.0"', 'name = "nvidia-cuda-nvrtc-cu11"\nversion = "11.8.89"'):
        assert pin in lock
