"""Tests for TensorRT rows (trt.py): file names, the check, and the rule that a runtime row is scored before it is
timed. No GPU and no TensorRT needed."""

import json
from pathlib import Path

import pytest

from mtr_measure import cli, errors, main, paths, protocol, record, runqueue, trt

ENGINE = b"engine bytes"


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake checkout with one model directory, a timing image and a built engine."""
    monkeypatch.setattr(paths, "root", lambda: tmp_path)
    monkeypatch.setattr(paths, "tool_dir", lambda: tmp_path / "tools" / "measure")
    monkeypatch.setenv("MTR_TRT", str(tmp_path / "trt"))
    (tmp_path / "tools" / "measure").mkdir(parents=True)
    model = tmp_path / "models" / "fake-det"
    model.mkdir(parents=True)
    (model / "adapter.py").write_text("SIZES = ['s']\n")
    (model / "pyproject.toml").write_text("")
    (model / "uv.lock").write_text("lock 1")
    (tmp_path / "timing.jpg").write_bytes(b"jpg")
    engine = trt.engine_dir("fake-det", "s") / "model.engine"
    engine.parent.mkdir(parents=True)
    engine.write_bytes(ENGINE)
    return tmp_path


def engine_sha(repo: Path) -> str:
    return record.sha256_file(trt.engine_dir("fake-det", "s") / "model.engine")


def runtime_info(sha: str) -> dict:
    return {
        "name": "tensorrt", "version": "10.16.1.11", "status": "runs-as-is", "patch": None,
        "export": {"exporter": "Org/repo@abc tools/export.py", "args": ["-r", "w.pth"], "opset": 18, "onnx_sha256": "b" * 64, "onnx_bytes": 10},
        "engine": {"sha256": sha, "bytes": len(ENGINE), "builder": {"flags": ["FP16"]}, "built": "2026-11-10T22:00:00+09:00", "build_s": 60.0},
    }


def trt_check(repo: Path, **over) -> dict:
    model = repo / "models" / "fake-det"
    sha = engine_sha(repo)
    check = {"ok": True, "size": "s", "protocol": protocol.PROTOCOL_VERSION, **record.fingerprint(model), "load_s": 1.0,
             "approx_ms": 5.0, "weight_files": [], "runtime": "tensorrt", "engine": str(trt.engine_dir("fake-det", "s") / "model.engine"),
             "runtime_info": runtime_info(sha), **over}
    path = record.check_path(model, "s", "tensorrt")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(check))
    return check


def write_eval(repo: Path, sha: str, name: str = "coco-val2017-bbox.tensorrt.json") -> Path:
    d = repo / "models" / "fake-det" / "evals" / "s"
    d.mkdir(parents=True, exist_ok=True)
    f = d / name
    f.write_text(json.dumps({"kind": "image", "runtime": {"name": "tensorrt", "version": "10.16.1.11", "engine_sha256": sha}}))
    return f


def result(capsys) -> dict:
    return errors.parse_result(capsys.readouterr().out)


def test_file_names_carry_the_runtime(tmp_path):
    assert trt.suffix("pytorch-cuda") == "" and trt.suffix("tensorrt") == ".tensorrt"
    assert record.check_path(tmp_path, "s").name == "s.json"
    assert record.check_path(tmp_path, "s", "tensorrt").name == "s.tensorrt.json"


def test_queue_refuses_a_tensorrt_row_until_its_engine_is_scored(repo, capsys):
    model = repo / "models" / "fake-det"
    args = ["queue", "add", str(model), "--size", "s", "--runtime", "tensorrt"]
    assert cli.main(args) == errors.exit_code("NOT_CHECKED")  # no TensorRT check yet
    trt_check(repo)
    assert cli.main(args) == errors.exit_code("NOT_SCORED")
    assert "No eval record" in result(capsys)["message"]

    # An eval of another engine, or a predictions file next to the record, doesn't count.
    write_eval(repo, "c" * 64)
    write_eval(repo, engine_sha(repo), "coco-val2017-bbox.tensorrt.predictions.json")
    assert cli.main(args) == errors.exit_code("NOT_SCORED")
    capsys.readouterr()

    write_eval(repo, engine_sha(repo))
    assert cli.main(args) == 0
    items = json.loads((repo / "tools" / "measure" / "queue.json").read_text())
    assert items[0]["runtime"] == "tensorrt"


def test_a_rebuilt_engine_needs_a_new_check_and_eval(repo, capsys):
    model = repo / "models" / "fake-det"
    trt_check(repo)
    write_eval(repo, engine_sha(repo))
    (trt.engine_dir("fake-det", "s") / "model.engine").write_bytes(b"other bytes")
    assert cli.main(["queue", "add", str(model), "--size", "s", "--runtime", "tensorrt"]) == errors.exit_code("NOT_CHECKED")
    assert "changed since it was checked" in result(capsys)["message"]


def test_pytorch_and_tensorrt_rows_are_separate_queue_items(repo, capsys):
    model = repo / "models" / "fake-det"
    pt = {"ok": True, "size": "s", "protocol": protocol.PROTOCOL_VERSION, **record.fingerprint(model), "load_s": 1.0, "approx_ms": 5.0, "weight_files": []}
    record.check_path(model, "s").parent.mkdir(parents=True, exist_ok=True)
    record.check_path(model, "s").write_text(json.dumps(pt))
    trt_check(repo)
    write_eval(repo, engine_sha(repo))
    # A queue written before runtimes existed: its items have no runtime key.
    (repo / "tools" / "measure" / "queue.json").write_text(json.dumps([{"n": 1, "model_dir": "models/fake-det", "size": "s", "inputs": [],
                                                                         "note": "", "estimate_s": 30, "added": "x", "last_error": None}]))
    assert cli.main(["queue", "add", str(model), "--size", "s"]) == 0
    assert result(capsys)["message"].endswith("already queued")
    assert cli.main(["queue", "add", str(model), "--size", "s", "--runtime", "tensorrt"]) == 0
    assert result(capsys)["outputs"]["queued"] == 2


def test_queue_run_passes_the_runtime_and_checks_the_eval_again(repo, capsys, monkeypatch):
    model = repo / "models" / "fake-det"
    trt_check(repo)
    write_eval(repo, engine_sha(repo))
    assert cli.main(["queue", "add", str(model), "--size", "s", "--runtime", "tensorrt"]) == 0
    capsys.readouterr()
    seen = []
    monkeypatch.setattr(runqueue, "run_one", lambda item: seen.append(item) or {"status": "ok", "code": "OK", "message": "done"})
    monkeypatch.setattr(runqueue.desktop, "paused", lambda log: __import__("contextlib").nullcontext())
    monkeypatch.setattr(runqueue.desktop, "extend_fallback", lambda expected_s=0: None)
    # The eval goes away after queueing: the run is refused, and the item stays.
    for f in (model / "evals" / "s").glob("*.json"):
        f.unlink()
    assert cli.main(["queue", "run"]) == errors.exit_code("QUEUE")
    assert not seen and result(capsys)["outputs"]["results"][0]["code"] == "NOT_SCORED"
    write_eval(repo, engine_sha(repo))
    assert cli.main(["queue", "run"]) == 0
    assert seen[0]["runtime"] == "tensorrt"


def test_run_one_and_the_model_command_ask_for_the_tensorrt_group(repo, monkeypatch):
    calls = []

    class Proc:
        returncode = 0

        def __init__(self):
            self.stdout = iter(['mtr-result: {"status": "ok", "code": "OK", "message": ""}\n'])

        def wait(self):
            return 0

    monkeypatch.setattr(runqueue.subprocess, "Popen", lambda cmd, **kw: calls.append(cmd) or Proc())
    runqueue.run_one({"model_dir": "models/fake-det", "size": "s", "inputs": [], "runtime": "tensorrt"})
    assert calls[0][-2:] == ["--runtime", "tensorrt"]
    runqueue.run_one({"model_dir": "models/fake-det", "size": "s", "inputs": []})
    assert "--runtime" not in calls[1]

    monkeypatch.setattr(main, "_uv", lambda: "uv")
    model = repo / "models" / "fake-det"
    assert main.model_command("run", model, ["--size", "s", "--runtime", "tensorrt"])[:7] == \
        ["uv", "run", "--project", str(model), "--frozen", "--group", "tensorrt"]
    assert "--group" not in main.model_command("run", model, ["--size", "s"])


def test_the_runtime_block_goes_into_the_record_and_passes_the_schema(repo):
    check = trt_check(repo)
    worker = {"precision": "fp16", "input": "640x640", "latency_scope": "predict()", "times_ms": [5.0, 5.2, 4.9],
              "torch_max_reserved_bytes": 2**20, "env": {"cuda": "13.0", "torch": "2.14.1", "python": "3.12.9", "package": "repo@abc"}}
    runs = [{"worker": worker, "baseline_bytes": 500 * 2**20, "peak_bytes": 900 * 2**20}]
    rec = record.build(model="fake-det", size="s", weights=[{"file": "w.pth", "bytes": 1, "sha256": "a" * 64}], runs=runs,
                       gpu_name="NVIDIA GeForce RTX 4090", driver="580", image_digest=None, warmup=10, measured="2026-11-10T22:00:00+09:00",
                       runtime=trt.runtime_block(check))
    m = rec["measurement"]
    assert m["runtime"]["engine"]["sha256"] == engine_sha(repo) and "patch" not in m["runtime"]
    assert m["torch_max_reserved_mib"] is None  # TensorRT allocates outside PyTorch
    assert record.validate(rec, "run-record") == []
    ev = {"name": "tensorrt", "version": "10.16.1.11", "engine_sha256": engine_sha(repo)}
    assert trt.eval_runtime(check) == ev


def test_only_image_models_take_a_runtime(repo, capsys):
    model = repo / "models" / "fake-llm"
    model.mkdir()
    (model / "adapter.py").write_text('KIND = "llm"\nSIZES = ["q4"]\n')
    assert cli.main(["run", str(model), "--size", "q4", "--runtime", "tensorrt", "--dry-run"]) == errors.exit_code("USAGE")
    assert "image models only" in result(capsys)["message"]


def test_adapter_spec(monkeypatch):
    class A:
        TENSORRT = {"precision": "fp16", "builder": {"fp16": True}}

        @staticmethod
        def export_onnx(size, out): ...

        @staticmethod
        def load_engine(size, engine): ...

    assert trt.adapter_spec(A) == {"precision": "fp16", "builder": {"fp16": True}, "status": "runs-as-is", "patch": None}
    assert trt.builder_settings(trt.adapter_spec(A))["flags"] == ["FP16"]
    A.TENSORRT = {"precision": "fp16", "builder": {"workspace": 4}}
    with pytest.raises(errors.MeasureError, match="only the flags"):
        trt.adapter_spec(A)
    A.TENSORRT = {"precision": "fp16", "builder": {}, "status": "runs-with-fixes"}
    with pytest.raises(errors.MeasureError, match="patch"):
        trt.adapter_spec(A)

    class NoRow:
        pass

    with pytest.raises(errors.MeasureError, match="no TensorRT row"):
        trt.adapter_spec(NoRow)


EVAL_ADAPTER = '''
import numpy as np
from mtr_measure.predictions import Boxes

TASK = "object-detection"
SIZES = ["s"]
PRECISIONS = ["fp32"]
PACKAGE = "pytest"
INPUT = {"s": "64x48"}
TENSORRT = {"precision": "fp16", "builder": {"fp16": True}}

def load(size, precision):
    return {"precision": precision}

def export_onnx(size, out): ...

def load_engine(size, engine):
    return {"engine": engine}

def predict(model, image):
    return {"boxes": np.array([[4, 4, 32, 24]]), "scores": np.array([0.9])}

def to_prediction(model, image, out):
    return Boxes(out["boxes"], out["scores"], ["cat"])

def weight_files(size):
    return []
'''


@pytest.mark.parametrize("runtime", ["pytorch-cuda", "tensorrt"])
def test_an_image_eval_scores_and_names_its_files_by_runtime(repo, monkeypatch, capsys, runtime):
    """A one-image COCO stand-in, scored with --limit: no GPU, no record, but every step up to the score."""
    from PIL import Image

    from mtr_measure import eval as evaluate

    model = repo / "models" / "fake-det"
    (model / "adapter.py").write_text(EVAL_ADAPTER)
    if runtime == "tensorrt":
        trt_check(repo)
    else:
        pt = {"ok": True, "size": "s", "protocol": protocol.PROTOCOL_VERSION, **record.fingerprint(model), "load_s": 1.0,
              "approx_ms": 5.0, "approx_vram_mib": 100, "weight_files": []}
        record.check_path(model, "s").parent.mkdir(parents=True, exist_ok=True)
        record.check_path(model, "s").write_text(json.dumps(pt))
    coco = repo / "coco"
    (coco / "annotations").mkdir(parents=True)
    (coco / "val2017").mkdir()
    Image.new("RGB", (64, 48)).save(coco / "val2017" / "a.jpg")
    (coco / "annotations" / "instances_val2017.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "a.jpg", "width": 64, "height": 48}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 17, "bbox": [4, 4, 28, 20], "area": 560, "iscrowd": 0}],
        "categories": [{"id": 17, "name": "cat"}, {"id": 18, "name": "dog"}],
    }))
    monkeypatch.setattr(evaluate.machine, "pin_gpu", lambda index: None)
    monkeypatch.setattr(evaluate, "free_vram_mib", lambda index: (20000.0, 0.0))
    monkeypatch.setattr(evaluate.datasets, "ensure", lambda d, say=print: coco)

    assert evaluate.main([str(model), "--size", "s", "--benchmark", "coco-val2017-bbox", "--limit", "1", "--runtime", runtime],
                         cli.Log()) == 0
    out = result(capsys)
    assert out["outputs"]["value"] == 100.0
    sfx = trt.suffix(runtime)
    assert (model / "evals" / "s" / f"coco-val2017-bbox{sfx}.limit.predictions.json").is_file()
    assert (model / "evals" / "s" / f"coco-val2017-bbox{sfx}.limit.log").is_file()


def test_a_tensorrt_run_writes_its_record_and_times_under_the_runtime_name(repo, monkeypatch, capsys):
    """`mtr run --runtime tensorrt` end to end on the CPU: the GPU, the CPU window and the worker are stand-ins."""
    from types import SimpleNamespace

    model = repo / "models" / "fake-det"
    (model / "adapter.py").write_text(EVAL_ADAPTER)
    weight = model / "w.pth"
    weight.write_bytes(b"weights")
    check = trt_check(repo)
    MIB = 2**20

    class FakeGpu:
        def __init__(self, index):
            pass

        def name(self):
            return "NVIDIA GeForce RTX 4090"

        def driver(self):
            return "580.173.02"

        def processes(self):
            return []

        def close(self):
            pass

    class Window:
        def start(self):
            return self

        def others(self):
            return 0.01

    ok = SimpleNamespace(ok=lambda: True, warnings=[], codes=[], errors=[], lines=[])
    specs = []

    def spawn_worker(spec, gpu, poll_ms, index, module):
        specs.append(spec)
        worker = {"precision": "fp16", "runtime": "tensorrt", "input": "640x640", "latency_scope": "predict()",
                  "times_ms": [5.0 + i / 1000 for i in range(spec["timed"])], "torch_max_reserved_bytes": 2 * MIB,
                  "weights": [str(weight)], "weight_links": [str(weight)],
                  "env": {"cuda": "13.0", "torch": "2.14.1", "python": "3.12.9", "package": "pytest"},
                  "load_s": 1.0, "output": "Boxes", "device": "NVIDIA GeForce RTX 4090"}
        return worker, SimpleNamespace(peak_bytes=900 * MIB)

    monkeypatch.setattr(cli, "Gpu", FakeGpu)
    monkeypatch.setattr(cli.cpu, "Window", Window)
    monkeypatch.setattr(cli, "sample_for", lambda gpu, seconds, poll_ms: SimpleNamespace(median_bytes=500 * MIB))
    monkeypatch.setattr(cli.checks, "before_run", lambda *a: ok)
    monkeypatch.setattr(cli.checks, "after_run", lambda *a: ok)
    monkeypatch.setattr(cli, "spawn_worker", spawn_worker)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    monkeypatch.setattr(cli.desktop, "paused", lambda log: __import__("contextlib").nullcontext())
    monkeypatch.setattr(cli.machine, "facts", lambda gpu: {"cpu": "Test CPU", "gpu_memory_mib": 24564, "compute_capability": "8.9",
                                                           "os": "Linux", "power_limit_w": 450, "power_default_w": 450})
    args = ["run", str(model), "--size", "s", "--input", str(repo / "timing.jpg"), "--runtime", "tensorrt"]

    # Not scored yet: refused before anything runs.
    assert cli.main(args) == errors.exit_code("NOT_SCORED")
    assert not specs
    capsys.readouterr()

    write_eval(repo, engine_sha(repo))
    assert cli.main(args) == 0, result(capsys)
    assert all(s["runtime"] == "tensorrt" and s["engine"] == check["engine"] for s in specs) and len(specs) == protocol.RUNS
    runs = model / "runs"
    # The refused attempt left its failed log, under the runtime's name too.
    assert sorted(p.name for p in runs.iterdir()) == ["s.tensorrt.failed.log", "s.tensorrt.json", "s.tensorrt.log", "s.tensorrt.times.json"]
    rec = json.loads((runs / "s.tensorrt.json").read_text())
    assert rec["measurement"]["runtime"]["engine"]["sha256"] == engine_sha(repo)
    assert rec["measurement"]["precision"] == "fp16" and rec["measurement"]["torch_max_reserved_mib"] is None
    assert record.validate(rec, "run-record") == []
    assert record.validate(json.loads((runs / "s.tensorrt.times.json").read_text()), "run-times") == []
    # The PyTorch row's files are left alone.
    assert not (runs / "s.json").exists()


PACKAGE_BUILD_ADAPTER = '''
import os
from pathlib import Path

SIZES = ["s"]
PRECISIONS = ["fp32"]
PACKAGE = "pytest"
INPUT = {"s": "640x448"}
TENSORRT = {"precision": "fp16", "builder": {"fp16": True}, "input": "640x640"}
def load(size, precision): ...

def predict(model, image): ...

def weight_files(size):
    return [Path(__file__).parent / "w.pt"]

def export_engine(size, out):
    (out / "w.onnx").write_bytes(b"onnx")
    (out / "w.engine").write_bytes(os.urandom(16))  # TensorRT times tactics, so every build gives other bytes
    return {"engine": out / "w.engine", "onnx": out / "w.onnx", "exporter": "pytest YOLO.export", "args": ["format=engine"],
            "builder": {"input": "1x3x640x640, static"}}

def load_engine(size, engine): ...
'''


def test_a_package_that_builds_its_own_engine(repo, monkeypatch):
    """export_engine(): the package's one call exports and builds, the tool keeps and records what it made."""
    from mtr_measure import adapter as ad

    model = repo / "models" / "fake-det"
    (model / "adapter.py").write_text(PACKAGE_BUILD_ADAPTER)
    (model / "w.pt").write_bytes(b"weights")
    monkeypatch.setattr(trt, "trt_version", lambda: "10.16.1.11")
    monkeypatch.setattr(ad, "package", lambda a: "pytest==1.0")
    spec = {"adapter": str(model / "adapter.py"), "size": "s", "model": "fake-det", "gpu": "NVIDIA GeForce RTX 4090"}
    info = trt.prepare(spec, say=lambda s: None)
    rt = info["runtime"]
    assert not info["reused"] and Path(info["engine"]["path"]).name == "w.engine"
    assert rt["export"]["exporter"] == "pytest YOLO.export" and rt["engine"]["sha256"] == record.sha256_file(Path(info["engine"]["path"]))
    assert rt["engine"]["builder"] == {"api": "pytest==1.0, which builds with TensorRT's Python API", "flags": ["FP16"],
                                       "other": "the package's own settings", "input": "1x3x640x640, static"}
    # Kept on the next check, built again only with --rebuild.
    assert trt.prepare(spec, say=lambda s: None)["reused"]
    again = trt.prepare({**spec, "rebuild": True}, say=lambda s: None)
    assert again["engine"]["sha256"] != info["engine"]["sha256"]

    # The run records the engine's own input, not the PyTorch row's.
    a = ad.load_adapter(model / "adapter.py")
    assert ad.input_size(a, "s", None, "tensorrt") == "640x640" and ad.input_size(a, "s", None) == "640x448"

    class Both:
        TENSORRT = {"precision": "fp16", "builder": {}}
        export_onnx = export_engine = load_engine = staticmethod(lambda *a: None)

    with pytest.raises(errors.MeasureError, match="not both"):
        trt.adapter_spec(Both)
