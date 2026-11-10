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
    monkeypatch.setattr(runqueue.desktop, "extend_fallback", lambda: None)
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
