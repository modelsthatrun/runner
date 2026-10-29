"""Tests for the queue of timed runs and for `new` and `clean`. No GPU needed."""

import json
from pathlib import Path

import pytest

from mtr_measure import cli, errors, manage, paths, protocol, record, runqueue


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A fake checkout with one model directory and a timing image."""
    monkeypatch.setattr(paths, "root", lambda: tmp_path)
    monkeypatch.setattr(paths, "tool_dir", lambda: tmp_path / "tools" / "measure")
    (tmp_path / "tools" / "measure").mkdir(parents=True)
    model = tmp_path / "models" / "fake-det"
    model.mkdir(parents=True)
    (model / "adapter.py").write_text("SIZES = ['s']\n")
    (model / "pyproject.toml").write_text("")
    (model / "uv.lock").write_text("lock 1")
    (tmp_path / "timing.jpg").write_bytes(b"jpg")
    return tmp_path


def passed_check(model: Path, size: str = "s", **over) -> None:
    check = {"ok": True, "size": size, "protocol": protocol.PROTOCOL_VERSION, **record.fingerprint(model),
             "load_s": 2.0, "approx_ms": 10.0, "weight_files": [], **over}
    path = record.check_path(model, size)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(check))


def result(capsys) -> dict:
    return errors.parse_result(capsys.readouterr().out)


def test_add_needs_a_passed_check(repo, capsys):
    model = repo / "models" / "fake-det"
    args = ["queue", "add", str(model), "--size", "s", "--input", str(repo / "timing.jpg")]
    assert cli.main(args) == 14
    assert result(capsys)["code"] == "NOT_CHECKED"

    passed_check(model)
    assert cli.main(args) == 0
    r = result(capsys)
    assert r["outputs"]["queued"] == 1
    item = json.loads((repo / "tools" / "measure" / "queue.json").read_text())[0]
    assert item["model_dir"] == "models/fake-det" and item["inputs"] == ["timing.jpg"]
    # 3 runs of (worker start + load + 110 calls of 10 ms + baselines)
    assert item["estimate_s"] == round(protocol.RUNS * (runqueue.WORKER_START_S + 2.0 + 1.1 + runqueue.BASELINE_S))

    assert cli.main(args) == 0  # the same run twice stays one item
    assert result(capsys)["outputs"]["queued"] == 1


def test_editing_the_adapter_or_lock_voids_the_check(repo, capsys):
    model = repo / "models" / "fake-det"
    passed_check(model)
    (model / "uv.lock").write_text("lock 2")
    with pytest.raises(errors.MeasureError, match="changed"):
        runqueue.read_check(model, "s")
    passed_check(model, protocol=1)
    with pytest.raises(errors.MeasureError, match="protocol"):
        runqueue.read_check(model, "s")
    passed_check(model, ok=False)
    with pytest.raises(errors.MeasureError, match="failed"):
        runqueue.read_check(model, "s")


def test_list_and_drop(repo, capsys):
    model = repo / "models" / "fake-det"
    passed_check(model)
    passed_check(model, "m")
    for size in ("s", "m"):
        assert cli.main(["queue", "add", str(model), "--size", size, "--input", str(repo / "timing.jpg")]) == 0
    capsys.readouterr()
    assert cli.main(["queue", "list"]) == 0
    assert len(result(capsys)["outputs"]["items"]) == 2
    assert cli.main(["queue", "drop", "1"]) == 0
    assert result(capsys)["outputs"]["queued"] == 1
    assert cli.main(["queue", "drop", "9"]) == 31
    assert cli.main(["queue", "drop", "all"]) == 0
    capsys.readouterr()
    assert cli.main(["queue", "run"]) == 0
    assert "empty" in result(capsys)["message"]


def test_run_keeps_failures_with_their_reason(repo, capsys, monkeypatch):
    model = repo / "models" / "fake-det"
    for size in ("s", "m"):
        passed_check(model, size)
        assert cli.main(["queue", "add", str(model), "--size", size, "--input", str(repo / "timing.jpg")]) == 0
    capsys.readouterr()

    class NoPause:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(runqueue.desktop, "paused", lambda log: NoPause())
    monkeypatch.setattr(runqueue.desktop, "extend_fallback", lambda: None)
    outcomes = {"s": {"status": "ok", "code": "OK", "message": "fine", "outputs": {}},
                "m": {"status": "fail", "code": "VARIANCE", "message": "spread 9%", "outputs": {}}}
    monkeypatch.setattr(runqueue, "run_one", lambda item: outcomes[item["size"]])
    assert cli.main(["queue", "run"]) == 31
    r = result(capsys)
    assert [x["code"] for x in r["outputs"]["results"]] == ["OK", "VARIANCE"]
    left = runqueue.load()
    assert [i["size"] for i in left] == ["m"] and left[0]["last_error"]["code"] == "VARIANCE"


def test_the_queue_can_leave_the_input_to_the_timing_image(repo, capsys):
    model = repo / "models" / "fake-det"
    passed_check(model)
    assert cli.main(["queue", "add", str(model), "--size", "s"]) == 0
    assert runqueue.load()[0]["inputs"] == []


def test_new_copies_the_template(repo, capsys):
    tpl = repo / "models" / "_template"
    tpl.mkdir()
    (tpl / "adapter.py").write_text("")
    (tpl / "pyproject.toml").write_text('name = "mtr-model-template"\n')
    (tpl / "meta.toml").write_text('id = "model-id"\n')
    (tpl / ".python-version").write_text("3.12\n")
    assert cli.main(["new", "rf-detr"]) == 0
    made = repo / "models" / "rf-detr"
    assert (made / ".python-version").is_file()
    assert 'mtr-model-rf-detr' in (made / "pyproject.toml").read_text()
    assert 'id = "rf-detr"' in (made / "meta.toml").read_text()
    assert cli.main(["new", "rf-detr"]) == 2
    assert cli.main(["new", "Bad_Name"]) == 2


def test_clean_lists_before_it_deletes(repo, capsys, monkeypatch, tmp_path):
    model = repo / "models" / "fake-det"
    blob = tmp_path / "cache" / "blob"
    blob.parent.mkdir()
    blob.write_bytes(b"x" * 2048)
    link = tmp_path / "cache" / "model.safetensors"
    link.symlink_to(blob)
    (model / ".venv").mkdir()
    (model / ".venv" / "lib.so").write_bytes(b"y" * 1024)
    passed_check(model, weight_files=[str(blob), str(link)])
    assert cli.main(["clean", str(model)]) == 0
    assert result(capsys)["outputs"]["bytes"] >= 3072
    assert blob.exists() and (model / ".venv").exists()

    monkeypatch.setattr(manage.subprocess, "run", lambda *a, **k: type("P", (), {"stderr": "", "stdout": ""})())
    assert cli.main(["clean", str(model), "--yes"]) == 0
    assert not blob.exists() and not link.is_symlink() and not (model / ".venv").exists()
    assert (model / "adapter.py").exists()
