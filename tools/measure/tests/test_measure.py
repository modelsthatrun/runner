"""Tests for the parts of the tool that don't need a GPU. Run: uv run --frozen --project tools/measure pytest tools/measure/tests -q"""

import hashlib
import json
import types
from pathlib import Path

import pytest

from mtr_measure import adapter as ad
from mtr_measure import checks, errors, protocol, record
from mtr_measure.gpu import MIB, GpuProcess, Samples


def test_percentile_matches_numpy_linear():
    values = [float(v) for v in range(1, 101)]
    assert record.percentile(values, 50) == pytest.approx(50.5)
    assert record.percentile(values, 90) == pytest.approx(90.1)
    assert record.percentile([3.0, 1.0, 2.0], 90) == pytest.approx(2.8)
    assert record.percentile([7.0], 90) == 7.0


def test_precision_rule():
    assert ad.choose_precision(["fp32", "fp16"]) == "fp16"
    assert ad.choose_precision(["fp32"]) == "fp32"
    assert ad.choose_precision(["bf16", "fp32"]) == "fp32"
    with pytest.raises(ad.AdapterError):
        ad.choose_precision(["bf16"])


def fake_adapter(**over):
    a = types.SimpleNamespace(
        SIZES=["s"], PRECISIONS=["fp16", "fp32"], PACKAGE="pytest", INPUT={"s": "640x640"},
        load=lambda size, precision: None, predict=lambda m, x: None, weight_files=lambda size: [],
    )
    for k, v in over.items():
        if v is None:
            delattr(a, k)
        else:
            setattr(a, k, v)
    return a


def test_interface_check():
    ad.check_interface(fake_adapter())
    with pytest.raises(ad.AdapterError, match="weight_files"):
        ad.check_interface(fake_adapter(weight_files=None))
    with pytest.raises(ad.AdapterError, match="INPUT"):
        ad.check_interface(fake_adapter(INPUT=None))
    with pytest.raises(ad.AdapterError, match="PRECISIONS"):
        ad.check_interface(fake_adapter(PRECISIONS=["int8"]))


def test_input_size_and_package():
    a = fake_adapter()
    assert ad.input_size(a, "s", None) == "640x640"
    a.input_size = lambda size, x: "800x600"
    assert ad.input_size(a, "s", None) == "800x600"
    a.input_size = lambda size, x: "800"
    with pytest.raises(ad.AdapterError):
        ad.input_size(a, "s", None)
    assert ad.package(fake_adapter()).startswith("pytest==")


def test_load_adapter_from_file(tmp_path: Path):
    (tmp_path / "adapter.py").write_text(
        "SIZES=['s']\nPRECISIONS=['fp32']\nPACKAGE='pytest'\nINPUT={'s':'1x1'}\n"
        "def load(size, precision): return 1\ndef predict(m, x): return x\ndef weight_files(size): return []\n"
    )
    a = ad.load_adapter(tmp_path / "adapter.py")
    assert a.SIZES == ["s"]
    with pytest.raises(ad.AdapterError, match="No adapter"):
        ad.load_adapter(tmp_path / "missing.py")


def worker_result(times):
    return {
        "precision": "fp16", "input": "640x640", "latency_scope": ad.DEFAULT_SCOPE, "times_ms": times,
        "torch_max_reserved_bytes": 300 * MIB,
        "env": {"cuda": "13.0", "torch": "2.14.1+cu130", "python": "3.12.9", "package": "rfdetr==1.11.2"},
    }


def run(times, baseline_mib=600, peak_mib=1800, reserved_mib=300):
    w = worker_result(times)
    w["torch_max_reserved_bytes"] = reserved_mib * MIB
    return {"worker": w, "baseline_bytes": baseline_mib * MIB, "peak_bytes": peak_mib * MIB}


def test_record_passes_the_cms_schema(tmp_path: Path):
    w = tmp_path / "model.pth"
    w.write_bytes(b"weights")
    times = [float(t) for t in range(1, 101)]
    rec = record.build(
        model="rf-detr", size="nano", weights=record.weight_entries([w]),
        runs=[run(times), run([t + 1 for t in times], peak_mib=1900, reserved_mib=320), run([t + 2 for t in times], baseline_mib=610)],
        gpu_name="NVIDIA GeForce RTX 4090", driver="580.173.02", image_digest=None, warmup=10, measured="2026-10-22T10:00:00+09:00",
    )
    assert record.validate(rec) == []
    assert rec["kind"] == "image"
    m = rec["measurement"]
    assert m["protocol"] == protocol.PROTOCOL_VERSION
    assert [r["median"] for r in m["runs"]] == [50.5, 51.5, 52.5]
    # The medians of the runs' medians and p90s, and the highest peak over each run's own baseline.
    assert m["latency_ms"] == {"median": 51.5, "p90": 91.1}
    assert m["vram_peak_mib"] == 1300
    assert m["torch_max_reserved_mib"] == 320
    assert m["env"]["baseline_mib"] == 600
    assert m["iterations"] == {"warmup": 10, "timed": 100}
    assert rec["weights"] == [{"file": "model.pth", "bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}]

    broken = json.loads(json.dumps(rec))
    broken["measurement"]["batch"] = 2
    broken["measurement"]["extra"] = 1
    problems = record.validate(broken)
    assert any("batch" in p for p in problems) and any("extra" in p for p in problems)

    # A version-1 record, with one run and no kind, protocol or runs, still passes.
    old = json.loads(json.dumps(rec))
    for k in ("protocol", "runs"):
        del old["measurement"][k]
    del old["kind"]
    assert record.validate(old) == []
    llm = {**rec, "kind": "llm"}
    assert record.validate(llm)


def test_spread_of_run_medians():
    assert record.spread([{"median": 10.0}, {"median": 10.4}, {"median": 10.2}]) == pytest.approx(0.4 / 10.2)
    assert record.spread([{"median": 5.0}]) == 0


def test_weight_entries_keep_order_and_need_files(tmp_path: Path):
    a, b = tmp_path / "main.pth", tmp_path / "aux.bin"
    a.write_bytes(b"a")
    b.write_bytes(b"bb")
    entries = record.weight_entries([a, b])
    assert [e["file"] for e in entries] == ["main.pth", "aux.bin"]
    assert entries[1]["bytes"] == 2
    with pytest.raises(FileNotFoundError):
        record.weight_entries([tmp_path / "nope.pth"])
    with pytest.raises(ValueError):
        record.weight_entries([])


def samples(mib, util=0, gap=5.0):
    used = [int(m * MIB) for m in mib]
    return Samples(used, [gap] * (len(used) - 1), [util] * len(used))


def test_checks_before_run(monkeypatch):
    monkeypatch.setattr(checks, "running_blockers", lambda: {})
    quiet = samples([600] * 200)
    assert checks.before_run([GpuProcess(1, "G", 300, "Xorg")], quiet).ok()

    monkeypatch.setattr(checks, "running_blockers", lambda: {"Sunshine streaming host": ["sunshine (pid 9)"]})
    assert any("Sunshine" in e for e in checks.before_run([], quiet).errors)

    monkeypatch.setattr(checks, "running_blockers", lambda: {})
    game = GpuProcess(42, "C+G", 4000, "SomeGame.exe")
    assert any("SomeGame" in e for e in checks.before_run([game], quiet).errors)
    assert checks.before_run([], samples([600] * 200, util=30)).errors
    assert checks.before_run([], samples([600, 800] * 100)).errors
    assert checks.before_run([], samples([2000] * 200)).warnings


def test_checks_after_run():
    before = samples([600] * 200)
    during = samples([600, 1800, 1700] * 100)
    assert checks.after_run(before, samples([610] * 200), during).ok()
    assert not checks.after_run(before, samples([900] * 200), during).ok()
    assert not checks.after_run(before, samples([600] * 200), samples([600] * 100, gap=20.0)).ok()


def test_check_errors_carry_result_codes(monkeypatch):
    monkeypatch.setattr(checks, "running_blockers", lambda: {"Steam": ["steam (pid 3)"]})
    r = checks.before_run([GpuProcess(42, "C", 4000, "SomeGame.exe")], samples([600] * 200))
    assert r.codes == ["DESKTOP", "GPU_BUSY"]
    post = checks.after_run(samples([600] * 200), samples([900] * 200), samples([600] * 100))
    assert post.codes == ["DRIFT"]


def test_result_line_round_trip():
    lines: list[str] = []
    rc = errors.emit("VARIANCE", "spread 7%", {"spread": 0.07}, log=lines.append)
    assert rc == 24
    parsed = errors.parse_result("noise\n" + lines[-1])
    assert parsed["status"] == "fail" and parsed["code"] == "VARIANCE" and "MAX_SPREAD" in parsed["hint"]
    assert errors.parse_result("no result here") is None
    with pytest.raises(ValueError):
        errors.MeasureError("NOPE", "x")
    assert ad.AdapterError("x").code == "ADAPTER_INTERFACE"
    assert ad.AdapterError("x", "USAGE").code == "USAGE"


def test_fingerprint_follows_adapter_and_lock(tmp_path: Path):
    model = tmp_path / "m"
    model.mkdir()
    (model / "adapter.py").write_text("A = 1\n")
    (model / "pyproject.toml").write_text("")
    first = record.fingerprint(model)
    assert first["lock_sha256"] is None
    (model / "uv.lock").write_text("lock")
    second = record.fingerprint(model)
    assert second["lock_sha256"] and second["adapter_sha256"] == first["adapter_sha256"]
    (model / "adapter.py").write_text("A = 2\n")
    assert record.fingerprint(model)["adapter_sha256"] != second["adapter_sha256"]
    # An example adapter uses its parent's environment and lock.
    shared = tmp_path / "examples" / "resnet"
    shared.mkdir(parents=True)
    (shared / "adapter.py").write_text("")
    (shared.parent / "pyproject.toml").write_text("")
    assert record.project_dir(shared) == shared.parent


def test_usage_errors_end_with_a_result_line(capsys):
    from mtr_measure import cli

    assert cli.main(["check", "models/does-not-exist", "--size", "s"]) == 2
    out = capsys.readouterr().out.strip().splitlines()
    assert errors.parse_result(out[-1])["code"] == "USAGE"
    assert cli.main(["run", "--size"]) == 2


def test_cpu_checks(monkeypatch):
    monkeypatch.setattr(checks, "running_blockers", lambda: {})
    quiet = samples([600] * 200)
    assert checks.before_run([], quiet, 0.03).ok()
    r = checks.before_run([], quiet, 0.40)
    assert r.codes == ["CPU_BUSY"]
    post = checks.after_run(quiet, quiet, samples([600, 900] * 100), 0.25)
    assert post.codes == ["CPU_BUSY"]
    assert checks.after_run(quiet, quiet, samples([600, 900] * 100), 0.05).ok()


def test_cpu_window_leaves_out_its_own_time(monkeypatch):
    from mtr_measure import cpu

    stats = iter([(1000, 100), (2600, 900)])  # 1600 ticks pass; 800 of them busy
    own = iter([50.0, 450.0])  # 400 of the busy ticks are this process and its worker
    monkeypatch.setattr(cpu, "machine", lambda: next(stats))
    monkeypatch.setattr(cpu, "own_ticks", lambda: next(own))
    assert cpu.Window().start().others() == pytest.approx(400 / 1600)
