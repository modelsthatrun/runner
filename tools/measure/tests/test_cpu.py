"""Tests for CPU rows (cpurun.py, cpullm.py, ort.py, match.py): file names, the match with the GPU's output, the
checks around a run, the records and the queue. No GPU, no model and no ONNX Runtime needed."""

import ast
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from mtr_measure import checks, cli, cpurun, errors, main, match, paths, predictions, protocol, record, runqueue, runtimes, submit

ROOT = Path(__file__).resolve().parents[3]
FACTS = {"cpu": "Intel(R) Core(TM) i7-10700K CPU @ 3.80GHz", "cpu_cores": 8, "cpu_threads": 16, "ram_total_mib": 96441,
         "os": "Linux 6.2.6 glibc 2.35", "cpu_power_limit_w": 125.0}
WEIGHTS = [{"file": "w.pth", "bytes": 10, "sha256": "a" * 64}]
NOW = datetime.now().astimezone().isoformat(timespec="seconds")
MATCH = {"type": "boxes", "reference": "pytorch-cuda", "metrics": {"recall": 1.0}, "tolerance": {"recall": 0.9}}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "root", lambda: tmp_path)
    monkeypatch.setattr(paths, "tool_dir", lambda: tmp_path / "tools" / "measure")
    (tmp_path / "tools" / "measure").mkdir(parents=True)
    model = tmp_path / "models" / "fake-det"
    model.mkdir(parents=True)
    (model / "adapter.py").write_text("SIZES = ['s', '1.6b']\n")
    (model / "pyproject.toml").write_text("")
    (model / "uv.lock").write_text("lock 1")
    (tmp_path / "timing.jpg").write_bytes(b"jpg")
    return tmp_path


def write_check(model: Path, size: str = "s", runtime: str = runtimes.DEFAULT, **over) -> None:
    check = {"ok": True, "size": size, "protocol": protocol.PROTOCOL_VERSION, **record.fingerprint(model), "load_s": 2.0,
             "approx_ms": 100.0, "weight_files": [], "kind": "image", **({"runtime": runtime} if runtimes.is_cpu(runtime) else {}), **over}
    path = record.check_path(model, size, runtime)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(check))


def result(capsys) -> dict:
    return errors.parse_result(capsys.readouterr().out)


# --- names ---------------------------------------------------------------------------------------


def test_sizes_with_dots_and_runtimes_part_cleanly():
    assert runtimes.split_name("1.6b.pytorch-cpu.times.json") == ("1.6b", "pytorch-cpu", "times")
    assert runtimes.split_name("0.6b-q8_0.llamacpp-cpu.json") == ("0.6b-q8_0", "llamacpp-cpu", "")
    assert runtimes.split_name("1.0.json") == ("1.0", "pytorch-cuda", "")
    assert runtimes.split_name("s.tensorrt.times.json") == ("s", "tensorrt", "times")
    files = ["1.6b.json", "1.6b.times.json", "1.6b.onnxruntime-cpu.json", "s.json", "s.pytorch-cpu.times.json"]
    assert runtimes.record_files(files) == [("1.6b.json", "1.6b", "pytorch-cuda"), ("1.6b.onnxruntime-cpu.json", "1.6b", "onnxruntime-cpu"),
                                            ("s.json", "s", "pytorch-cuda")]
    assert record.check_path(Path("m"), "1.6b", "pytorch-cpu").name == "1.6b.pytorch-cpu.json"
    assert record.prediction_path(Path("m"), "s", "onnxruntime-cpu").name == "s.onnxruntime-cpu.prediction.npz"


def test_each_runtime_takes_its_kinds():
    runtimes.check("image", "pytorch-cpu")
    runtimes.check("generation", "pytorch-cpu")
    runtimes.check("llm", "llamacpp-cpu")
    for kind, runtime in (("llm", "pytorch-cpu"), ("image", "llamacpp-cpu"), ("generation", "onnxruntime-cpu"), ("llm", "tensorrt")):
        with pytest.raises(errors.MeasureError) as e:
            runtimes.check(kind, runtime)
        assert e.value.code == "USAGE"


def test_local_and_submittable_records_are_the_default_runtimes_only(repo):
    runs = repo / "models" / "fake-det" / "runs"
    runs.mkdir()
    for f in ("s.json", "s.times.json", "1.6b.json", "s.pytorch-cpu.json", "s.tensorrt.json", "1.6b.llamacpp-cpu.json"):
        (runs / f).write_text("{}")
    assert main.local_runs(repo / "models" / "fake-det") == ["1.6b", "s"]
    assert [s for s, _ in submit.records(repo / "models" / "fake-det")] == ["1.6b", "s"]


# --- the match with the GPU ----------------------------------------------------------------------


def boxes(shift: float = 0.0, scores=(0.9, 0.8, 0.6), labels=("dog", "person", "car")) -> predictions.Boxes:
    xyxy = np.array([[10, 10, 110, 110], [200, 50, 260, 250], [300, 300, 400, 350]], dtype=np.float32) + shift
    return predictions.Boxes(xyxy, np.array(scores), list(labels))


def compare(tmp_path, ref, got, task="object-detection", runtime="pytorch-cpu") -> dict:
    match.save(ref, task, tmp_path / "ref.npz")
    match.save(got, task, tmp_path / "got.npz")
    return match.compare(tmp_path / "ref.npz", tmp_path / "got.npz", runtime)


def test_boxes_match_when_the_cpu_finds_the_same_objects(tmp_path):
    r = compare(tmp_path, boxes(), boxes(0.5, scores=(0.89, 0.81, 0.6)))
    assert r["passed"] and r["metrics"]["recall"] == 1.0 and r["metrics"]["mean_iou"] > 0.97
    # An object just under the display threshold on one side still finds its match.
    assert compare(tmp_path, boxes(scores=(0.9, 0.8, 0.51)), boxes(scores=(0.9, 0.8, 0.49)))["passed"]
    moved = compare(tmp_path, boxes(), boxes(40.0))
    assert not moved["passed"] and moved["metrics"]["recall"] < 0.9
    relabelled = compare(tmp_path, boxes(), boxes(labels=("cat", "person", "car")))
    assert not relabelled["passed"] and relabelled["metrics"]["recall"] < 0.9
    # ONNX Runtime's looser tolerance takes a smaller shift.
    assert compare(tmp_path, boxes(), boxes(4.0), runtime="onnxruntime-cpu")["passed"]
    assert not compare(tmp_path, boxes(), boxes(4.0))["passed"]


def test_maps_and_masks_match(tmp_path):
    a = np.zeros((40, 60), np.float32)
    a[10:30, 10:50] = 1
    assert compare(tmp_path, predictions.Alpha(a), predictions.Alpha(a * 0.995), "image-matting")["passed"]
    assert not compare(tmp_path, predictions.Alpha(a), predictions.Alpha(np.roll(a, 8, 1)), "image-matting")["passed"]
    d = np.linspace(1, 5, 40 * 60, dtype=np.float32).reshape(40, 60)
    assert compare(tmp_path, predictions.Depth(d), predictions.Depth(d * 1.001), "depth-estimation")["passed"]
    assert not compare(tmp_path, predictions.Depth(d), predictions.Depth(d[::-1].copy()), "depth-estimation")["passed"]
    m = np.zeros((1, 40, 60), bool)
    m[0, 5:30, 5:30] = True
    ok = compare(tmp_path, predictions.Masks(m, ["dog"], np.array([0.9])), predictions.Masks(m, ["dog"], np.array([0.88])), "instance-segmentation")
    assert ok["passed"] and ok["metrics"]["mean_iou"] == 1.0


def test_keypoints_match(tmp_path):
    pts = np.array([[[10, 10], [50, 10], [30, 80]]], np.float32)
    kp = lambda p, s=0.9: predictions.Keypoints(p, np.array([s]), ["a", "b", "c"])  # noqa: E731
    assert compare(tmp_path, kp(pts), kp(pts + 0.5), "pose-estimation")["passed"]
    assert not compare(tmp_path, kp(pts), kp(pts + np.array([[[0, 0], [0, 0], [25, 0]]], np.float32)), "pose-estimation")["passed"]


# --- checks around a run -------------------------------------------------------------------------


def test_a_cpu_run_needs_a_quiet_cool_steady_machine():
    assert checks.before_cpu_run(0.02, 50_000, 1000).ok()
    assert checks.before_cpu_run(0.065, 50_000, 1000).ok()  # a quiet desktop on the reference machine
    assert checks.before_cpu_run(0.12, 50_000, 1000).codes == ["CPU_BUSY"]  # ... and one more busy logical CPU
    assert checks.before_cpu_run(0.02, 1500, 1000).codes == ["RAM_LOW"]
    assert checks.after_cpu_run(0.02, 3, 3, {"VmSwap": 0}, 0.01).ok()
    assert checks.after_cpu_run(0.02, 3, 4, {"VmSwap": 0}, 0.01).codes == ["THROTTLED"]
    assert checks.after_cpu_run(0.02, None, None, {"VmSwap": 2048}, None).codes == ["RAM_LOW"]
    assert checks.after_cpu_run(0.02, 0, 0, {}, 0.08).codes == ["UNSTEADY"]
    assert cpurun.steadiness([10.0] * 5) is None
    assert cpurun.steadiness([10.0] * 15 + [11.0] * 15) == pytest.approx(0.1, abs=0.02)


def test_calls_and_estimates_follow_the_check():
    assert cpurun.calls_for("image", {"approx_ms": 100}) == (protocol.CPU_WARMUP, protocol.CPU_TIMED)
    assert cpurun.calls_for("image", {"approx_ms": 20_000}) == (protocol.CPU_SLOW_WARMUP, protocol.CPU_SLOW_TIMED)
    assert cpurun.calls_for("generation", {"approx_ms": 1000}) == (protocol.CPU_GEN_WARMUP, protocol.CPU_GEN_TIMED)
    fast = cpurun.estimate_s({"kind": "image", "runtime": "pytorch-cpu", "approx_ms": 100, "load_s": 2})
    assert fast > protocol.RUNS * protocol.CPU_WARMUP_MIN_S
    assert runqueue.estimate_s({"kind": "image", "runtime": "pytorch-cpu", "approx_ms": 100, "load_s": 2}) == fast


# --- records -------------------------------------------------------------------------------------


def cpu_worker(times, ram_kib=900_000) -> dict:
    return {"precision": "fp32", "input": "640x640", "latency_scope": "predict()", "times_ms": times, "memory": {"VmHWM": ram_kib},
            "env": {"python": "3.12.9", "torch": "2.14.1+cu130", "package": "repo@abc"}}


def test_cpu_records_pass_the_cpu_schema_and_never_the_gpu_one():
    runtime = {"name": "pytorch-cpu", "version": "2.14.1", "status": "runs-as-is", "threads": 8, "match": MATCH}
    runs = [{"worker": cpu_worker([100.0, 101.0, 99.0] * 10)} for _ in range(3)]
    rec = record.build_cpu(model="fake-det", size="s", weights=WEIGHTS, runs=runs, runtime=runtime, warmup=5, measured=NOW, machine=FACTS)
    assert record.validate(rec, "cpu-run-record") == []
    assert record.validate(rec, "run-record")  # not an RTX 4090 record
    assert rec["measurement"]["ram_peak_mib"] == round(900_000 / 1024)
    nt = record.build_not_timed(model="fake-det", size="s", weights=WEIGHTS, kind="generation", runtime="pytorch-cpu", version="2.14.1",
                                reason="over-limit", detail="one image ran 301 s", measured=NOW, machine=FACTS, env={"python": "3.12.9"},
                                input="1024x1024", limit_s=protocol.CPU_CALL_LIMIT_S, call_s=301.2)
    assert record.validate(nt, "cpu-run-record") == []
    llm_worker = {"gpu_layers": "0/29", "python": "3.12.9", "memory": {"VmHWM": 1_700_000},
                  "requests": [{"ttft_ms": 1500.0, "e2e_ms": 9000.0, "prefill_tps": 330.0, "decode_tps": 30.0, "prompt_n": 512, "predicted_n": 256}] * 5}
    engine = {"name": "llama.cpp", "release": "v0.7.0", "build": "b11669", "cuda": "13.0", "binary_sha256": "c" * 64}
    lrec = record.build_cpu_llm(model="qwen3", size="0.6b-q8_0", weights=WEIGHTS, runs=[{"worker": llm_worker}] * 3,
                                runtime={"name": "llamacpp-cpu", "version": "b11669", "status": "runs-as-is", "threads": 8,
                                         "match": {"type": "tokens", "reference": "pytorch-cuda", "metrics": {"top1_agree": 1.0}, "tolerance": {"top1_agree": 0.875}}},
                                quant="Q8_0", engine=engine, server_args=protocol.LLM_CPU_SERVER_ARGS + ["-t", "8", "-tb", "8"], warmup=1,
                                measured=NOW, machine=FACTS)
    assert record.validate(lrec, "cpu-run-record") == []
    assert record.validate(lrec, "llm-run-record")


def test_the_cpu_server_args_keep_every_layer_and_batch_off_the_gpu():
    args = protocol.LLM_CPU_SERVER_ARGS
    assert args[args.index("-ngl") + 1] == "0" and args[args.index("-dev") + 1] == "none"
    assert args[args.index("-c") + 1] == str(protocol.LLM_CONTEXT)


# --- the queue -----------------------------------------------------------------------------------


def test_the_queue_takes_a_cpu_row_once_it_matches_the_gpu(repo, capsys):
    model = repo / "models" / "fake-det"
    args = ["queue", "add", str(model), "--size", "s", "--runtime", "pytorch-cpu", "--input", str(repo / "timing.jpg")]
    assert cli.main(args) == errors.exit_code("NOT_CHECKED")
    write_check(model, runtime="pytorch-cpu", match=None)
    assert cli.main(args) == errors.exit_code("NOT_MATCHED")
    write_check(model, runtime="pytorch-cpu", match=MATCH)
    assert cli.main(args) == 0
    item = runqueue.load()[0]
    assert item["runtime"] == "pytorch-cpu" and item["estimate_s"] > protocol.RUNS * protocol.CPU_WARMUP_MIN_S
    capsys.readouterr()


def test_the_queue_takes_a_row_whose_check_found_why_it_has_no_number(repo, capsys):
    model = repo / "models" / "fake-det"
    write_check(model, runtime="pytorch-cpu", ok=False, not_timed={"code": "CPU_UNSUPPORTED", "reason": "unsupported", "detail": "Triton"})
    assert cli.main(["queue", "add", str(model), "--size", "s", "--runtime", "pytorch-cpu"]) == 0
    # A plain failed check is still refused.
    write_check(model, "1.6b", runtime="pytorch-cpu", ok=False)
    assert cli.main(["queue", "add", str(model), "--size", "1.6b", "--runtime", "pytorch-cpu"]) == errors.exit_code("NOT_CHECKED")


def test_queue_run_measures_cpu_rows_after_the_pause(repo, capsys, monkeypatch):
    model = repo / "models" / "fake-det"
    write_check(model)
    write_check(model, runtime="pytorch-cpu", match=MATCH)
    assert cli.main(["queue", "add", str(model), "--size", "s", "--runtime", "pytorch-cpu"]) == 0
    assert cli.main(["queue", "add", str(model), "--size", "s"]) == 0
    capsys.readouterr()
    events: list[str] = []

    class Pause:
        def __enter__(self):
            events.append("pause")

        def __exit__(self, *a):
            events.append("resume")
            return False

    monkeypatch.setattr(runqueue.desktop, "paused", lambda log: Pause())
    monkeypatch.setattr(runqueue.desktop, "extend_fallback", lambda s: None)
    monkeypatch.setattr(runqueue, "run_one", lambda item: events.append(item.get("runtime", "gpu")) or {"status": "ok", "code": "OK", "message": "", "outputs": {}})
    assert cli.main(["queue", "run"]) == 0
    assert events == ["pause", "gpu", "resume", "pytorch-cpu"]


# --- adapters ------------------------------------------------------------------------------------


# Where an adapter may still name the GPU: its TensorRT hooks, which only ever run there, and a GENERATION entry's
# generator device, which the tool moves to the CPU for a CPU row.
GPU_ONLY = {"load_engine", "trt_call", "export_engine"}


def test_adapters_put_the_model_on_the_tools_device():
    found = []
    for f in sorted((ROOT / "models").glob("*/adapter.py")):
        if f.parent.name.startswith("_"):
            continue
        tree = ast.parse(f.read_text())
        for fn in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name not in GPU_ONLY]:
            for node in ast.walk(fn):
                if isinstance(node, ast.Constant) and node.value in ("cuda", "cuda:0"):
                    found.append(f"{f.parent.name}:{fn.name}:{node.lineno}")
    assert found == []


# --- rows without a number -----------------------------------------------------------------------


def test_the_parent_stops_a_call_past_the_limit(tmp_path, monkeypatch):
    (tmp_path / "fake_slow.py").write_text(
        "import json, sys, time\nfrom pathlib import Path\n"
        "spec = json.loads(Path(sys.argv[1]).read_text())\n"
        "Path(spec['heartbeat']).write_text(json.dumps({'weights': ['w.pth'], 'phase': 'timed', 'call': 0, 'started': time.time()}))\n"
        "time.sleep(60)\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    started = datetime.now()
    with pytest.raises(cpurun.OverLimit) as e:
        cpurun.spawn({}, "fake_slow", None, print, limit_s=0.5)
    assert e.value.code == "OVER_LIMIT" and e.value.call_s >= 0.5 and e.value.beat["weights"] == ["w.pth"]
    assert (datetime.now() - started).total_seconds() < 20


@pytest.mark.parametrize("failure", ["over-limit", "unsupported", "precision"])
def test_a_row_without_a_number_goes_from_check_to_queue_to_record(repo, capsys, monkeypatch, failure):
    model = repo / "models" / "fake-det"
    weights = repo / "w.pth"
    weights.write_bytes(b"weights")
    jpg = str(repo / "timing.jpg")

    def spawn(spec, module, gpu, log, limit_s=None):
        if failure == "over-limit":
            raise cpurun.OverLimit(312.0, {"weights": [str(weights)], "phase": "timed"})
        code = {"unsupported": "CPU_UNSUPPORTED", "precision": "PRECISION"}[failure]
        raise cpurun.WorkerFailed({"ok": False, "code": code, "error": f"{failure}: what the worker saw", "weights": [str(weights)]})

    monkeypatch.setattr(cpurun, "spawn", spawn)
    monkeypatch.setattr(cpurun, "BASELINE_S", 0)
    monkeypatch.setattr(cpurun.cpu.Window, "others", lambda self: 0.0)
    monkeypatch.setattr(cpurun, "open_gpu", lambda index: None)
    cli.main(["check", str(model), "--size", "s", "--runtime", "pytorch-cpu", "--input", jpg])
    check = json.loads(record.check_path(model, "s", "pytorch-cpu").read_text())
    assert check.get("over_limit") if failure == "over-limit" else check["not_timed"]["reason"] == failure
    assert cli.main(["queue", "add", str(model), "--size", "s", "--runtime", "pytorch-cpu", "--input", jpg]) == 0
    capsys.readouterr()

    assert cli.main(["run", str(model), "--size", "s", "--runtime", "pytorch-cpu", "--input", jpg]) == 0
    rec = json.loads((model / "runs" / "s.pytorch-cpu.json").read_text())
    assert record.validate(rec, "cpu-run-record") == []
    assert rec["measurement"]["not_timed"]["reason"] == failure
    assert rec["weights"][0]["sha256"] == record.sha256_file(weights)
    assert not (model / "runs" / "s.pytorch-cpu.times.json").exists()
    if failure == "over-limit":
        assert rec["measurement"]["not_timed"]["limit_s"] == protocol.CPU_CALL_LIMIT_S and rec["measurement"]["not_timed"]["call_s"] == 312.0


def test_a_busy_machine_never_records_a_row_as_over_the_limit(repo, capsys, monkeypatch):
    model = repo / "models" / "fake-det"
    write_check(model, runtime="pytorch-cpu", over_limit={"call_s": 310.0}, weight_files=[])
    monkeypatch.setattr(cpurun, "spawn", lambda *a, **k: (_ for _ in ()).throw(cpurun.OverLimit(320.0, {"weights": []})))
    monkeypatch.setattr(cpurun, "BASELINE_S", 0)
    monkeypatch.setattr(cpurun, "open_gpu", lambda index: None)
    others = iter([0.01, 0.30])  # quiet before the run, busy during it
    monkeypatch.setattr(cpurun.cpu.Window, "others", lambda self: next(others))
    assert cli.main(["run", str(model), "--size", "s", "--runtime", "pytorch-cpu", "--input", str(repo / "timing.jpg")]) == errors.exit_code("CPU_BUSY")
    assert not (model / "runs" / "s.pytorch-cpu.json").exists()
