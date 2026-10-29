"""Tests for what makes the tool work on anyone's machine: pause hooks, the engine's architecture, provenance,
the timing images, model ids, the timings file and submitting. No GPU or network needed."""

import getpass
import json
import socket
from pathlib import Path

import pytest

from mtr_measure import adapter as ad
from mtr_measure import checks, desktop, engine, errors, machine, main, paths, protocol, record, submit, timing

ROOT = Path(__file__).resolve().parents[3]
TIMING_SHA = "787c3de9bbd90c014a272ea35650164a5cd62f1a98a38904657f131236ebcee7"


@pytest.fixture
def config(tmp_path, monkeypatch):
    """An empty config directory for machine.toml and credentials.json."""
    monkeypatch.setenv("MTR_CONFIG_DIR", str(tmp_path / "config"))
    machine.config.cache_clear()
    yield tmp_path / "config"
    machine.config.cache_clear()


def write_config(d: Path, text: str) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "machine.toml").write_text(text)
    machine.config.cache_clear()


def test_nothing_pauses_without_machine_toml(config):
    assert machine.pause_hooks() == []
    assert desktop.programs() == []
    assert checks.blocking_programs() == {}
    assert checks._blocking("steam") is None  # an ordinary program elsewhere


def test_pause_hooks_come_from_machine_toml(config):
    write_config(config, 'pause = ["sunshine", "steam"]\n')
    assert [p.label for p in desktop.programs()] == ["Sunshine streaming host", "Steam"]
    assert checks._blocking("sunshine") == "Sunshine streaming host"
    write_config(config, 'pause = ["obs"]\n')
    with pytest.raises(errors.MeasureError, match="unknown pause hooks"):
        desktop.programs()


def test_the_engine_builds_for_the_local_architecture(monkeypatch, tmp_path):
    monkeypatch.setenv("MTR_ENGINES", str(tmp_path))
    monkeypatch.setattr(engine, "arch", lambda gpu=0: "86")
    assert engine.server_path().parts[-4] == f"{engine.LLAMA_CPP['build']}-sm86"
    assert not any("ARCHITECTURES" in f for f in engine.CMAKE_FLAGS)


class FakeGpu:
    def total_bytes(self):
        return 24 * 2**30

    def compute_capability(self):
        return "8.6"

    def power_limit_w(self):
        return 300.0

    def power_default_w(self):
        return 350.0


def test_records_name_the_machine_but_never_the_host_user_or_paths(tmp_path):
    model = ROOT / "models" / "rf-detr"
    facts = machine.facts(FakeGpu())
    prov = machine.provenance(model, [ROOT / "timing" / "dog-walk-square.jpg"])
    text = json.dumps({"facts": facts, "prov": prov})
    assert facts["gpu_memory_mib"] == 24576 and facts["compute_capability"] == "8.6" and facts["power_limit_w"] == 300.0
    assert prov["adapter_sha256"] == record.sha256_file(model / "adapter.py")
    assert prov["input_sha256"] == ["787c3de9bbd90c014a272ea35650164a5cd62f1a98a38904657f131236ebcee7"]
    for secret in (socket.gethostname(), getpass.getuser(), str(Path.home()), str(ROOT)):
        assert secret not in text


def test_a_bare_id_names_the_model_directory():
    assert paths.model_dir("rf-detr") == ROOT / "models" / "rf-detr"
    assert paths.model_dir("models/rf-detr") == Path("models/rf-detr")
    assert paths.model_dir("no-such-model") == Path("no-such-model")


def test_every_adapter_declares_its_task_and_readable_sizes():
    for d in main.adapters():
        f = d / "adapter.py"
        assert ad.task_of_file(f), f"{d.name} has no TASK"
        assert ad.sizes_of_file(f), f"{d.name}'s SIZES can't be read without importing it"


def test_image_tasks_have_a_matching_timing_image():
    for d in main.adapters():
        if ad.kind_of_file(d / "adapter.py") == "image":
            assert timing.image_for(ad.task_of_file(d / "adapter.py")), d.name
    assert timing.image_for("chat") is None


def worker_run(times, vram=900):
    return {"worker": {"times_ms": times, "precision": "fp16", "input": "640x640", "latency_scope": "x",
                       "torch_max_reserved_bytes": 0, "weights": [], "env": {"cuda": "13.0", "torch": "2.14.1", "python": "3.12.3", "package": "p==1"}},
            "baseline_bytes": 100 * 2**20, "peak_bytes": (100 + vram) * 2**20}


def a_record(times_by_run):
    runs = [worker_run(t) for t in times_by_run]
    rec = record.build(model="rf-detr", size="nano", weights=[{"file": "w.pth", "bytes": 1, "sha256": "a" * 64}], runs=runs,
                       gpu_name="NVIDIA GeForce RTX 3090", driver="580", image_digest=None, warmup=10, measured="2026-10-29T10:00:00+09:00",
                       machine={"cpu": "CPU", "gpu_memory_mib": 24576, "compute_capability": "8.6", "os": "Linux", "power_limit_w": 350.0, "power_default_w": 350.0},
                       provenance={"tool": {"name": "mtr", "version": machine.TOOL_VERSION, "commit": "b" * 40, "dirty": False},
                                   "adapter_sha256": "c" * 64, "lock_sha256": "d" * 64, "input_sha256": [TIMING_SHA]})
    return rec, record.times(runs, llm=False)


def test_the_timings_file_validates_and_pairs_with_the_record():
    rec, times = a_record([[10.0 + i * 0.01 for i in range(100)]] * 3)
    assert record.validate(rec) == []
    assert record.validate(times, "run-times") == []
    assert len(times["runs"]) == protocol.RUNS and len(times["runs"][0]) == 100
    body = submit.payload(rec, times, "")
    assert record.validate(body, "community-submission") == []


def test_only_records_from_an_unmodified_commit_can_be_sent(tmp_path):
    rec, times = a_record([[10.0] * 100] * 3)
    t = tmp_path / "nano.times.json"
    t.write_text(json.dumps(times))
    assert submit.problems(rec, t, "object-detection") == []
    rec["provenance"]["input_sha256"] = ["e" * 64]
    assert "another input" in submit.problems(rec, t, "object-detection")[0]
    rec["provenance"]["input_sha256"] = [TIMING_SHA]
    rec["provenance"]["tool"]["dirty"] = True
    assert "uncommitted" in submit.problems(rec, t, "object-detection")[0]
    del rec["provenance"]
    assert "no provenance" in submit.problems(rec, t)[0]
    rec2, _ = a_record([[10.0] * 100] * 3)
    assert "missing" in submit.problems(rec2, tmp_path / "gone.times.json", "object-detection")[0]


def test_the_timing_images_count_toward_a_modified_checkout(monkeypatch):
    seen = []
    monkeypatch.setattr(machine, "_git", lambda *a: seen.append(a) or ("b" * 40 + "\n" if a[0] == "rev-parse" else ""))
    machine.tool(ROOT / "models" / "rf-detr")
    status = next(a for a in seen if a[0] == "status")
    assert "timing" in status and "tools/measure" in status and "models/rf-detr" in status


def test_submit_without_a_sign_in_asks_for_one(config, capsys):
    with pytest.raises(errors.MeasureError) as e:
        submit.cmd_submit(["rf-detr"])
    assert e.value.code == "AUTH"


def test_login_against_a_test_inbox_keeps_only_the_inbox_token(config, monkeypatch):
    monkeypatch.setenv("MTR_INBOX", "http://inbox.test")
    calls = []

    def fake_http(method, url, body=None, token=None, form=False):
        calls.append((method, url, body))
        if url.endswith("/v1/config"):
            return 200, {"test_auth": True}
        return 200, {"token": "inbox-token", "login": "tester", "expires_at": "2099-01-01T00:00:00.000Z"}

    monkeypatch.setattr(submit, "http", fake_http)
    assert submit.cmd_login([]) == 0
    saved = json.loads((config / "credentials.json").read_text())
    assert saved == {"inbox": "http://inbox.test", "token": "inbox-token", "login": "tester", "expires_at": "2099-01-01T00:00:00.000Z"}
    assert oct((config / "credentials.json").stat().st_mode & 0o777) == "0o600"
    assert calls[1] == ("POST", "http://inbox.test/v1/session", {"github_token": "test:tester"})
    assert submit.credentials()["login"] == "tester"
    monkeypatch.setenv("MTR_INBOX", "http://elsewhere.test")
    assert submit.credentials() is None  # a sign-in belongs to one inbox


def test_the_device_flow_waits_for_approval(monkeypatch):
    answers = iter([(200, {"error": "authorization_pending"}), (200, {"error": "slow_down"}), (200, {"access_token": "gh-token"})])

    def fake_http(method, url, body=None, token=None, form=False):
        if url == submit.GITHUB_DEVICE:
            return 200, {"device_code": "d", "user_code": "ABCD-1234", "verification_uri": "https://github.com/login/device", "interval": 0, "expires_in": 60}
        return next(answers)

    monkeypatch.setattr(submit, "http", fake_http)
    monkeypatch.setattr(submit.time, "sleep", lambda s: None)
    assert submit.github_device_flow("client") == "gh-token"


def test_model_commands_run_from_the_committed_lock():
    argv = main.model_command("check", ROOT / "models" / "rf-detr", ["--size", "nano"])
    assert "--frozen" in argv and argv[argv.index("--project") + 1] == str(ROOT / "models" / "rf-detr")
    assert argv[-4:] == ["check", str(ROOT / "models" / "rf-detr"), "--size", "nano"]
    llm = main.model_command("run", ROOT / "models" / "qwen3", [])
    assert llm[llm.index("--project") + 1] == str(paths.tool_dir())


FIXTURES = Path(__file__).with_name("fixtures")


def golden_submissions() -> dict[str, dict]:
    """One image and one language-model submission built by the real record code from synthetic worker data.
    packages/schema's community.test.ts checks the same files with checkSubmission, so Python and TypeScript agree."""
    image_runs = [worker_run([9.5 + k * 0.071 + ((i * 37) % 13) * 0.0123 for i in range(100)]) for k in range(3)]
    image_times = record.times(image_runs, llm=False)
    image = record.build(model="rf-detr", size="nano", weights=[{"file": "rf-detr-nano.pth", "bytes": 1, "sha256": "a" * 64}], runs=image_runs,
                         gpu_name="NVIDIA GeForce RTX 3090", driver="580.95.05", image_digest=None, warmup=10, measured="2026-10-29T10:00:00+09:00",
                         machine={"cpu": "CPU", "gpu_memory_mib": 24576, "compute_capability": "8.6", "os": "Linux", "power_limit_w": 350.0, "power_default_w": 350.0},
                         provenance={"tool": {"name": "mtr", "version": "0.2.0", "commit": "b" * 40, "dirty": False},
                                     "adapter_sha256": "c" * 64, "lock_sha256": "d" * 64, "input_sha256": [TIMING_SHA]})

    def request(k: int, i: int) -> dict:
        return {"ttft_ms": 31.7 + k * 0.9 + (i % 4) * 0.137, "e2e_ms": 2411.3 + i * 1.7, "prompt_n": 512, "prompt_ms": 120.0,
                "predicted_n": 256, "predicted_ms": 2290.0, "prefill_tps": 4123.4 + i * 3.3, "decode_tps": 104.21 + k * 0.33 + (i % 3) * 0.071, "text": "x"}

    llm_runs = [{"worker": {"requests": [request(k, i) for i in range(10)], "gpu_layers": "37/37", "python": "3.12.3"},
                 "baseline_bytes": 400 * 2**20, "peak_bytes": 6800 * 2**20} for k in range(3)]
    llm = record.build_llm(model="qwen3", size="8b-q4_k_m", weights=[{"file": "Qwen3-8B-Q4_K_M.gguf", "bytes": 1, "sha256": "a" * 64}], runs=llm_runs,
                           quant="Q4_K_M", engine={"name": "llama.cpp", "release": "v0.7.0", "build": "b11669", "cuda": "12.6", "binary_sha256": "f" * 64},
                           server_args=["-c", "8192"], gen_tokens=256, context=8192, gpu_name="NVIDIA GeForce RTX 3090", driver="580.95.05", warmup=2,
                           measured="2026-10-29T10:00:00+09:00",
                           machine={"cpu": "CPU", "gpu_memory_mib": 24576, "compute_capability": "8.6", "os": "Linux", "power_limit_w": 350.0, "power_default_w": 350.0},
                           provenance={"tool": {"name": "mtr", "version": "0.2.0", "commit": "b" * 40, "dirty": False},
                                       "adapter_sha256": "c" * 64, "lock_sha256": "d" * 64, "input_sha256": None})
    return {"submission-image.json": submit.payload(image, image_times, "golden"),
            "submission-llm.json": submit.payload(llm, record.times(llm_runs, llm=True), "golden")}


def test_golden_submissions_match_their_fixtures():
    import os

    for name, body in golden_submissions().items():
        assert record.validate(body, "community-submission") == [], name
        text = json.dumps(body, indent=2) + "\n"
        path = FIXTURES / name
        if os.environ.get("MTR_UPDATE_FIXTURES") == "1":
            path.parent.mkdir(exist_ok=True)
            path.write_text(text)
        assert path.read_text() == text, f"{name} changed; rerun with MTR_UPDATE_FIXTURES=1 and check community.test.ts"


def test_a_streaming_host_on_the_gpu_gets_the_pause_hook_tip(config):
    busy = checks.before_run([checks.GpuProcess(7, "C", 300, "sunshine")], checks.Samples([600 * 2**20] * 200, [5.0] * 199, [0] * 200))
    assert busy.codes == ["GPU_BUSY"] and 'pause = ["sunshine"]' in busy.errors[0]
    other = checks.before_run([checks.GpuProcess(8, "C", 300, "python3")], checks.Samples([600 * 2**20] * 200, [5.0] * 199, [0] * 200))
    assert "pause =" not in other.errors[0]


def test_each_model_names_the_cuda_its_environment_needs():
    assert machine.cuda_of_lock(ROOT / "models" / "rf-detr" / "uv.lock") == "13.0"
    assert machine.cuda_of_lock(ROOT / "models" / "gtr" / "uv.lock") == "12.8"
    assert main.model_cuda(ROOT / "models" / "qwen3") is None  # llama.cpp is built for the local GPU
    assert machine.cuda_fits("12.8", "13.0") and not machine.cuda_fits("13.0", "12.4") and machine.cuda_fits("13.0", None)
    assert machine.driver_advice("13.0") == "NVIDIA driver 580 or newer"


def test_a_model_the_driver_cannot_run_stops_before_anything_downloads(monkeypatch):
    monkeypatch.setattr(machine, "driver_cuda", lambda: "12.4")
    with pytest.raises(errors.MeasureError) as e:
        main.run_in_model("check", ROOT / "models" / "rf-detr", ["--size", "nano"])
    assert e.value.code == "UNSUPPORTED" and "580" in str(e.value)


def test_submit_without_records_says_to_measure_first(config, tmp_path, monkeypatch):
    model = tmp_path / "models" / "fake"
    model.mkdir(parents=True)
    (model / "adapter.py").write_text('TASK = "object-detection"\nSIZES = ["s"]\n')
    with pytest.raises(errors.MeasureError) as e:
        submit.cmd_submit([str(model)])
    assert e.value.code == "NOT_SUBMITTABLE" and "mtr bench" in str(e.value)
