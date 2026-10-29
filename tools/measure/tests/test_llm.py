"""Tests for the language-model path that need no GPU: a fake llama-server stands in."""

import json
import sys
import textwrap
import types
from pathlib import Path

import pytest

from mtr_measure import adapter as ad
from mtr_measure import llm, protocol, record

FAKE_SERVER = textwrap.dedent('''
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    args = sys.argv[1:]
    port = int(args[args.index("--port") + 1])
    if "--fail" in args:
        print("ggml_backend_cuda_buffer_type_alloc_buffer: allocating 30000 MiB on device 0: cudaMalloc failed: out of memory", flush=True)
        sys.exit(1)
    layers = "2/3" if "--partial" in args else "3/3"
    print(f"load_tensors: offloaded {layers} layers to GPU", flush=True)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def _json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        def do_GET(self):
            self._json({"status": "ok"})
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path == "/tokenize":
                return self._json({"tokens": list(range(100))})
            if self.path == "/v1/chat/completions":
                text = body["messages"][-1]["content"][:20]
                msg = {"content": f"echo: {text}", **({"reasoning_content": "Okay, let me think."} if "--think" in args else {})}
                return self._json({"choices": [{"message": msg, "finish_reason": "stop"}], "usage": {"completion_tokens": 7}})
            n = body["n_predict"]
            assert body["cache_prompt"] is False and body["ignore_eos"] is True
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(n):
                last = i == n - 1
                ev = {"content": "x", "stop": last}
                if last:
                    ev["timings"] = {"prompt_n": len(body["prompt"]), "prompt_ms": 10.0, "predicted_n": n,
                                     "predicted_ms": 100.0, "prompt_per_second": 51200.0, "predicted_per_second": 10.0 * n}
                self.wfile.write(f"data: {json.dumps(ev)}\\n\\n".encode())
                self.wfile.flush()
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
''')


@pytest.fixture
def server(tmp_path):
    script = tmp_path / "llama-server"
    script.write_text(f"#!{sys.executable}\n" + FAKE_SERVER)
    script.chmod(0o755)
    gguf = tmp_path / "m.gguf"
    gguf.write_bytes(b"gguf")
    return script, gguf


def spec(tmp_path, server, args=()):
    script, gguf = server
    return {"server": str(script), "gguf": str(gguf), "server_args": list(args), "prompt_tokens": 512, "gen_tokens": 8,
            "warmup": 1, "timed": 3, "server_log": str(tmp_path / "server.log")}


def test_worker_drives_the_server(tmp_path, server):
    out = llm.run(spec(tmp_path, server))
    assert out["gpu_layers"] == "3/3"
    assert len(out["requests"]) == 3
    r = out["requests"][0]
    # 100 prompt tokens repeated up to exactly 512, and every requested token generated
    assert r["prompt_n"] == 512 and r["predicted_n"] == 8 and r["decode_tps"] == 80.0
    assert r["ttft_ms"] > 0 and r["e2e_ms"] >= r["ttft_ms"]


def test_worker_reports_oom(tmp_path, server):
    with pytest.raises(llm.MeasureError) as e:
        llm.run(spec(tmp_path, server, ["--fail"]))
    assert e.value.code == "OOM"
    with pytest.raises(llm.MeasureError) as e:
        llm.run(spec(tmp_path, server, ["--partial"]))
    assert e.value.code == "OOM" and "2 of 3" in str(e.value)


def test_worker_needs_its_files(tmp_path, server):
    s = spec(tmp_path, server)
    s["gguf"] = str(tmp_path / "missing.gguf")
    with pytest.raises(llm.MeasureError) as e:
        llm.run(s)
    assert e.value.code == "WEIGHTS_MISSING"


def llm_adapter(**over):
    a = types.SimpleNamespace(
        KIND="llm", SIZES=["8b-q4_k_m"],
        WEIGHTS={"8b-q4_k_m": {"repo": "org/M-GGUF", "revision": "a" * 40, "files": ["m.gguf"], "quant": "Q4_K_M"}},
    )
    for k, v in over.items():
        setattr(a, k, v)
    return a


def test_llm_interface():
    ad.check_interface(llm_adapter())
    ad.check_interface(llm_adapter(SERVER_ARGS=["--jinja"]))
    with pytest.raises(ad.AdapterError, match="40-hex"):
        ad.check_interface(llm_adapter(WEIGHTS={"8b-q4_k_m": {"repo": "o/m", "revision": "main", "files": ["m.gguf"], "quant": "Q4"}}))
    with pytest.raises(ad.AdapterError, match="protocol sets"):
        ad.check_interface(llm_adapter(SERVER_ARGS=["-c", "32768"]))
    with pytest.raises(ad.AdapterError, match="protocol sets"):
        ad.check_interface(llm_adapter(SERVER_ARGS=["--ctx-size=4096"]))
    with pytest.raises(ad.AdapterError, match="THINKING_OFF_ARGS"):
        ad.check_interface(llm_adapter(THINKING_OFF_ARGS=["-np", "4"]))
    with pytest.raises(ad.AdapterError, match="no entry"):
        ad.check_interface(llm_adapter(SIZES=["8b-q4_k_m", "8b-q8_0"]))


def test_kind_of_file(tmp_path):
    f = tmp_path / "adapter.py"
    f.write_text('"""doc"""\nKIND = "llm"\n')
    assert ad.kind_of_file(f) == "llm"
    f.write_text("SIZES = []\n")
    assert ad.kind_of_file(f) == "image"


def test_llm_record_passes_its_schema(tmp_path, server):
    worker = {**llm.run(spec(tmp_path, server)), "python": "3.12.9"}
    runs = [{"worker": worker, "baseline_bytes": 800 * 2**20, "peak_bytes": (800 + m) * 2**20} for m in (1500, 1520, 1510)]
    w = tmp_path / "m.gguf"
    rec = record.build_llm(
        model="qwen3", size="0.6b-q8_0", weights=record.weight_entries([w]), runs=runs, quant="Q8_0",
        engine={"name": "llama.cpp", "release": "v0.7.0", "build": "b11669", "cuda": "12.4", "binary_sha256": "b" * 64},
        server_args=protocol.LLM_SERVER_ARGS, gen_tokens=8, context=protocol.LLM_CONTEXT,
        gpu_name="NVIDIA GeForce RTX 4090", driver="580.173.02", warmup=1, measured="2026-10-26T10:00:00+09:00",
    )
    assert record.validate(rec, "llm-run-record") == []
    m = rec["measurement"]
    assert m["vram_peak_mib"] == 1520 and m["decode_tps"] == 80.0 and m["prompt_tokens"] == 512
    assert record.spread(m["runs"], "decode_tps") == 0
    # It is not an image run record, so the image record schema refuses it.
    assert record.validate(rec, "run-record")


def test_eval_halves_slots_until_the_desktop_keeps_its_margin(tmp_path, monkeypatch):
    from mtr_measure import llmeval

    started = []

    class Proc:
        def poll(self):
            return None

    def fake_start(binary, gguf, args, log_path):
        started.append(int(args[args.index("-np") + 1]))
        return Proc(), 1

    monkeypatch.setattr(llmeval, "start_server", fake_start)
    monkeypatch.setattr(llmeval, "stop", lambda proc: None)
    monkeypatch.setattr(llmeval.llm, "check_no_thinking", lambda port: None)
    # 8 and 4 slots leave too little memory free; 2 leave enough.
    left = iter([200, 600, protocol.EVAL_VRAM_MARGIN_MIB + 1])
    monkeypatch.setattr(llmeval, "free_mib", lambda index: next(left))
    monkeypatch.setattr(llmeval, "read_check", lambda d, s: {})
    monkeypatch.setattr(llmeval.llmcli, "load", lambda d, s: llm_adapter())
    monkeypatch.setattr(llmeval.llmcli, "weight_paths", lambda *a, **k: [tmp_path / "m.gguf"])
    monkeypatch.setattr(llmeval.llmcli, "server", lambda download, **k: tmp_path / "llama-server")

    def stop_here(*a, **k):
        raise llmeval.MeasureError("EVAL", "stop after the server started")

    monkeypatch.setattr(llmeval.subprocess, "run", stop_here)
    model = tmp_path / "m"
    model.mkdir()
    from mtr_measure.cli import Log

    assert llmeval.main([str(model), "--size", "8b-q4_k_m", "--benchmark", "ifeval"], Log()) == 42
    assert started == [8, 4, 2]


def test_the_chat_prompt_set_loads():
    from mtr_measure import prompts

    s = prompts.load("chat")
    assert s.version >= 1 and 10 <= len(s.prompts) <= 20  # the set is planned at 10-20 prompts
    assert s.temperature == 0 and s.thinking == "off" and s.max_tokens >= 256
    assert len({p.id for p in s.prompts}) == len(s.prompts)
    assert len({p.category for p in s.prompts}) == len(s.prompts)  # one prompt per kind of skill
    assert all(p.check for p in s.prompts)
    with pytest.raises(prompts.MeasureError):
        prompts.load("no-such-task")


def test_output_step_answers_every_prompt(tmp_path, server, monkeypatch):
    from types import SimpleNamespace

    from mtr_measure import llmcli, llmoutputs, prompts
    from mtr_measure.cli import Log

    script, gguf = server
    model = tmp_path / "models" / "fake-chat"
    model.mkdir(parents=True)
    a = llm_adapter(SIZES=["s"], WEIGHTS={"s": {"repo": "o/m", "revision": "a" * 40, "files": ["m.gguf"], "quant": "Q8_0"}}, THINKING_OFF_ARGS=["--x"])
    monkeypatch.setattr(llmcli, "load", lambda d, s: a)
    monkeypatch.setattr(llmcli, "weight_paths", lambda *args, **k: [gguf])
    monkeypatch.setattr(llmcli, "server", lambda download, **k: script)
    monkeypatch.setattr(llmcli.engine, "describe", lambda b: {"name": "llama.cpp", "release": "v0.7.0", "build": "b11669", "cuda": "12.4", "binary_sha256": "b" * 64})
    assert llmoutputs.run(model, "fake-chat", "s", "chat", Log()) == 0
    out = json.loads((model / "outputs" / "s" / "answers.json").read_text())
    s = prompts.load("chat")
    assert [x["prompt"] for x in out["answers"]] == [p.id for p in s.prompts]
    assert out["prompt_set"]["sha256"] == s.sha256 and out["settings"]["thinking"] == "off"
    assert all(x["text"].startswith("echo: ") and x["finish"] == "stop" for x in out["answers"])
    assert record.validate(out, "answers") == []


def test_output_step_stops_when_thinking_leaks(tmp_path, server, monkeypatch):
    from mtr_measure import llmcli, llmoutputs
    from mtr_measure.cli import Log

    script, gguf = server
    model = tmp_path / "models" / "thinker"
    model.mkdir(parents=True)
    # No THINKING_OFF_ARGS, and the fake server keeps thinking.
    a = llm_adapter(SIZES=["s"], WEIGHTS={"s": {"repo": "o/m", "revision": "a" * 40, "files": ["m.gguf"], "quant": "Q8_0"}}, SERVER_ARGS=["--think"])
    monkeypatch.setattr(llmcli, "load", lambda d, s: a)
    monkeypatch.setattr(llmcli, "weight_paths", lambda *args, **k: [gguf])
    monkeypatch.setattr(llmcli, "server", lambda download, **k: script)
    with pytest.raises(llm.MeasureError) as e:
        llmoutputs.run(model, "thinker", "s", "chat", Log())
    assert e.value.code == "ADAPTER_INTERFACE" and "THINKING_OFF_ARGS" in str(e.value)
    assert not (model / "outputs" / "s" / "answers.json").exists()


def test_metric_keys_keep_the_filter():
    from mtr_measure.llmeval import metric_key

    assert metric_key("prompt_level_strict_acc,none") == "prompt_level_strict_acc"
    assert metric_key("exact_match,flexible-extract") == "exact_match_flexible-extract"
    assert metric_key("exact_match,strict-match") == "exact_match_strict-match"
