"""The worker for language models (KIND = "llm"): one llama-server run, driven over HTTP.

The parent starts this worker after taking the idle baseline and polls device memory while it
runs, exactly as for image models. The worker starts the pinned llama-server with the protocol's
flags, waits until the model is loaded, then sends the fixed workload (protocol.py):

- the first LLM_PROMPT_TOKENS tokens of llm_prompt.txt, as this model's tokenizer splits it,
  sent as token ids so every model reads exactly that many
- LLM_GEN_TOKENS generated tokens, greedy, with the end-of-sequence token ignored
- prompt caching off, so every request runs the whole prefill again

For each request it records the time to the first streamed token (client side), the end-to-end
time, and the server's own prefill and decode rates (`timings`). All layers must sit on the GPU;
if llama-server can't fit them at LLM_CONTEXT, the result is OOM. It stops the server before it
exits, so the parent's second baseline sees the GPU free again.
Usage: python -m mtr_measure.llm <spec.json>
"""

from __future__ import annotations

import json
import platform
import re
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

from .errors import MeasureError

PROMPT_FILE = Path(__file__).with_name("llm_prompt.txt")
READY_TIMEOUT_S = 600
# Only memory failures. "failed to load model" also follows an unsupported architecture or a bad file,
# and those are ENGINE failures, never "doesn't fit".
OOM_SIGNS = ("out of memory", "failed to allocate", "cudamalloc failed", "unable to allocate")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(port: int, path: str, body: dict | None = None, timeout: float = 600):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def server_failure(log_path: Path) -> MeasureError:
    tail = log_path.read_text(errors="replace")[-4000:] if log_path.is_file() else ""
    lines = [l for l in tail.splitlines() if l.strip()][-6:]
    detail = " | ".join(lines) or "no output"
    if any(s in tail.lower() for s in OOM_SIGNS):
        return MeasureError("OOM", f"llama-server could not load the model into GPU memory: {detail}")
    return MeasureError("ENGINE", f"llama-server stopped: {detail}")


def wait_ready(port: int, proc: subprocess.Popen, log_path: Path) -> None:
    deadline = time.monotonic() + READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise server_failure(log_path)
        try:
            with request(port, "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        time.sleep(0.25)
    raise MeasureError("ENGINE", f"llama-server didn't become ready within {READY_TIMEOUT_S} s")


def gpu_layers(log_path: Path) -> tuple[int, int] | None:
    found = re.findall(r"offloaded (\d+)/(\d+) layers to GPU", log_path.read_text(errors="replace"))
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def prompt_tokens(port: int, n: int) -> list[int]:
    text = PROMPT_FILE.read_text()
    with request(port, "/tokenize", {"content": text, "add_special": False}) as r:
        tokens = json.loads(r.read())["tokens"]
    if not tokens:
        raise MeasureError("ENGINE", "The tokenizer returned no tokens for the fixed prompt")
    while len(tokens) < n:
        tokens = tokens + tokens
    return tokens[:n]


def complete(port: int, tokens: list[int], n_predict: int) -> dict:
    """One streamed completion. Times the first token on the client; reads the server's timings."""
    body = {"prompt": tokens, "n_predict": n_predict, "ignore_eos": True, "temperature": 0, "top_k": 1,
            "cache_prompt": False, "stream": True}
    t0 = time.perf_counter()
    first = None
    text: list[str] = []
    timings = None
    with request(port, "/completion", body) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = json.loads(line[5:])
            if first is None:
                first = time.perf_counter()
            text.append(data.get("content", ""))
            if data.get("stop"):
                timings = data.get("timings")
                break
    t1 = time.perf_counter()
    if not timings or first is None:
        raise MeasureError("ENGINE", "llama-server's stream ended without timings")
    if timings["predicted_n"] != n_predict:
        raise MeasureError("ENGINE", f"llama-server generated {timings['predicted_n']} tokens, not {n_predict}")
    return {
        "ttft_ms": (first - t0) * 1000,
        "e2e_ms": (t1 - t0) * 1000,
        "prompt_n": timings["prompt_n"],
        "prompt_ms": timings["prompt_ms"],
        "predicted_n": timings["predicted_n"],
        "predicted_ms": timings["predicted_ms"],
        "prefill_tps": timings["prompt_per_second"],
        "decode_tps": timings["predicted_per_second"],
        "text": "".join(text),
    }


THINKING_LEAK = ("The model is still thinking although the protocol asks for thinking off: llama-server returned "
                 "reasoning_content. Declare THINKING_OFF_ARGS in the adapter, the way the model card switches thinking off")


def check_no_thinking(port: int) -> None:
    """With thinking off, a short chat answer must come back without reasoning_content."""
    body = {"messages": [{"role": "user", "content": "Reply with the single word: ready"}], "max_tokens": 64, "temperature": 0}
    with request(port, "/v1/chat/completions", body) as r:
        msg = json.loads(r.read())["choices"][0]["message"]
    if (msg.get("reasoning_content") or "").strip():
        raise MeasureError("ADAPTER_INTERFACE", THINKING_LEAK)


def run(spec: dict) -> dict:
    server, gguf = Path(spec["server"]), Path(spec["gguf"])
    for f in [gguf, *map(Path, spec.get("more_files", []))]:
        if not f.is_file():
            raise MeasureError("WEIGHTS_MISSING", f"{f} doesn't exist. Run `mtr check` first, which downloads it")
    if not server.is_file():
        raise MeasureError("ENGINE", f"{server} doesn't exist. Run `mtr check` first, which builds it")
    port = free_port()
    log_path = Path(spec["server_log"])
    cmd = [str(server), "-m", str(gguf), *spec["server_args"], "--host", "127.0.0.1", "--port", str(port)]
    t = time.perf_counter()
    with log_path.open("w") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    try:
        wait_ready(port, proc, log_path)
        load_s = time.perf_counter() - t
        layers = gpu_layers(log_path)
        if layers and layers[0] < layers[1]:
            raise MeasureError("OOM", f"Only {layers[0]} of {layers[1]} layers fit on the GPU at the protocol's context")
        tokens = prompt_tokens(port, spec["prompt_tokens"])
        sample = ""
        for i in range(spec["warmup"]):
            out = complete(port, tokens, spec["gen_tokens"])
            if i == 0:
                sample = out["text"][:200]
        requests = []
        for _ in range(spec["timed"]):
            out = complete(port, tokens, spec["gen_tokens"])
            out.pop("text")
            requests.append(out)
        if proc.poll() is not None:
            raise server_failure(log_path)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    return {
        "load_s": load_s,
        "gpu_layers": f"{layers[0]}/{layers[1]}" if layers else "unknown",
        "requests": requests,
        "sample": sample,
        "python": platform.python_version(),
    }


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    try:
        result = {"ok": True, **run(spec)}
    except MeasureError as e:
        result = {"ok": False, "code": e.code, "error": str(e)}
    except Exception as e:
        result = {"ok": False, "code": "ENGINE", "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
    Path(spec["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
