"""The output step for language models: one checkpoint answers its task's prompt set.

    mtr outputs <id> --size <size>

It starts the pinned llama-server with the model's own chat template, thinking off the way the model
card says (THINKING_OFF_ARGS) and one slot, asks every prompt of prompts/<task>.toml once with the
set's settings, and writes models/<id>/outputs/<size>/answers.json, plus outputs/<size>.log. It
follows schema/answers.schema.json. Failures are flagged by hand afterwards, against each prompt's check.

Nothing is timed, so nothing is paused.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

from . import llm, llmcli, prompts, record
from .cli import Log
from .errors import MeasureError

# One slot with this much context: every prompt and its answer fit easily.
CONTEXT = 8192


def ask(port: int, s: prompts.PromptSet, p: prompts.Prompt) -> dict:
    messages = ([{"role": "system", "content": s.system}] if s.system else []) + [{"role": "user", "content": p.prompt}]
    body = {"messages": messages, "max_tokens": s.max_tokens, "temperature": s.temperature}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.loads(r.read())
    choice = out["choices"][0]
    if s.thinking == "off" and (choice["message"].get("reasoning_content") or "").strip():
        raise MeasureError("ADAPTER_INTERFACE", llm.THINKING_LEAK)
    text = (choice["message"].get("content") or "").strip()
    finish = "length" if choice.get("finish_reason") == "length" else "stop"
    return {"prompt": p.id, "text": text, "tokens": int(out["usage"]["completion_tokens"]), "finish": finish}


def run(model_dir: Path, model_id: str, size: str, task: str, log: Log, gpu: int = 0) -> int:
    s = prompts.load(task)
    a = llmcli.load(model_dir, size)
    paths = llmcli.weight_paths(a, model_id, size, download=False)
    binary = llmcli.server(download=False, gpu=gpu)
    thinking = list(getattr(a, "THINKING_OFF_ARGS", [])) if s.thinking == "off" else []
    args = ["-c", str(CONTEXT), "-np", "1", "-ngl", "999", "--fit", "off", "--no-webui", *thinking, *getattr(a, "SERVER_ARGS", [])]
    out_dir = model_dir / "outputs" / size
    out_dir.mkdir(parents=True, exist_ok=True)
    server_log = out_dir.parent / f"{size}.server.log"
    log(f"mtr-outputs {model_id} / {size}: {len(s.prompts)} prompts of the {task} set, version {s.version}")
    log(f"command: {shlex.join(sys.argv)}", echo=False)
    log(f"settings: greedy, thinking {s.thinking if thinking or s.thinking != 'off' else 'not switchable'}, at most {s.max_tokens} tokens")

    port = llm.free_port()
    cmd = [str(binary), "-m", str(paths[0]), *args, "--host", "127.0.0.1", "--port", str(port)]
    with server_log.open("w") as f:
        f.write("$ " + " ".join(cmd) + "\n")
        f.flush()
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    answers = []
    try:
        llm.wait_ready(port, proc, server_log)
        for p in s.prompts:
            ans = ask(port, s, p)
            answers.append(ans)
            log(f"{p.id}: {ans['tokens']} tokens{' (cut at the limit)' if ans['finish'] == 'length' else ''}")
            log(f"  {ans['text'][:300]!r}", echo=False)
    finally:
        proc.terminate()
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    eng = llmcli.engine.describe(binary)
    doc = {
        "kind": "answers",
        "task": task,
        "model": model_id,
        "size": size,
        "weights": record.weight_entries(paths),
        "prompt_set": {"task": task, "version": s.version, "sha256": s.sha256},
        "settings": {"max_tokens": s.max_tokens, "temperature": s.temperature, "thinking": s.thinking if thinking else "not switchable",
                     "system": s.system, "server_args": args, "quant": a.WEIGHTS[size]["quant"]},
        "engine": {"name": eng["name"], "release": eng["release"], "build": eng["build"], "binary_sha256": eng["binary_sha256"]},
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "answers": answers,
    }
    problems = record.validate(doc, "answers")
    if problems:
        raise MeasureError("SCHEMA", f"answers.json fails the schema: {problems[0]}")
    path = out_dir / "answers.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)
    cut = sum(1 for x in answers if x["finish"] == "length")
    log(f"wrote {path}" + (f" ({cut} answer(s) cut at {s.max_tokens} tokens)" if cut else ""))
    log("next: read each answer against its check and flag the failures")
    log_path = out_dir.parent / f"{size}.log"
    log_path.write_text(log.text())
    print(f"log: {log_path}")
    return 0
