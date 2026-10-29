"""Result codes, and the one line every command ends with.

Every `mtr` command prints `mtr-result: {json}` as its last line, so an agent can read
the outcome with `tail -1` instead of parsing the log. The JSON has:

    status     "ok" or "fail"
    code       one of CODES, "OK" on success
    message    what happened, in one sentence
    hint       what to do next (from CODES), empty on success
    outputs    files the command wrote, and its main numbers
    warnings   things that didn't stop the command

The exit code is fixed per code, so scripts can branch on it too.
"""

from __future__ import annotations

import json
from typing import Any

RESULT_PREFIX = "mtr-result: "

# code: (exit code, what to do next)
CODES: dict[str, tuple[int, str]] = {
    "OK": (0, ""),
    "FAILED": (1, "Read the log above the result line."),
    "USAGE": (2, "Check the command line: `mtr --help`, or `mtr <command> --help`."),
    "UNSUPPORTED": (3, "The tool runs on Linux with an NVIDIA GPU and its driver. `mtr doctor` says what is missing."),
    "ADAPTER_INTERFACE": (10, "The adapter declares something wrongly or misses it. Compare it with models/_template/adapter.py."),
    "ADAPTER_RUNTIME": (11, "load() or predict() raised. Read the traceback in the log, fix the adapter, then run `mtr check` again."),
    "WEIGHTS_MISSING": (12, "weight_files() lists a file that doesn't exist. `mtr check` downloads weights; otherwise fix weight_files()."),
    "OOM": (13, "It didn't fit in the free GPU memory at the protocol's settings. Never lower them. Other programs may hold GPU "
                "memory: close them, or ask whoever uses the machine. If it still doesn't fit, the checkpoint doesn't run on this GPU."),
    "NOT_CHECKED": (14, "Run `mtr check` for this checkpoint first. It must pass with the current adapter.py and uv.lock."),
    "LOCK_STALE": (15, "pyproject.toml and uv.lock disagree. If you changed pyproject.toml on purpose, run `uv lock --project <model dir>` "
                       "and commit uv.lock with it; otherwise restore pyproject.toml with git."),
    "GPU_BUSY": (20, "Another program uses the GPU. Close it, or ask whoever uses the machine. Never close other programs yourself."),
    "BASELINE_HIGH": (21, "Idle GPU memory is higher than usual. If the run fails, find what holds it."),
    "DRIFT": (22, "The idle baseline moved during the run, so something else used the GPU. Run it again."),
    "POLL_GAP": (23, "Memory polling fell behind. Run it again. If it repeats, report it as a tool bug."),
    "VARIANCE": (24, "The independent runs disagreed by more than the protocol allows. Run it again on an idle machine. "
                     "If it repeats, report the spread. Never change MAX_SPREAD."),
    "DESKTOP": (25, "A pause hook (machine.toml) couldn't stop or restart its program. Check that program now."),
    "VRAM_LOW": (26, "Not enough free GPU memory next to the other programs on the GPU. Close what holds it, or ask whoever uses the machine."),
    "CPU_BUSY": (27, "Another program kept the CPU busy (a build, a download, an encoder), and timed calls include CPU work. "
                     "Run it again when the machine is quiet. Never stop other programs yourself."),
    "SCHEMA": (30, "The record fails its schema. If the adapter is fine, this is a tool bug: report it, and never edit the record."),
    "QUEUE": (31, "See `mtr queue list`."),
    "DATASET": (40, "The benchmark data is missing or damaged. Run the command again to download it, or check the disk space."),
    "ENGINE": (41, "The inference engine (llama.cpp) failed to build or start. Read the log; never switch to another build."),
    "EVAL": (42, "The accuracy evaluation failed. Read the log. Never change the benchmark settings to make it pass."),
    "AUTH": (50, "Sign in with `mtr login`, then try again."),
    "NETWORK": (51, "The inbox couldn't be reached. Check the connection, or MTR_INBOX if you set it, and try again."),
    "SUBMIT_REJECTED": (52, "The inbox refused the submission. The message says why; fix that and measure again if needed. Never edit a record."),
    "DIRTY": (53, "The tool or the model directory has uncommitted changes, so its records can't be traced to a public commit. "
                  "Commit or undo the changes, then measure again."),
    "NOT_SUBMITTABLE": (54, "This record can't be submitted. The message says why; usually: measure it again with the current `mtr run`."),
}


class MeasureError(Exception):
    """A failure with a result code. Raised anywhere, turned into the result line at the top."""

    def __init__(self, code: str, message: str):
        if code not in CODES:
            raise ValueError(f"unknown result code {code}")
        super().__init__(message)
        self.code = code


def exit_code(code: str) -> int:
    return CODES[code][0]


def result_line(code: str, message: str, outputs: dict[str, Any] | None = None, warnings: list[str] | None = None) -> str:
    ok = code == "OK"
    body = {
        "status": "ok" if ok else "fail",
        "code": code,
        "message": message,
        "hint": CODES[code][1],
        "outputs": outputs or {},
        "warnings": warnings or [],
    }
    return RESULT_PREFIX + json.dumps(body, ensure_ascii=False)


def emit(code: str, message: str, outputs: dict[str, Any] | None = None, warnings: list[str] | None = None, log=print) -> int:
    """Print the result line through `log` and return the exit code for it."""
    log(result_line(code, message, outputs, warnings))
    return exit_code(code)


def parse_result(text: str) -> dict | None:
    """The last result line in a command's output, or None."""
    for line in reversed(text.splitlines()):
        if line.startswith(RESULT_PREFIX):
            try:
                return json.loads(line[len(RESULT_PREFIX):])
            except json.JSONDecodeError:
                return None
    return None
