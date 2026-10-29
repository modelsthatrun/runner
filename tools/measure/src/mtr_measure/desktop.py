"""Pause hooks: stop the programs that share the GPU for a timed run, and start them again right afterwards.

A machine asks for them by name in ~/.config/modelsthatrun/machine.toml (machine.py), for example
`pause = ["sunshine", "steam"]`. Without that file nothing is paused, and the checks only refuse to run
while another program computes on the GPU. Games are never closed automatically.

    sunshine   the Sunshine streaming host (Flatpak, as a systemd user unit) and its liveness timer.
               Someone may be reaching the desktop through it, so it must come back as soon as a run
               ends, even when the run fails: a fallback timer starts it again after FALLBACK_MINUTES
               in case this process dies first.
    steam      the Steam client (Flatpak)

Several timed runs share one pause through the queue (`mtr queue run`), which pushes the fallback
timer back before each run (extend_fallback). Any command can also be wrapped:
    uv run --project tools/measure python -m mtr_measure.desktop -- bash -c 'mtr run …; mtr run …'
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from . import checks, machine
from .errors import MeasureError

SUNSHINE_UNIT = "app-dev.lizardbyte.app.Sunshine.service"
# Restarts Sunshine whenever its probe fails, so it pauses with Sunshine.
SUNSHINE_LIVENESS_TIMER = "sunshine-liveness.timer"
STEAM_FLATPAK = "com.valvesoftware.Steam"
FALLBACK_MINUTES = 10
STOP_TIMEOUT_S = 60
START_TIMEOUT_S = 45
# Set while paused, so measurements inside a wrapped batch don't pause again.
PAUSED_ENV = "MTR_DESKTOP_PAUSED"


def _run(cmd: list[str], timeout: float = 30) -> bool:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=timeout).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _unit_active(unit: str) -> bool:
    return _run(["systemctl", "--user", "is-active", "--quiet", unit])


def _wait_until(done: Callable[[], bool], seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if done():
            return True
        time.sleep(0.5)
    return False


def _wait_gone(label: str) -> bool:
    return _wait_until(lambda: label not in checks.running_blockers(), STOP_TIMEOUT_S)


@dataclass
class Program:
    label: str  # as in checks.PAUSABLE_PROGRAMS
    is_running: Callable[[], bool]
    stop: Callable[[], bool]
    start: Callable[[], bool]


class Sunshine:
    def __init__(self) -> None:
        self.units: list[str] = []
        self.armed = 0
        self.fallback = ""

    def is_running(self) -> bool:
        return _unit_active(SUNSHINE_UNIT)

    def arm(self) -> None:
        """(Re)start the fallback timer, FALLBACK_MINUTES from now. A batch calls this before each run."""
        previous = self.fallback
        self.armed += 1
        self.fallback = f"mtr-sunshine-fallback-{os.getpid()}-{self.armed}"
        systemctl = shutil.which("systemctl") or "/usr/bin/systemctl"
        _run(["systemd-run", "--user", "--collect", f"--unit={self.fallback}", f"--on-active={FALLBACK_MINUTES}min",
              systemctl, "--user", "start", *self.units])
        if previous:
            _run(["systemctl", "--user", "stop", f"{previous}.timer"])

    def stop(self) -> bool:
        global _armed_sunshine
        self.units = [u for u in (SUNSHINE_UNIT, SUNSHINE_LIVENESS_TIMER) if _unit_active(u)]
        self.arm()
        _armed_sunshine = self
        _run(["systemctl", "--user", "stop", *reversed(self.units)])
        return _wait_gone("Sunshine streaming host")

    def start(self) -> bool:
        # ExecStartPre sleeps, so wait for "active". Only a confirmed start cancels the fallback timer.
        for attempt in range(2):
            if attempt:
                _run(["systemctl", "--user", "reset-failed", SUNSHINE_UNIT])
            _run(["systemctl", "--user", "start", *self.units], timeout=START_TIMEOUT_S)
            if _wait_until(lambda: _unit_active(SUNSHINE_UNIT), START_TIMEOUT_S):
                _run(["systemctl", "--user", "stop", f"{self.fallback}.timer"])
                return True
        return False


# The Sunshine this process paused, so a batch can push its fallback timer back (extend_fallback).
_armed_sunshine: Sunshine | None = None


def extend_fallback() -> None:
    """Restart the fallback timer before each run of a batch, so a batch longer than FALLBACK_MINUTES
    doesn't get Sunshine back in the middle of a run. Each single run must still be shorter."""
    if _armed_sunshine is not None:
        _armed_sunshine.arm()


class Steam:
    def is_running(self) -> bool:
        return "Steam" in checks.running_blockers()

    def stop(self) -> bool:
        _run(["flatpak", "run", STEAM_FLATPAK, "-shutdown"], timeout=STOP_TIMEOUT_S)
        if _wait_gone("Steam"):
            return True
        _run(["flatpak", "kill", STEAM_FLATPAK])
        return _wait_gone("Steam")

    def start(self) -> bool:
        # A transient user service: it gets the session's display and lives outside this process tree.
        return _run(["systemd-run", "--user", "--collect", "flatpak", "run", STEAM_FLATPAK, "-silent"])


def _sunshine() -> Program:
    s = Sunshine()
    return Program(checks.PAUSABLE_PROGRAMS["sunshine"], s.is_running, s.stop, s.start)


def _steam() -> Program:
    s = Steam()
    return Program(checks.PAUSABLE_PROGRAMS["steam"], s.is_running, s.stop, s.start)


HOOKS: dict[str, Callable[[], Program]] = {"sunshine": _sunshine, "steam": _steam}


def programs() -> list[Program]:
    """The pause hooks machine.toml asks for, in its order."""
    names = machine.pause_hooks()
    unknown = [n for n in names if n not in HOOKS]
    if unknown:
        raise MeasureError("USAGE", f"machine.toml names unknown pause hooks {unknown}. Known hooks: {sorted(HOOKS)}")
    return [HOOKS[n]() for n in names]


def _raise_on_signal(signum, frame):
    raise SystemExit(128 + signum)


@contextmanager
def paused(say: Callable[[str], None] = print) -> Iterator[None]:
    """Stop the configured programs if they run, and start them again on the way out, whatever happens."""
    if os.environ.get(PAUSED_ENV):
        yield
        return
    # SIGTERM and SIGHUP would skip the restart below. Turn them into SystemExit.
    old = {s: signal.signal(s, _raise_on_signal) for s in (signal.SIGTERM, signal.SIGHUP)}
    stopped: list[Program] = []
    try:
        for p in programs():
            if p.is_running():
                say(f"pausing {p.label} until the run ends")
                stopped.append(p)
                if not p.stop():
                    say(f"{p.label} didn't stop within {STOP_TIMEOUT_S} s")
        if stopped:
            time.sleep(2)  # let the driver release their memory before the baseline
        os.environ[PAUSED_ENV] = "1"
        yield
    finally:
        global _armed_sunshine
        os.environ.pop(PAUSED_ENV, None)
        _armed_sunshine = None
        for p in reversed(stopped):
            say(f"starting {p.label} again")
            if not p.start():
                say(f"FAILED to start {p.label} again. Sunshine's fallback timer still starts it within {FALLBACK_MINUTES} minutes.")
        for s, h in old.items():
            signal.signal(s, h)


def main() -> int:
    args = sys.argv[1:]
    if args[:1] == ["--"]:
        args = args[1:]
    if not args:
        print("usage: python -m mtr_measure.desktop -- <command…>", file=sys.stderr)
        return 2
    with paused():
        return subprocess.run(args).returncode


if __name__ == "__main__":
    raise SystemExit(main())
