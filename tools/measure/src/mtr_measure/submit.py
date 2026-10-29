"""Sending results to ModelsThatRun: sign in, submit run records, follow their review.

    mtr login | logout | whoami      GitHub's device flow; the inbox keeps your GitHub login, never your GitHub token
    mtr results                      the run records in this checkout, and which were sent
    mtr submit <model> [--size …]    send run records with their raw timings, after you confirm
    mtr submissions                  what you sent, and the review's state and note

A submission is one run record from `mtr run`, its <size>.times.json and your note. The inbox checks
them and holds them for review; accepted ones show on the model's page under "Community results", with
your GitHub login, the GPU and the date. Records are dedicated to the public domain (CC0 1.0).

The inbox is MTR_INBOX (default https://inbox.modelsthatrun.dev). Credentials live in
~/.config/modelsthatrun/credentials.json, readable by you only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from . import adapter as ad
from . import machine, paths, protocol, record
from .cli import Parser
from .errors import MeasureError, emit

DEFAULT_INBOX = "https://inbox.modelsthatrun.dev"
GITHUB_DEVICE = "https://github.com/login/device/code"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
LICENSE = "CC0-1.0"


def inbox() -> str:
    return os.environ.get("MTR_INBOX", DEFAULT_INBOX).rstrip("/")


def cred_path() -> Path:
    return machine.config_dir() / "credentials.json"


def credentials() -> dict | None:
    """The saved sign-in for the current inbox, if it hasn't expired."""
    path = cred_path()
    if not path.is_file():
        return None
    try:
        c = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if c.get("inbox") != inbox() or c.get("expires_at", "") < datetime.now(timezone.utc).isoformat():
        return None
    return c


def http(method: str, url: str, body: dict | None = None, token: str | None = None, form: bool = False) -> tuple[int, dict]:
    headers = {"Accept": "application/json", "User-Agent": f"mtr/{machine.TOOL_VERSION}"}
    data = None
    if body is not None:
        if form:
            data = urllib.parse.urlencode(body).encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        else:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except json.JSONDecodeError:
            return e.code, {"error": e.reason}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise MeasureError("NETWORK", f"Can't reach {url}: {getattr(e, 'reason', e)}") from None


# --- sign-in -------------------------------------------------------------------------------------


def github_device_flow(client_id: str) -> str:
    """GitHub's device flow: the user approves this machine in a browser, and GitHub hands back a token."""
    status, start = http("POST", GITHUB_DEVICE, {"client_id": client_id, "scope": ""}, form=True)
    if status != 200 or "device_code" not in start:
        raise MeasureError("AUTH", f"GitHub didn't start the sign-in: {start.get('error_description') or start.get('error') or status}")
    print(f"Open {start['verification_uri']} and enter the code {start['user_code']}", flush=True)
    print("Waiting for you to approve it on GitHub…", flush=True)
    interval = int(start.get("interval", 5))
    deadline = time.monotonic() + int(start.get("expires_in", 900))
    while time.monotonic() < deadline:
        time.sleep(interval)
        _, r = http("POST", GITHUB_TOKEN, {"client_id": client_id, "device_code": start["device_code"],
                                           "grant_type": "urn:ietf:params:oauth:grant-type:device_code"}, form=True)
        if "access_token" in r:
            return r["access_token"]
        err = r.get("error")
        if err == "slow_down":
            interval += 5
        elif err != "authorization_pending":
            raise MeasureError("AUTH", f"GitHub sign-in failed: {r.get('error_description') or err}")
    raise MeasureError("AUTH", "The code expired before it was approved. Run `mtr login` again")


def cmd_login(argv: list[str]) -> int:
    Parser(prog="mtr login", description="Sign in to the inbox with your GitHub account (device flow).").parse_args(argv)
    status, config = http("GET", f"{inbox()}/v1/config")
    if status != 200:
        raise MeasureError("NETWORK", f"{inbox()} didn't answer /v1/config ({status})")
    if config.get("test_auth"):
        # A local inbox for development takes a made-up login instead of GitHub (never in production).
        github_token = f"test:{os.environ.get('MTR_TEST_LOGIN', 'tester')}"
        print(f"{inbox()} is a test inbox: signing in as {github_token[5:]} without GitHub", flush=True)
    else:
        github_token = github_device_flow(config["github_client_id"])
    status, session = http("POST", f"{inbox()}/v1/session", {"github_token": github_token})
    if status != 200:
        raise MeasureError("AUTH", f"The inbox refused the sign-in: {session.get('error', status)}")
    path = cred_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump({"inbox": inbox(), "token": session["token"], "login": session["login"], "expires_at": session["expires_at"]}, f)
    print(f"Signed in as {session['login']} until {session['expires_at'][:10]}. Your GitHub token was not kept.")
    return emit("OK", f"Signed in as {session['login']}", {"login": session["login"], "expires_at": session["expires_at"]})


def cmd_logout(argv: list[str]) -> int:
    Parser(prog="mtr logout").parse_args(argv)
    c = credentials()
    if c:
        http("DELETE", f"{inbox()}/v1/session", token=c["token"])
    cred_path().unlink(missing_ok=True)
    return emit("OK", "Signed out")


def cmd_whoami(argv: list[str]) -> int:
    Parser(prog="mtr whoami").parse_args(argv)
    c = credentials()
    if not c:
        raise MeasureError("AUTH", f"Not signed in to {inbox()}")
    return emit("OK", f"{c['login']} on {c['inbox']}", {"login": c["login"], "inbox": c["inbox"], "expires_at": c["expires_at"]})


# --- results -------------------------------------------------------------------------------------


def records(model_dir: Path) -> list[tuple[str, Path]]:
    """(size, record path) for each run record of a model: runs/<size>.json."""
    runs = model_dir / "runs"
    if not runs.is_dir():
        return []
    return [(f.name[: -len(".json")], f) for f in sorted(runs.glob("*.json")) if f.name.count(".") == 1]


def sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def problems(rec: dict, times_path: Path, task: str | None = None) -> list[str]:
    """Why a record can't be submitted, checked here before the inbox checks it again. `task` is the
    adapter's TASK: an image model must have been timed on that task's timing image."""
    out = []
    p = rec.get("provenance")
    if not p:
        out.append("it has no provenance: it was measured before the tool recorded it. Measure it again")
        return out
    if not p["tool"].get("commit"):
        out.append("it wasn't measured from a git checkout")
    if p["tool"].get("dirty") is not False:
        out.append("the tool or the model had uncommitted changes when it was measured")
    if rec["measurement"].get("protocol") != protocol.PROTOCOL_VERSION:
        out.append(f"it was measured under protocol {rec['measurement'].get('protocol')}, and the current one is {protocol.PROTOCOL_VERSION}")
    if not times_path.is_file():
        out.append(f"{times_path.name} is missing")
    if rec.get("kind", "image") == "image":
        from . import timing

        want = timing.sha_for(task)
        if want is None or p.get("input_sha256") != [want]:
            out.append("it was measured on another input than its task's timing image (timing/timing.toml). Measure it again without --input")
    return out


def cmd_results(argv: list[str]) -> int:
    Parser(prog="mtr results", description="List the run records in this checkout, and which were sent.").parse_args(argv)
    rows = []
    from .main import adapters

    for d in adapters():
        for size, path in records(d):
            rec = json.loads(path.read_text())
            m = rec["measurement"]
            sent = path.with_name(f"{size}.submitted.json")
            sent_info = json.loads(sent.read_text()) if sent.is_file() else None
            fresh = sent_info and sent_info.get("record_sha256") == sha256_text(path)
            why = problems(rec, path.with_name(f"{size}.times.json"), ad.task_of_file(d / "adapter.py"))
            speed = f"{m['decode_tps']} tok/s" if "decode_tps" in m else f"{m['latency_ms']['median']} ms"
            state = f"sent ({sent_info['id']})" if fresh else ("can't be sent: " + why[0] if why else "ready to send")
            rows.append({"model": d.name, "size": size, "gpu": m["env"]["gpu"], "speed": speed, "vram_peak_mib": m["vram_peak_mib"],
                         "measured": m["measured"], "state": state})
            print(f"{d.name} / {size}: {speed}, {m['vram_peak_mib']} MiB on {m['env']['gpu']}, {m['measured'][:10]}  [{state}]")
    if not rows:
        print("No run records yet. `mtr bench <model>` makes some.")
    return emit("OK", f"{len(rows)} record(s)", {"records": rows})


# --- submit --------------------------------------------------------------------------------------


def payload(rec: dict, times: dict, message: str) -> dict:
    return {"record": rec, "times": times, "message": message, "client": {"name": "mtr", "version": machine.TOOL_VERSION}, "license": LICENSE}


def preview(login: str, model: str, size: str, rec: dict) -> str:
    m = rec["measurement"]
    env = m["env"]
    speed = (f"{m['decode_tps']} tokens/s decode, first token {m['ttft_ms']['median']} ms" if "decode_tps" in m
             else f"{m['latency_ms']['median']} ms median, {m['precision']}, input {m['input']}")
    return (f"  {model} / {size}: {speed}, VRAM peak {m['vram_peak_mib']} MiB\n"
            f"    on {env['gpu']} ({env.get('gpu_memory_mib')} MiB, power limit {env.get('power_limit_w')} W), driver {env['driver']}\n"
            f"    cpu {env.get('cpu')}, {env.get('os')}; measured {m['measured']}; tool {rec['provenance']['tool']['commit'][:12]}")


def cmd_submit(argv: list[str]) -> int:
    p = Parser(prog="mtr submit", description="Send run records for review. Accepted ones show on modelsthatrun.dev under your GitHub login.")
    p.add_argument("model_dir", type=paths.model_dir, help="The model: an id such as rf-detr, or its directory")
    p.add_argument("--size", action="append", dest="sizes", help="A size to send. Repeat for several (default: every record not sent yet)")
    p.add_argument("-m", "--message", default="", help="A note for the reviewer: anything unusual about this machine or run")
    p.add_argument("--yes", action="store_true", help="Send without asking. Only when the person whose GitHub account this is has agreed")
    p.add_argument("--dry-run", action="store_true", help="Check and show what would be sent, without sending")
    args = p.parse_args(argv)
    model_dir = args.model_dir
    if not (model_dir / "adapter.py").is_file():
        raise MeasureError("USAGE", f"No model {str(model_dir)!r}. `mtr models` lists them.")
    c = credentials()
    if not c and not args.dry_run:
        raise MeasureError("AUTH", f"Not signed in to {inbox()}")
    login = c["login"] if c else "(not signed in)"

    chosen = [(s, f) for s, f in records(model_dir) if not args.sizes or s in args.sizes]
    if args.sizes and len(chosen) < len(args.sizes):
        missing = sorted(set(args.sizes) - {s for s, _ in chosen})
        raise MeasureError("NOT_SUBMITTABLE", f"No run record for {', '.join(missing)}. Measure it with `mtr run`")
    todo, skipped = [], []
    for size, path in chosen:
        sent = path.with_name(f"{size}.submitted.json")
        if not args.sizes and sent.is_file() and json.loads(sent.read_text()).get("record_sha256") == sha256_text(path):
            skipped.append(size)
            continue
        rec = json.loads(path.read_text())
        times_path = path.with_name(f"{size}.times.json")
        why = problems(rec, times_path, ad.task_of_file(model_dir / "adapter.py"))
        if why:
            raise MeasureError("NOT_SUBMITTABLE", f"{model_dir.name} / {size}: {why[0]}")
        times = json.loads(times_path.read_text())
        body = payload(rec, times, args.message)
        errs = record.validate(body, "community-submission")
        if errs:
            raise MeasureError("NOT_SUBMITTABLE", f"{model_dir.name} / {size} fails the submission schema: {errs[0]}")
        if machine.upstream_has(rec["provenance"]["tool"]["commit"]) is False:
            print(f"warning: {model_dir.name} / {size} was measured at a commit that isn't on a remote branch; "
                  "the review checks the adapter against the public repository", flush=True)
        todo.append((size, path, body))
    if not todo:
        return emit("OK", "Nothing new to send" + (f" ({', '.join(skipped)} already sent)" if skipped else ""), {"skipped": skipped})

    print(f"These go to {inbox()} as {login}. Once reviewed, they are public on modelsthatrun.dev with your GitHub login,")
    print(f"and dedicated to the public domain ({LICENSE}). No image, path, host name or IP address is sent.\n")
    for size, _, body in todo:
        print(preview(login, model_dir.name, size, body["record"]))
    if args.dry_run:
        return emit("OK", f"Dry run: {len(todo)} record(s) would be sent", {"sizes": [s for s, _, _ in todo]})
    if not args.yes:
        if not sys.stdin.isatty():
            raise MeasureError("USAGE", "Pass --yes to send without a prompt, once the person whose GitHub account this is has agreed")
        if input("\nSend them? [y/N] ").strip().lower() not in ("y", "yes"):
            return emit("OK", "Not sent")

    sent_ids = []
    for size, path, body in todo:
        status, r = http("POST", f"{inbox()}/v1/submissions", body, token=c["token"])
        if status == 401:
            raise MeasureError("AUTH", "The inbox no longer accepts your sign-in. Run `mtr login` again")
        if status not in (200, 201):
            detail = "; ".join(f"{i.get('path')}: {i.get('message')}" for i in r.get("issues", [])[:3]) or r.get("error", str(status))
            raise MeasureError("SUBMIT_REJECTED", f"{model_dir.name} / {size}: {detail}")
        path.with_name(f"{size}.submitted.json").write_text(json.dumps(
            {"id": r["id"], "inbox": inbox(), "login": login, "record_sha256": sha256_text(path),
             "submitted": datetime.now().astimezone().isoformat(timespec="seconds")}, indent=2) + "\n")
        sent_ids.append(r["id"])
        print(f"sent {model_dir.name} / {size}: {r['id']}, {r.get('state', 'pending')}")
    print("\nnext: `mtr submissions` shows the review's state")
    return emit("OK", f"Sent {len(sent_ids)} record(s) for review", {"ids": sent_ids})


def cmd_submissions(argv: list[str]) -> int:
    Parser(prog="mtr submissions", description="What you sent, and the state and note of each review.").parse_args(argv)
    c = credentials()
    if not c:
        raise MeasureError("AUTH", f"Not signed in to {inbox()}")
    status, r = http("GET", f"{inbox()}/v1/submissions", token=c["token"])
    if status == 401:
        raise MeasureError("AUTH", "The inbox no longer accepts your sign-in. Run `mtr login` again")
    if status != 200:
        raise MeasureError("NETWORK", f"The inbox answered {status}: {r.get('error', '')}")
    items = r.get("submissions", [])
    for s in items:
        note = f"\n    note: {s['note']}" if s.get("note") else ""
        print(f"{s['id']}  {s['model']} / {s['size']} on {s['gpu']}  {s['state']}  {s['created_at'][:10]}{note}")
    if not items:
        print("Nothing sent yet.")
    return emit("OK", f"{len(items)} submission(s)", {"submissions": items})


def main(cmd: str, argv: list[str]) -> int:
    commands = {"login": cmd_login, "logout": cmd_logout, "whoami": cmd_whoami, "results": cmd_results,
                "submit": cmd_submit, "submissions": cmd_submissions}
    return commands[cmd](argv)
