"""Download pinned files from Hugging Face into the shared weights cache, checking their sha256.

Language-model adapters name each GGUF file by repository, commit and file name (adapter.py). The
file lands in ~/.cache/modelsthatrun/weights/<model id>/<repo>/<commit>/<file> and is checked
against the sha256 Hugging Face reports for it (X-Linked-ETag). A gated repository needs a token
in HF_TOKEN or ~/.cache/huggingface/token, and you accepting its terms on the website.
"""

from __future__ import annotations

import os
import re
import urllib.request
from pathlib import Path

from .adapter import weights_dir
from .datasets import fetch
from .engine import sha256
from .errors import MeasureError

COMMIT = re.compile(r"^[0-9a-f]{40}$")


def token() -> str | None:
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    f = Path("~/.cache/huggingface/token").expanduser()
    return f.read_text().strip() if f.is_file() else None


def local_path(model_id: str, repo: str, revision: str, file: str) -> Path:
    return weights_dir(model_id) / repo.replace("/", "--") / revision / file


def url(repo: str, revision: str, file: str) -> str:
    return f"https://huggingface.co/{repo}/resolve/{revision}/{file}"


def head(repo: str, revision: str, file: str) -> tuple[int, str]:
    """The file's size and sha256 as Hugging Face reports them, without following the redirect."""

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    req = urllib.request.Request(url(repo, revision, file), method="HEAD")
    if token():
        req.add_header("Authorization", f"Bearer {token()}")
    opener = urllib.request.build_opener(NoRedirect)
    try:
        resp = opener.open(req, timeout=30)
        headers = resp.headers
    except urllib.error.HTTPError as e:
        if e.code in (301, 302, 303, 307, 308):
            headers = e.headers
        elif e.code in (401, 403):
            raise MeasureError("WEIGHTS_MISSING", f"{repo} is gated or private. Accept its terms on huggingface.co with your account, then set HF_TOKEN") from None
        else:
            raise MeasureError("WEIGHTS_MISSING", f"Hugging Face answered {e.code} for {repo}@{revision[:12]}/{file}") from None
    size, etag = headers.get("X-Linked-Size"), (headers.get("X-Linked-ETag") or "").strip('"')
    if not size or not re.fullmatch(r"[0-9a-f]{64}", etag):
        raise MeasureError("WEIGHTS_MISSING", f"Hugging Face gave no size or sha256 for {repo}/{file}. Is it stored with LFS?")
    return int(size), etag


def ensure(model_id: str, repo: str, revision: str, file: str, say=print) -> Path:
    """The local copy of one pinned file, downloaded and checked first if needed."""
    dest = local_path(model_id, repo, revision, file)
    if dest.is_file():
        return dest
    size, digest = head(repo, revision, file)
    dest.parent.mkdir(parents=True, exist_ok=True)
    say(f"downloading {repo}/{file} at {revision[:12]} ({size / 2**30:.2f} GiB)")
    fetch(url(repo, revision, file), dest, size, say, {"Authorization": f"Bearer {token()}"} if token() else None)
    if sha256(dest) != digest:
        dest.unlink()
        raise MeasureError("WEIGHTS_MISSING", f"{file} doesn't match the sha256 Hugging Face reports. Deleted it; run check again")
    return dest
