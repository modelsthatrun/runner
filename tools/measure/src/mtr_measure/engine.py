"""The pinned llama.cpp build that runs every language model, built once into a shared cache.

The official prebuilt Linux binaries of this release need glibc 2.38, which many machines don't have,
so the tool builds the pinned source with the local CUDA toolkit instead, for the local GPU's
architecture only. A build takes a few minutes (about 15 on the reference machine) and happens on the
first `mtr check` of any language model. Each architecture gets its own build directory.

Changing the build means measuring every language model again: raise PROTOCOL_VERSION with it.
Set MTR_ENGINES to move the cache (default ~/.cache/modelsthatrun/engines).
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

from .datasets import fetch
from .errors import MeasureError

LLAMA_CPP = {
    "name": "llama.cpp",
    "release": "v0.7.0",  # released 2026-10-15; its nightly-tag.txt names the build below
    "build": "b11669",
}
SOURCE_URL = "https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11669.tar.gz"
SOURCE_BYTES = 38_185_755
SOURCE_SHA256 = "e06ba4fa263e30fdb9f0e65f00a8965202ddaa0c51f548a67e1f9b2164d04efc"
# Only the server and the bench are built, for one GPU architecture (arch()).
CMAKE_FLAGS = [
    "-DGGML_CUDA=ON",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF",
    "-DLLAMA_CURL=OFF",
]
TARGETS = ["llama-server", "llama-bench"]


def arch(gpu: int = 0) -> str:
    """The GPU's CUDA architecture as CMake names it, e.g. "89" for compute capability 8.9."""
    from .gpu import Gpu

    g = Gpu(gpu)
    try:
        return g.compute_capability().replace(".", "")
    finally:
        g.close()


def root(gpu: int = 0) -> Path:
    base = Path(os.environ.get("MTR_ENGINES", "~/.cache/modelsthatrun/engines")).expanduser() / "llama.cpp"
    return base / f"{LLAMA_CPP['build']}-sm{arch(gpu)}"


def server_path(gpu: int = 0) -> Path:
    return root(gpu) / "build" / "bin" / "llama-server"


def nvcc() -> Path:
    for c in (os.environ.get("CUDACXX"), shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc"):
        if c and Path(c).is_file():
            return Path(c)
    raise MeasureError("ENGINE", "No CUDA compiler (nvcc) found to build llama.cpp. Install the CUDA toolkit, or set CUDACXX to its nvcc")


def nvcc_version(path: Path) -> str:
    out = subprocess.run([str(path), "--version"], capture_output=True, text=True).stdout
    found = re.search(r"release (\d+\.\d+)", out)
    return found.group(1) if found else "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure(say=print, gpu: int = 0) -> Path:
    """llama-server of the pinned build for this GPU's architecture, building it first if needed."""
    server = server_path(gpu)
    if server.is_file():
        return server
    base = root(gpu)
    base.mkdir(parents=True, exist_ok=True)
    archive = base / "source.tar.gz"
    if not archive.is_file():
        say(f"downloading llama.cpp {LLAMA_CPP['release']} ({LLAMA_CPP['build']}) source")
        fetch(SOURCE_URL, archive, SOURCE_BYTES, say)
    if sha256(archive) != SOURCE_SHA256:
        archive.unlink()
        raise MeasureError("ENGINE", "The llama.cpp source archive doesn't match its pinned sha256, so it was deleted. Run the command again; never pin another build yourself")
    src = base / "src"
    if src.exists():
        shutil.rmtree(src)
    with tarfile.open(archive) as t:
        top = t.getnames()[0].split("/")[0]
        t.extractall(base, filter="data")
    (base / top).rename(src)
    compiler = nvcc()
    cuda_arch = base.name.rsplit("-sm", 1)[1]
    say(f"building llama.cpp {LLAMA_CPP['build']} for sm_{cuda_arch} with CUDA {nvcc_version(compiler)} (a few minutes, once)")
    log = base / "build.log"
    build = base / "build"
    jobs = str(max(1, (os.cpu_count() or 2) - 2))
    with log.open("w") as f:
        for cmd in (["cmake", "-S", str(src), "-B", str(build), *CMAKE_FLAGS, f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch}",
                     f"-DCMAKE_CUDA_COMPILER={compiler}"],
                    ["cmake", "--build", str(build), "--target", *TARGETS, "-j", jobs]):
            f.write("$ " + " ".join(cmd) + "\n")
            f.flush()
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode != 0:
                raise MeasureError("ENGINE", f"Building llama.cpp failed. See {log}")
    if not server.is_file():
        raise MeasureError("ENGINE", f"The build finished without {server}. See {log}")
    return server


def describe(server: Path) -> dict:
    """What a record says about the engine. The binary's hash pins the exact build that ran."""
    return {
        "name": LLAMA_CPP["name"],
        "release": LLAMA_CPP["release"],
        "build": LLAMA_CPP["build"],
        "cuda": nvcc_version(nvcc()),
        "binary_sha256": sha256(server),
    }
