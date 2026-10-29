"""Benchmark data for `mtr eval`, downloaded once into a shared cache.

The images are never published or uploaded. Only the scores computed on them are. Each
dataset's terms are read from its primary source before its first use; they are listed with it.

Set MTR_DATASETS to move the cache (default ~/.cache/modelsthatrun/datasets).
"""

from __future__ import annotations

import os
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .errors import MeasureError


@dataclass(frozen=True)
class Download:
    url: str
    bytes: int  # the expected size, checked after download


@dataclass(frozen=True)
class Dataset:
    id: str
    name: str
    terms: str  # what the primary source says, and where; checked on the date given
    downloads: tuple[Download, ...]
    # Files and directories that must exist after extraction, relative to the dataset directory.
    needs: tuple[str, ...]
    # Archive members to keep, by prefix. Others are not extracted, to save disk.
    keep: tuple[str, ...]


COCO_VAL2017 = Dataset(
    id="coco-val2017",
    name="COCO val2017",
    terms="Annotations CC BY 4.0; images under Flickr terms. https://cocodataset.org/#termsofuse, checked 2026-10-26",
    downloads=(
        Download("http://images.cocodataset.org/zips/val2017.zip", 815_585_330),
        Download("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", 252_907_541),
    ),
    needs=("val2017", "annotations/instances_val2017.json"),
    keep=("val2017/", "annotations/instances_val2017.json", "annotations/person_keypoints_val2017.json"),
)

DATASETS = {d.id: d for d in (COCO_VAL2017,)}


def root() -> Path:
    return Path(os.environ.get("MTR_DATASETS", "~/.cache/modelsthatrun/datasets")).expanduser()


def ready(d: Dataset) -> bool:
    base = root() / d.id
    return all((base / n).exists() for n in d.needs)


def fetch(url: str, dest: Path, expected: int, say=print, headers: dict[str, str] | None = None) -> None:
    """Download with resume into dest. Fails unless the file has exactly the expected size."""
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={**(headers or {}), **({"Range": f"bytes={have}-"} if have else {})})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if have and resp.status != 206:  # the server ignored the range: start over
                have = 0
            with part.open("ab" if have else "wb") as f:
                done, last = have, 0
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    done += len(chunk)
                    if done - last >= 100 << 20:
                        say(f"  {dest.name}: {done / 2**20:.0f} of {expected / 2**20:.0f} MiB")
                        last = done
    except OSError as e:
        raise MeasureError("DATASET", f"Downloading {url} failed: {e}. Run the command again to resume") from None
    size = part.stat().st_size
    if size != expected:
        part.unlink()
        raise MeasureError("DATASET", f"{dest.name} is {size} bytes, expected {expected}. Deleted it; run the command again")
    part.replace(dest)


def ensure(d: Dataset, say=print) -> Path:
    """The dataset directory, downloading and extracting it first if needed."""
    base = root() / d.id
    if ready(d):
        return base
    base.mkdir(parents=True, exist_ok=True)
    need = sum(x.bytes for x in d.downloads) * 2
    if shutil.disk_usage(base).free < need:
        raise MeasureError("DATASET", f"{d.name} needs about {need / 2**30:.1f} GiB free to download and extract")
    say(f"downloading {d.name} into {base} (once; the images are never published). Terms: {d.terms}")
    for dl in d.downloads:
        archive = base / dl.url.rsplit("/", 1)[-1]
        if not archive.exists():
            fetch(dl.url, archive, dl.bytes, say)
        say(f"  extracting {archive.name}")
        try:
            with zipfile.ZipFile(archive) as z:
                for m in z.infolist():
                    if any(m.filename.startswith(k) for k in d.keep):
                        z.extract(m, base)
        except zipfile.BadZipFile:
            archive.unlink()
            raise MeasureError("DATASET", f"{archive.name} was damaged and is deleted. Run the command again") from None
        archive.unlink()
    if not ready(d):
        raise MeasureError("DATASET", f"{d.name} is incomplete after extraction: expected {', '.join(d.needs)} in {base}")
    return base


def main() -> int:
    """python -m mtr_measure.datasets <id>: download a dataset ahead of the first eval."""
    for name in sys.argv[1:] or list(DATASETS):
        print(ensure(DATASETS[name]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
