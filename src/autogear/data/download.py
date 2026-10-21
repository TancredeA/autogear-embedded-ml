"""Download the pinned public sources and check their checksums."""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from .sources import ALL_SOURCES, Source


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(src: Source, dest_dir: Path, force: bool = False) -> Path:
    dest = dest_dir / src.filename
    if dest.exists() and not force and sha256(dest) == src.sha256:
        return dest
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(src.url, tmp)
    digest = sha256(tmp)
    if digest != src.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {src.key}: expected {src.sha256}, got {digest}")
    tmp.replace(dest)
    return dest


def fetch_all(dest_dir: Path, force: bool = False) -> dict[str, Path]:
    return {s.key: fetch(s, dest_dir, force) for s in ALL_SOURCES}
