"""Freeze record for one converted task.

A task counts as converted only when a manifest exists whose recorded hashes
still match what is on disk.  Presence of the files alone is never enough: a
half-written or stale artifact would otherwise be reused silently.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
_CHUNK = 8 << 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Manifest:
    paper_id: str
    instance_index: int
    instance_sha256: str
    transform_sha256: str
    targets_sha256: str
    converter_model: str
    generation_rounds: int
    sanity_check_passed: bool
    assertions: dict[str, Any] = field(default_factory=dict)
    shape_coverage: dict[str, Any] = field(default_factory=dict)
    generated_at: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def read(task_dir: Path) -> Manifest | None:
    path = task_dir / MANIFEST_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return Manifest(**raw)
    except (json.JSONDecodeError, TypeError):
        return None


def is_fresh(task_dir: Path, instance_path: Path) -> bool:
    """True only if the frozen artifacts still describe this exact instance."""
    manifest = read(task_dir)
    if manifest is None or not manifest.sanity_check_passed:
        return False
    transform = task_dir / "transform.py"
    targets = task_dir / "input_targets.json"
    if not (transform.is_file() and targets.is_file() and instance_path.is_file()):
        return False
    return (
        sha256_file(instance_path) == manifest.instance_sha256
        and sha256_file(transform) == manifest.transform_sha256
        and sha256_file(targets) == manifest.targets_sha256
    )


def commit(task_dir: Path, staging: Path, manifest: Manifest) -> None:
    """Move a fully validated staging directory into place atomically.

    The manifest is written inside staging first, so a crash at any point leaves
    either the previous state or nothing -- never a directory that looks valid
    but is not.
    """
    (staging / MANIFEST_NAME).write_text(manifest.to_json(), encoding="utf-8")
    task_dir.parent.mkdir(parents=True, exist_ok=True)
    if task_dir.exists():
        doomed = Path(tempfile.mkdtemp(dir=task_dir.parent, prefix=".old-"))
        os.replace(task_dir, doomed / "prev")
        shutil.rmtree(doomed, ignore_errors=True)
    os.replace(staging, task_dir)
