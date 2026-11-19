"""Freeze executable research sources before launching a long experiment."""

import hashlib
import json
import os
from pathlib import Path


def frozen_environment(root: Path, name: str, runner: Path, *, extra_files=None):
    destination = root / "New_Model/experiment_sources" / name
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for folder, pattern in (("src/fedxmag_next", "*.py"), ("native", "*")):
        source = root / "New_Model" / folder
        target = destination / folder
        target.mkdir(parents=True)
        for path in sorted(source.glob(pattern)):
            if not path.is_file():
                continue
            content = path.read_bytes()
            (target / path.name).write_bytes(content)
            hashes[f"{folder}/{path.name}"] = hashlib.sha256(content).hexdigest()
    for relative, path in (extra_files or {}).items():
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Extra execution sources must stay inside the new snapshot")
        content = Path(path).read_bytes()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[relative.as_posix()] = hashlib.sha256(content).hexdigest()
    content = runner.read_bytes()
    (destination / runner.name).write_bytes(content)
    hashes[runner.name] = hashlib.sha256(content).hexdigest()
    (destination / "manifest.json").write_text(
        json.dumps(
            {
                "original_source_root": str(root / "New_Model/src"),
                "execution_source_root": str(destination / "src"),
                "sha256": hashes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {**os.environ, "PYTHONPATH": str(destination / "src"), "PYTHONIOENCODING": "utf-8"}
