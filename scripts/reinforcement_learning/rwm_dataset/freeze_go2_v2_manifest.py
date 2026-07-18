"""Create a SHA-256 freeze manifest for an explicit Go2 V2 artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--root", required=True)
    parser.add_argument("--domain", choices=("real", "sim"), required=True)
    parser.add_argument("--path", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    files: list[Path] = []
    for raw_path in args.path:
        candidate = (root / raw_path).resolve()
        if candidate.is_file():
            files.append(candidate)
        elif candidate.is_dir():
            files.extend(path for path in candidate.rglob("*") if path.is_file())
        else:
            raise FileNotFoundError(candidate)

    records = []
    for path in sorted(set(files)):
        stat = path.stat()
        records.append({
            "path": str(path.relative_to(root)),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256(path),
        })
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    payload = {
        "schema": "go2_dataset_freeze_manifest_v2",
        "domain": args.domain,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "root": str(root),
        "git_commit": commit,
        "file_count": len(records),
        "total_bytes": sum(record["size"] for record in records),
        "files": records,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({
        "output": str(output.resolve()),
        "domain": args.domain,
        "file_count": len(records),
        "total_bytes": payload["total_bytes"],
        "git_commit": commit,
    }, indent=2))


if __name__ == "__main__":
    main()
