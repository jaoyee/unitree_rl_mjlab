#!/usr/bin/env python3
"""Atomically concatenate JSONL files while validating every row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".new")
    row_count = 0
    with temporary.open("w", encoding="utf-8") as target:
        for value in args.input:
            source = Path(value).expanduser().resolve()
            with source.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at {source}:{line_number}: {exc}") from exc
                    target.write(line if line.endswith("\n") else line + "\n")
                    row_count += 1
    temporary.replace(output)
    print(f"merged_jsonl_rows={row_count} output={output}")


if __name__ == "__main__":
    main()
