from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array: {path}")
    return [row for row in data if isinstance(row, dict)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple JSON arrays into one JSON array.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    merged: list[dict[str, Any]] = []
    for path in args.inputs:
        merged.extend(load_json_array(path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(merged), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
