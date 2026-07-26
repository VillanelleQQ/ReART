from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested_value(record: dict[str, Any], field: str) -> str:
    cur: Any = record
    for part in field.split("."):
        if not isinstance(cur, dict):
            return ""
        cur = cur.get(part)
    if cur is None:
        return ""
    return str(cur)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter JSON array records by keyword in a target field.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--field", required=True, help="Field path, e.g. caption or request_id or description.first_section.description")
    parser.add_argument("--include", required=True, help="Case-insensitive include keyword")
    parser.add_argument("--exclude", default=None, help="Optional case-insensitive exclude keyword")
    args = parser.parse_args()

    rows = load_json(args.input)
    if not isinstance(rows, list):
        raise SystemExit(f"Expected JSON array: {args.input}")

    include = args.include.lower()
    exclude = args.exclude.lower() if args.exclude else None

    kept: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = get_nested_value(row, args.field).lower()
        if include not in text:
            continue
        if exclude and exclude in text:
            continue
        kept.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"count": len(kept), "output": str(args.output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
