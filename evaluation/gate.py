from __future__ import annotations

import argparse
import json
from pathlib import Path


AXES = ("content_0_to_10", "style_0_to_10", "attribute_0_to_10")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether every official-like AAS sub-axis reaches the refinement threshold."
    )
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=9.0)
    args = parser.parse_args()

    payload = json.loads(args.result.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    missing = [axis for axis in AXES if axis not in scores]
    if missing:
        raise SystemExit(f"Missing score fields in {args.result}: {missing}")

    failed = {axis: float(scores[axis]) for axis in AXES if float(scores[axis]) < args.threshold}
    summary = {
        "sample_id": payload.get("sample_id"),
        "threshold": args.threshold,
        "scores": {axis: float(scores[axis]) for axis in AXES},
        "decision": "REFINE" if failed else "PASS",
        "failed_axes": failed,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    raise SystemExit(2 if failed else 0)


if __name__ == "__main__":
    main()
