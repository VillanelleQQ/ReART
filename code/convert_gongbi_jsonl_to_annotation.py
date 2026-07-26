from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"Line {line_no} is not a JSON object.")
            records.append(obj)
    return records


def normalize_image_path(value: str) -> str:
    return value.replace("\\", "/")


def build_annotation(record: dict[str, Any]) -> dict[str, Any]:
    image_name = str(record.get("image_name", "")).strip()
    image_path = normalize_image_path(str(record.get("image_path", "")).strip())
    if not image_name:
        raise ValueError("Missing image_name in Gongbi metadata record.")

    visual_attributes = record.get("visual_attributes", {})
    if not isinstance(visual_attributes, dict):
        visual_attributes = {}

    style_terms: list[str] = []
    style_family = str(record.get("style_family", "")).strip()
    style = str(record.get("style", "")).strip()
    if style_family:
        style_terms.append(style_family)
    if style and style not in style_terms:
        style_terms.append(style)

    motif_tags = record.get("motif_tags", [])
    layout_tags = record.get("layout_tags", [])

    auxiliary_parts = []
    if style_terms:
        auxiliary_parts.append("Style: " + ", ".join(style_terms))
    primary_subtype = str(record.get("primary_subtype", "")).strip()
    if primary_subtype:
        auxiliary_parts.append("Primary subtype: " + primary_subtype)
    if motif_tags:
        auxiliary_parts.append("Motif tags: " + ", ".join(str(x) for x in motif_tags if str(x).strip()))
    if layout_tags:
        auxiliary_parts.append("Layout tags: " + ", ".join(str(x) for x in layout_tags if str(x).strip()))

    emotional_impact = ""
    if auxiliary_parts:
        emotional_impact = " | ".join(auxiliary_parts)

    return {
        "request_id": image_name,
        "description": {
            "first_section": {
                "description": str(record.get("content_description", "")).strip(),
            },
            "second_section": {
                "visual_attributes": {
                    "brushstroke": str(visual_attributes.get("brushstroke", "")).strip(),
                    "color": str(visual_attributes.get("color", "")).strip(),
                    "composition": str(visual_attributes.get("composition", "")).strip(),
                    "light_and_shadow": str(visual_attributes.get("light_and_shadow", "")).strip(),
                    "line_quality": str(visual_attributes.get("line_quality", "")).strip(),
                },
                "emotional_impact": emotional_impact,
            },
            "third_section": {
                "emotional_arousal_level": str(record.get("emotional_arousal_level", "")).strip(),
                "emotional_valence": str(record.get("emotional_valence", "")).strip(),
                "dominant_emotion": str(record.get("dominant_emotion", "")).strip(),
            },
        },
        "image_path": image_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.input)
    converted = [build_annotation(record) for record in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(converted)} records to {args.output}")


if __name__ == "__main__":
    main()
