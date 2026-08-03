from __future__ import annotations

from typing import Any

from utils.common import join_field_text


ANCHOR_GROUPS = {
    "subject_anchor": ["main_subjects", "spatial_relations"],
    "layout_anchor": ["scene_or_setting", "composition_or_layout", "material_or_surface"],
    "texture_anchor": ["brushstroke_or_texture", "line_quality"],
    "tone_anchor": ["color_or_tone", "mood_or_atmosphere"],
}


def _non_empty_list(record: dict[str, Any], field: str) -> list[str]:
    value = record.get(field, [])
    if not isinstance(value, list):
        return []
    return [str(x).strip() for x in value if str(x).strip()]


def _requirement_rows(record: dict[str, Any], field: str, priority: str, category: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    values = _non_empty_list(record, field)
    for index, value in enumerate(values, start=1):
        rows.append(
            {
                "requirement_id": f"{field}_{index:02d}",
                "field": field,
                "category": category,
                "priority": priority,
                "requirement": value,
                "check_question": f"Does the image clearly satisfy this requirement: {value}?",
                "source_layer": "structured_caption",
                "open_failure_allowed": True,
            }
        )
    return rows


def build_requirement_packet(structured_record: dict[str, Any]) -> dict[str, Any]:
    sample_id = structured_record.get("sample_id") or structured_record.get("id")
    original_caption = structured_record.get("original_caption") or structured_record.get("caption") or ""
    style = _non_empty_list(structured_record, "style")

    packet: dict[str, Any] = {
        "schema_version": "track1_refine_requirement_packet_v1",
        "sample_id": sample_id,
        "original_caption": original_caption,
        "style": style,
        "anchor_groups": {},
        "hard_requirements": [],
        "soft_requirements": [],
        "binary_checklist": {
            "content_requirements": [],
            "style_requirements": [],
            "attribute_requirements": [],
            "affect_requirements": [],
            "negative_requirements": [],
        },
    }

    for group_name, fields in ANCHOR_GROUPS.items():
        packet["anchor_groups"][group_name] = {
            field: _non_empty_list(structured_record, field)
            for field in fields
            if _non_empty_list(structured_record, field)
        }

    content_fields = ["main_subjects", "spatial_relations"]
    style_fields = ["style"]
    attribute_fields = [
        "scene_or_setting",
        "composition_or_layout",
        "material_or_surface",
        "brushstroke_or_texture",
        "line_quality",
        "color_or_tone",
        "light_or_shadow",
    ]
    affect_fields = ["mood_or_atmosphere"]

    for field in content_fields:
        rows = _requirement_rows(structured_record, field, "P0", "content")
        packet["binary_checklist"]["content_requirements"].extend(rows)
        packet["hard_requirements"].extend(rows)

    for value_index, value in enumerate(style, start=1):
        row = {
            "requirement_id": f"style_{value_index:02d}",
            "field": "style",
            "category": "style",
            "priority": "P0",
            "requirement": value,
            "check_question": f"Does the image clearly read as this style family or named style: {value}?",
            "source_layer": "structured_caption",
            "open_failure_allowed": True,
        }
        packet["binary_checklist"]["style_requirements"].append(row)
        packet["hard_requirements"].append(row)

    for field in attribute_fields:
        priority = "P0" if field in {"composition_or_layout", "material_or_surface"} else "P1"
        rows = _requirement_rows(structured_record, field, priority, "attribute")
        packet["binary_checklist"]["attribute_requirements"].extend(rows)
        if priority == "P0":
            packet["hard_requirements"].extend(rows)
        else:
            packet["soft_requirements"].extend(rows)

    for field in affect_fields:
        rows = _requirement_rows(structured_record, field, "P2", "affect")
        packet["binary_checklist"]["affect_requirements"].extend(rows)
        packet["soft_requirements"].extend(rows)

    packet["binary_checklist"]["negative_requirements"] = [
        {
            "requirement_id": "negative_01",
            "field": "negative",
            "category": "negative",
            "priority": "P0",
            "requirement": "Do not add new major objects or scene relations that are not explicitly supported by the test caption.",
            "check_question": "Has the image avoided inventing unsupported major content?",
            "source_layer": "edit_guard",
            "open_failure_allowed": True,
        },
        {
            "requirement_id": "negative_02",
            "field": "negative",
            "category": "negative",
            "priority": "P0",
            "requirement": "Do not change a correct page format, carrier type, or composition skeleton while repairing local errors.",
            "check_question": "Has the repair preserved any already-correct carrier and composition skeleton?",
            "source_layer": "edit_guard",
            "open_failure_allowed": True,
        },
        {
            "requirement_id": "negative_03",
            "field": "negative",
            "category": "negative",
            "priority": "P0",
            "requirement": "Do not introduce glossy modern digital texture or over-polished rendering drift.",
            "check_question": "Has the image avoided modern glossy over-polish?",
            "source_layer": "edit_guard",
            "open_failure_allowed": True,
        },
    ]

    packet["summary_for_model"] = {
        "style": style,
        "subject_anchor_text": join_field_text(packet["anchor_groups"].get("subject_anchor", {})),
        "layout_anchor_text": join_field_text(packet["anchor_groups"].get("layout_anchor", {})),
        "texture_anchor_text": join_field_text(packet["anchor_groups"].get("texture_anchor", {})),
        "tone_anchor_text": join_field_text(packet["anchor_groups"].get("tone_anchor", {})),
    }
    return packet
