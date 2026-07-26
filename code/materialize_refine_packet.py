from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import dump_json, ensure_dir, load_json


def expand_annotation_sources(annotation_paths: tuple[Path, ...]) -> list[Path]:
    expanded: list[Path] = []
    seen: set[Path] = set()
    for path in annotation_paths:
        if not path:
            continue
        resolved = path.resolve()
        if resolved not in seen and resolved.exists():
            seen.add(resolved)
            expanded.append(resolved)

        if "_gpt54_compressed" in resolved.name:
            candidate = resolved.with_name(resolved.name.replace("_gpt54_compressed", ""))
            if candidate.exists():
                candidate = candidate.resolve()
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
    return expanded


def build_annotation_map(*annotation_paths: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for path in expand_annotation_sources(annotation_paths):
        records = load_json(path)
        for record in records:
            rid = record.get("request_id") or record.get("id")
            if not isinstance(rid, str):
                continue
            if rid not in mapping:
                mapping[rid] = dict(record)
                continue
            merged = dict(mapping[rid])
            merged.update({k: v for k, v in record.items() if v not in (None, "", [])})
            mapping[rid] = merged
    return mapping


def build_refine_prompt(
    *,
    structured_record: dict[str, Any],
    requirement_packet: dict[str, Any],
    refine_plan: dict[str, Any],
    references_by_role: dict[str, list[dict[str, Any]]],
) -> str:
    sections = [
        "1. Repair Goal",
        "You must perform a controlled repair without damaging the parts of the current image that are already correct. Always treat the test-sample requirement as the highest-priority source of truth.",
        structured_record.get("original_caption") or structured_record.get("caption") or "",
        "",
        "2. Must Preserve",
        json.dumps(refine_plan.get("keep", []), ensure_ascii=False, indent=2),
        "",
        "3. Must Fix",
        json.dumps(refine_plan.get("fix", []), ensure_ascii=False, indent=2),
        "",
        "4. Do Not Change",
        json.dumps(refine_plan.get("avoid", []), ensure_ascii=False, indent=2),
        "",
        "5. Repair Priority",
        json.dumps(refine_plan.get("priority", []), ensure_ascii=False, indent=2),
        "",
        "6. Repair Scope",
        json.dumps(
            {
                "repair_scope": refine_plan.get("repair_scope"),
                "repair_region": refine_plan.get("repair_region", []),
                "lock_region": refine_plan.get("lock_region", []),
            },
            ensure_ascii=False,
            indent=2,
        ),
        "",
        "7. Structured Requirement",
        json.dumps(requirement_packet, ensure_ascii=False, indent=2),
        "",
        "8. References Assigned by Purpose",
    ]
    for role, rows in references_by_role.items():
        sections.append(f"{role}:")
        sections.append(json.dumps(rows, ensure_ascii=False, indent=2))
        sections.append("")
    sections.extend(
        [
            "9. Execution Constraints",
            "- Repair subjects and relations first, then layout and support, then brush/texture quality, and finally tone and atmosphere.",
            "- Do not copy the full composition of any single reference image.",
            "- If the issue can be fixed locally, do not repaint the whole image.",
            "- Reference images are only for correcting the corresponding module; they must not override the test-sample requirements.",
        ]
    )
    return "\n".join(sections).strip() + "\n"


def build_retrieval_lookup(retrieval_packet: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    if not retrieval_packet:
        return lookup
    for row in retrieval_packet.get("topk_enriched", []):
        ref_id = row.get("reference_id")
        record = row.get("record")
        if isinstance(ref_id, str) and isinstance(record, dict):
            lookup[ref_id] = record
    return lookup


def merge_reference_record(
    *,
    ref_id: str,
    annotation_map: dict[str, dict[str, Any]],
    retrieval_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    annotation_record = annotation_map.get(ref_id)
    retrieval_record = retrieval_lookup.get(ref_id)
    if annotation_record is None and retrieval_record is None:
        return None
    merged: dict[str, Any] = {}
    if isinstance(annotation_record, dict):
        merged.update(annotation_record)
    if isinstance(retrieval_record, dict):
        merged.update(retrieval_record)
    return merged


def select_reference_records(
    refine_plan: dict[str, Any],
    annotation_map: dict[str, dict[str, Any]],
    retrieval_packet: dict[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    retrieval_lookup = build_retrieval_lookup(retrieval_packet)
    mapping = {
        "subject": refine_plan.get("ref_for_subject", []),
        "layout": refine_plan.get("ref_for_layout", []),
        "texture": refine_plan.get("ref_for_texture", []),
        "tone": refine_plan.get("ref_for_tone", []),
    }
    for role, ref_ids in mapping.items():
        rows = []
        for ref_id in ref_ids:
            merged = merge_reference_record(
                ref_id=ref_id,
                annotation_map=annotation_map,
                retrieval_lookup=retrieval_lookup,
            )
            if merged is not None:
                rows.append(merged)
        output[role] = rows
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize final refine prompt and reference packet.")
    parser.add_argument("--structured-record-json", type=Path, required=True)
    parser.add_argument("--requirement-packet-json", type=Path, required=True)
    parser.add_argument("--refine-plan-json", type=Path, required=True)
    parser.add_argument("--primary-annotation-json", type=Path, required=True)
    parser.add_argument("--secondary-annotation-json", type=Path, required=True)
    parser.add_argument("--retrieval-packet-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    structured_record = load_json(args.structured_record_json)
    requirement_packet = load_json(args.requirement_packet_json)
    refine_plan = load_json(args.refine_plan_json)
    annotation_map = build_annotation_map(args.primary_annotation_json, args.secondary_annotation_json)
    retrieval_packet = load_json(args.retrieval_packet_json) if args.retrieval_packet_json else None
    refs = select_reference_records(refine_plan, annotation_map, retrieval_packet)
    prompt = build_refine_prompt(
        structured_record=structured_record,
        requirement_packet=requirement_packet,
        refine_plan=refine_plan,
        references_by_role=refs,
    )
    ensure_dir(args.output_dir)
    (args.output_dir / "prompt_zh_refine.txt").write_text(prompt, encoding="utf-8")
    dump_json(
        args.output_dir / "prompt_bundle_refine.json",
        {
            "structured_record": structured_record,
            "requirement_packet": requirement_packet,
            "refine_plan": refine_plan,
            "references_by_role": refs,
        },
    )
    dump_json(
        args.output_dir / "reference_ids_by_role.json",
        {key: [row.get("request_id") or row.get("id") for row in value] for key, value in refs.items()},
    )


if __name__ == "__main__":
    main()
