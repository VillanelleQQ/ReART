import argparse
import json
from pathlib import Path
from typing import Dict, List, Any


VISUAL_FIELDS = [
    ("style", "Style"),
    ("main_subjects", "Main subjects"),
    ("scene_or_setting", "Scene or setting"),
    ("spatial_relations", "Spatial relations"),
    ("composition_or_layout", "Composition or layout"),
    ("material_or_surface", "Material or surface"),
    ("brushstroke_or_texture", "Brushstroke or texture"),
    ("line_quality", "Line quality"),
    ("color_or_tone", "Color or tone"),
    ("light_or_shadow", "Light or shadow"),
    ("mood_or_atmosphere", "Mood or atmosphere"),
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_id_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for record in records:
        key = record.get("id") or record.get("request_id") or record.get("sample_id")
        if key:
            out[key] = record
    return out


def build_topk_map(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {record["sample_id"]: record for record in records}


def pretty_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def format_visual_attributes(sample: Dict[str, Any]) -> str:
    lines = [
        "This section contains the hard constraints for generation and must be satisfied first. The reference images that follow may only help realize these requirements; they must not replace, weaken, or rewrite them.",
        f"- Original caption: {sample['original_caption']}",
    ]
    for field, label in VISUAL_FIELDS:
        values = sample.get(field) or []
        if values:
            joined = "; ".join(str(value) for value in values)
            lines.append(f"- {label}: {joined}")
    compressed = sample.get("compressed_caption")
    if compressed:
        lines.append(f"- Compressed caption: {compressed}")
    return "\n".join(lines)


def format_reference_block(
    title: str,
    topk_items: List[Dict[str, Any]],
    ref_map: Dict[str, Dict[str, Any]],
) -> str:
    lines = [title]
    for idx, item in enumerate(topk_items, start=1):
        ref_id = item["reference_id"]
        record = ref_map.get(ref_id)
        if record is None:
            record = {
                "id": ref_id,
                "original_caption": item.get("reference_caption", ""),
            }
        lines.append(f"\nReference {idx}")
        lines.append(f"- reference_id: {ref_id}")
        lines.append(f"- rank: {item.get('rank')}")
        lines.append(f"- final_score: {item.get('final_score')}")
        field_scores = item.get("field_scores") or {}
        if field_scores:
            lines.append(f"- field_scores: {json.dumps(field_scores, ensure_ascii=False)}")
        lines.append(pretty_json(record))
    return "\n".join(lines)


def build_prompt(
    sample: Dict[str, Any],
    ink_items: List[Dict[str, Any]],
    china_items: List[Dict[str, Any]],
    ink_ref_map: Dict[str, Dict[str, Any]],
    china_ref_map: Dict[str, Dict[str, Any]],
    primary_label: str,
    secondary_label: str,
) -> str:
    group_count = 1 + int(bool(china_items))
    reference_summary = (
        f"{group_count} group{'s' if group_count > 1 else ''} of reference images are provided below: "
        f"top{len(ink_items)} from {primary_label}"
    )
    if china_items:
        reference_summary += f", and top{len(china_items)} from {secondary_label}"
    reference_summary += "."
    sections = [
        "1. Task",
        "The highest-priority and non-negotiable goal of the final image is to satisfy the test sample itself, especially the requirements in Section 1 (Task) and Section 2 (Visual Attributes). All reference images that follow are only supporting materials. They may help you realize these requirements more accurately, but they must never override, replace, weaken, distort, or rewrite the requirements stated by the test sample.",
        sample["original_caption"],
        "",
        "2. Visual Attributes",
        format_visual_attributes(sample),
        "",
        "3. Structured Test Record",
        "During generation, always treat the structured description of this test sample as the highest-priority source of truth. Any content explicitly stated by the test sample should be reflected in the final image as faithfully as possible. If the reference images conflict with one another, or if any reference conflicts with the test sample, always follow the test sample.",
        pretty_json(sample),
        "",
        "4. Reference Usage",
        reference_summary + " Your task is not to mechanically average these references. First lock onto the hard requirements of the test sample, then read the structured description of each reference image and decide which aspects each image is most useful for, such as subject form, scene organization, composition and layout, support or material, brush texture, line treatment, color atmosphere, inscriptions or seals, or border and mounting cues. You may selectively absorb different aspects from different references, but every reference must serve the requirements of the test sample itself. Do not sacrifice any subject, layout, support, color, brushwork, or atmosphere explicitly required by the test sample just to resemble a reference image.",
        "",
        format_reference_block(f"A. Retrieved references from {primary_label}", ink_items, ink_ref_map),
        "",
    ]
    if china_items:
        sections.extend([
            format_reference_block(f"B. Retrieved references from {secondary_label}", china_items, china_ref_map),
            "",
        ])
    sections.extend([
        "5. Negative Constraints",
        "Do not weaken the original caption semantics. Do not omit any explicitly stated subject, layout, material, brushwork, color, tone, or atmosphere requirements from the test sample.",
        "Do not make the image overly clean, modern, uniform, or template-like in a generic digital-art sense.",
        "Avoid overly smooth contours, uniformly fogged backgrounds, overfilled subjects, composition drift, support-format drift, or atmosphere drift.",
        "Do not directly copy the full composition, exact object count, subject identity, or exact spatial arrangement of any single reference image.",
        "If any reference image conflicts with the test sample, discard the conflicting parts of the reference and prioritize the test sample.",
    ])
    return "\n".join(sections).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build generation prompts from test captions and retrieved references.")
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--primary-topk", "--ink-topk", dest="primary_topk", type=Path, required=True)
    parser.add_argument("--secondary-topk", "--china-topk", dest="secondary_topk", type=Path, default=None)
    parser.add_argument("--primary-ref", "--ink-ref", dest="primary_ref", type=Path, required=True)
    parser.add_argument("--secondary-ref", "--china-ref", dest="secondary_ref", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", type=str, default=None)
    parser.add_argument("--primary-k", "--ink-k", dest="primary_k", type=int, default=5)
    parser.add_argument("--secondary-k", "--china-k", dest="secondary_k", type=int, default=5)
    parser.add_argument("--primary-label", type=str, default="primary style pool")
    parser.add_argument("--secondary-label", type=str, default="secondary style pool")
    args = parser.parse_args()

    test_records = load_json(args.test)
    ink_topk_records = load_json(args.primary_topk)
    china_topk_records = load_json(args.secondary_topk) if args.secondary_topk else []
    ink_refs = load_json(args.primary_ref)
    china_refs = load_json(args.secondary_ref) if args.secondary_ref else []

    test_map = build_id_map(test_records)
    ink_topk_map = build_topk_map(ink_topk_records)
    china_topk_map = build_topk_map(china_topk_records)
    ink_ref_map = build_id_map(ink_refs)
    china_ref_map = build_id_map(china_refs)

    sample_ids = [args.sample_id] if args.sample_id else list(test_map.keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for sample_id in sample_ids:
        sample = test_map[sample_id]
        ink_items = (ink_topk_map.get(sample_id, {}).get("top10") or [])[: args.primary_k]
        china_items = (china_topk_map.get(sample_id, {}).get("top10") or [])[: args.secondary_k]
        prompt = build_prompt(
            sample,
            ink_items,
            china_items,
            ink_ref_map,
            china_ref_map,
            primary_label=args.primary_label,
            secondary_label=args.secondary_label,
        )

        sample_dir = args.output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        bundle = {
            "sample_id": sample_id,
            "test_sample": sample,
            "ink_topk": ink_items,
            "china_topk": china_items,
            "ink_reference_records": [ink_ref_map.get(item["reference_id"]) for item in ink_items],
            "china_reference_records": [china_ref_map.get(item["reference_id"]) for item in china_items],
            "prompt": prompt,
        }

        (sample_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (sample_dir / "prompt_bundle.json").write_text(pretty_json(bundle) + "\n", encoding="utf-8")

        print(f"wrote {sample_dir / 'prompt.txt'}")


if __name__ == "__main__":
    main()
