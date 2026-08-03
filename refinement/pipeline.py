from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from refinement.planner import build_refine_plan
from refinement.requirements import build_requirement_packet
from refinement.router import build_annotation_map, build_retrieval_map, route_references
from utils.common import build_id_map, dump_json, ensure_dir, load_json


def normalize_relative_image_path(raw_path: str) -> str:
    text = raw_path.replace("\\", "/")
    if text.startswith("Images/"):
        text = text[len("Images/") :]
    return text


def resolve_reference_image_paths(
    *,
    retrieval_packet: dict[str, Any],
    image_root: Path,
    fallback_image_root: Path | None,
) -> None:
    for row in retrieval_packet["topk_enriched"]:
        record = row.get("record", {})
        raw_image_path = record.get("image_path")
        if not raw_image_path:
            continue
        normalized = normalize_relative_image_path(str(raw_image_path))
        path = image_root / normalized
        if not path.exists() and fallback_image_root is not None:
            alt = fallback_image_root / normalized
            if alt.exists():
                path = alt
        if path.exists():
            record["_resolved_image_path"] = str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the refine decomposition + planning pipeline for one sample.")
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--structured-test-json", type=Path, required=True)
    parser.add_argument("--retrieval-json", type=Path, required=True)
    parser.add_argument("--primary-annotation-json", type=Path, default=None)
    parser.add_argument("--secondary-annotation-json", type=Path, default=None)
    parser.add_argument(
        "--annotation-json",
        type=Path,
        action="append",
        default=[],
        help="Annotation JSON containing structured reference records and/or raw image_path fields; repeat as needed.",
    )
    parser.add_argument("--current-image", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--fallback-image-root", type=Path, default=None)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=Path("refine/outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    structured_rows = load_json(args.structured_test_json)
    retrieval_rows = load_json(args.retrieval_json)
    annotation_paths = list(args.annotation_json)
    annotation_paths.extend(
        path for path in (args.primary_annotation_json, args.secondary_annotation_json) if path is not None
    )
    if not annotation_paths:
        raise SystemExit("Pass at least one --annotation-json.")
    annotation_map = build_annotation_map(*annotation_paths)
    structured_map = build_id_map(structured_rows)
    retrieval_map = build_retrieval_map(retrieval_rows)

    if args.sample_id not in structured_map:
        raise SystemExit(f"Missing sample_id in structured test json: {args.sample_id}")
    if args.sample_id not in retrieval_map:
        raise SystemExit(f"Missing sample_id in retrieval json: {args.sample_id}")

    structured_record = structured_map[args.sample_id]
    requirement_packet = build_requirement_packet(structured_record)
    retrieval_packet = route_references(
        sample_id=args.sample_id,
        retrieval_row=retrieval_map[args.sample_id],
        reference_annotations=annotation_map,
    )
    resolve_reference_image_paths(
        retrieval_packet=retrieval_packet,
        image_root=args.image_root,
        fallback_image_root=args.fallback_image_root,
    )
    refine_plan = build_refine_plan(
        sample_id=args.sample_id,
        current_image_path=args.current_image,
        structured_record=structured_record,
        requirement_packet=requirement_packet,
        retrieval_packet=retrieval_packet,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model=args.model,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )

    sample_dir = args.output_root / args.sample_id
    ensure_dir(sample_dir)
    dump_json(sample_dir / "structured_record.json", structured_record)
    dump_json(sample_dir / "requirement_packet.json", requirement_packet)
    dump_json(sample_dir / "retrieval_packet.json", retrieval_packet)
    dump_json(sample_dir / "refine_plan.json", refine_plan)

    from refinement.materialize import build_refine_prompt, select_reference_records

    refs_by_role = select_reference_records(refine_plan, annotation_map, retrieval_packet)
    prompt = build_refine_prompt(
        structured_record=structured_record,
        requirement_packet=requirement_packet,
        refine_plan=refine_plan,
        references_by_role=refs_by_role,
    )
    (sample_dir / "prompt_zh_refine.txt").write_text(prompt, encoding="utf-8")
    dump_json(
        sample_dir / "prompt_bundle_refine.json",
        {
            "structured_record": structured_record,
            "requirement_packet": requirement_packet,
            "retrieval_packet": retrieval_packet,
            "refine_plan": refine_plan,
            "references_by_role": refs_by_role,
            "current_image_path": str(args.current_image.resolve()),
        },
    )
    dump_json(
        sample_dir / "reference_ids_by_role.json",
        {key: [row.get("request_id") or row.get("id") for row in value] for key, value in refs_by_role.items()},
    )
    resolved_reference_paths: list[str] = []
    seen = set()
    for rows in refs_by_role.values():
        for row in rows:
            path = row.get("_resolved_image_path")
            if isinstance(path, str) and path not in seen:
                seen.add(path)
                resolved_reference_paths.append(path)
    dump_json(sample_dir / "reference_paths_used.json", resolved_reference_paths)
    print(f"Wrote refine packet to: {sample_dir}")


if __name__ == "__main__":
    main()
