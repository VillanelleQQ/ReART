from __future__ import annotations

from pathlib import Path
from typing import Any

from common import load_json


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

        # If the caller passes a compressed annotation file, also try the sibling
        # raw annotation file so image_path can be recovered automatically.
        raw_name = resolved.name
        if "_gpt54_compressed" in raw_name:
            candidate_name = raw_name.replace("_gpt54_compressed", "")
            candidate = resolved.with_name(candidate_name)
            if candidate.exists():
                candidate = candidate.resolve()
                if candidate not in seen:
                    seen.add(candidate)
                    expanded.append(candidate)
    return expanded


def build_annotation_map(*annotation_paths: Path) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for path in expand_annotation_sources(annotation_paths):
        if not path:
            continue
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


def build_retrieval_map(retrieval_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in retrieval_rows if isinstance(row, dict) and row.get("sample_id")}


def _pick_ids(topk_items: list[dict[str, Any]], wanted: list[str], fallback_k: int) -> list[str]:
    selected: list[str] = []
    wanted_set = set(wanted)
    for item in topk_items:
        ref_id = item.get("reference_id")
        if not isinstance(ref_id, str):
            continue
        item_text = " ".join(str(v).lower() for v in (item.get("reference_caption"), item.get("field_scores", {})))
        if any(token.lower() in item_text for token in wanted_set):
            selected.append(ref_id)
    if not selected:
        for item in topk_items[:fallback_k]:
            ref_id = item.get("reference_id")
            if isinstance(ref_id, str):
                selected.append(ref_id)
    deduped: list[str] = []
    for ref_id in selected:
        if ref_id not in deduped:
            deduped.append(ref_id)
    return deduped


def route_references(
    *,
    sample_id: str,
    retrieval_row: dict[str, Any],
    reference_annotations: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    topk_items = retrieval_row.get("top10") or retrieval_row.get("top5") or retrieval_row.get("top30") or []
    enriched = []
    for item in topk_items:
        ref_id = item.get("reference_id")
        if not isinstance(ref_id, str):
            continue
        record = reference_annotations.get(ref_id, {})
        enriched.append(
            {
                "reference_id": ref_id,
                "rank": item.get("rank"),
                "final_score": item.get("final_score"),
                "field_scores": item.get("field_scores", {}),
                "record": record,
            }
        )

    return {
        "sample_id": sample_id,
        "topk_enriched": enriched,
        "fallback_subject_refs": [row["reference_id"] for row in enriched[:4]],
        "fallback_layout_refs": [row["reference_id"] for row in enriched[:4]],
        "fallback_texture_refs": [row["reference_id"] for row in enriched[:3]],
        "fallback_tone_refs": [row["reference_id"] for row in enriched[:3]],
    }
