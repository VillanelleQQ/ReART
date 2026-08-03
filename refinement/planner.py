from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.common import call_openai_multimodal_json


SYSTEM_PROMPT = """You are a careful Track 1 artistic image refinement planner.

Your job is not to rewrite the whole image. Your job is to:
1. compare the current image against the structured requirements,
2. identify what is already correct and must be preserved,
3. identify what is wrong and must be repaired,
4. choose which reference images are useful for which repair target,
5. return one strict JSON object only.

Rules:
- Never invent caption requirements that are not explicitly supported.
- Protect already-correct subject identity, scene relations, composition skeleton, carrier type, and emotional direction.
- Prefer local or regional repair when possible.
- Use reference images selectively by function, not as a single averaged style pool.
- Hard requirements take precedence over soft atmospheric refinement.
- If the image content and layout are already correct but the rendering looks too polished, too realistic, too glossy, too modern, too volumetric, or not period-authentic enough, treat this primarily as texture/style-surface drift, not as a mere tone problem.
- Use `texture_error` when the main issue is surface handling, brush/line treatment, material feel, finish, realism level, or period-authentic paint logic.
- Use `tone_error` only when the main issue is mostly palette, warmth/coolness, value balance, or atmosphere, while surface handling is already appropriate.
- Use `mixed_error` if both tone and texture/style-surface are materially wrong.
- Only use `fullframe` when the whole image truly requires a global rebuild. If composition and entities are already correct and only finish, surface, panel readability, or selected tonal zones need repair, prefer `zone`.
- If a repair depends on period-authentic surface handling or historical finish, populate `ref_for_texture` with suitable references rather than leaving it empty.

Return strict JSON with this schema:
{
  "sample_id": "",
  "already_right": [],
  "error_type": "subject_error | layout_error | texture_error | tone_error | mixed_error",
  "repair_scope": "spot | zone | fullframe",
  "repair_region": [],
  "lock_region": [],
  "keep": [],
  "fix": [],
  "avoid": [],
  "priority": [],
  "ref_for_subject": [],
  "ref_for_layout": [],
  "ref_for_texture": [],
  "ref_for_tone": [],
  "plan_summary": ""
}
"""


def build_user_text(
    *,
    structured_record: dict[str, Any],
    requirement_packet: dict[str, Any],
    retrieval_packet: dict[str, Any],
) -> str:
    topk_preview = []
    for row in retrieval_packet["topk_enriched"][:10]:
        topk_preview.append(
            {
                "reference_id": row["reference_id"],
                "rank": row.get("rank"),
                "final_score": row.get("final_score"),
                "field_scores": row.get("field_scores", {}),
                "record": row.get("record", {}),
            }
        )
    payload = {
        "sample_id": structured_record.get("sample_id") or structured_record.get("id"),
        "structured_record": structured_record,
        "requirement_packet": requirement_packet,
        "retrieved_references": topk_preview,
    }
    return (
        "Below is the Track 1 refine input packet. "
        "Use the current image as primary evidence, and use the reference records only to decide how to repair specific errors. "
        "Be especially careful not to collapse style-surface drift into a generic tone issue. "
        "If the image is structurally correct but feels too polished, too modern, too realistic, or insufficiently period-authentic, diagnose that as texture/style-surface drift and select texture references. "
        "Output strict JSON only.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def validate_plan(plan: dict[str, Any], sample_id: str) -> None:
    required = [
        "sample_id",
        "already_right",
        "error_type",
        "repair_scope",
        "repair_region",
        "lock_region",
        "keep",
        "fix",
        "avoid",
        "priority",
        "ref_for_subject",
        "ref_for_layout",
        "ref_for_texture",
        "ref_for_tone",
        "plan_summary",
    ]
    missing = [key for key in required if key not in plan]
    if missing:
        raise ValueError(f"Missing keys in refine plan: {missing}")
    if str(plan["sample_id"]) != str(sample_id):
        raise ValueError(f"Plan sample_id mismatch: expected {sample_id}, got {plan['sample_id']}")
    list_fields = [
        "already_right",
        "repair_region",
        "lock_region",
        "keep",
        "fix",
        "avoid",
        "priority",
        "ref_for_subject",
        "ref_for_layout",
        "ref_for_texture",
        "ref_for_tone",
    ]
    for field in list_fields:
        if not isinstance(plan[field], list):
            raise ValueError(f"{field} must be a list")


def build_refine_plan(
    *,
    sample_id: str,
    current_image_path: Path,
    structured_record: dict[str, Any],
    requirement_packet: dict[str, Any],
    retrieval_packet: dict[str, Any],
    base_url: str,
    api_key_env: str,
    model: str,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict[str, Any]:
    image_paths = [current_image_path]
    for row in retrieval_packet["topk_enriched"][:6]:
        record = row.get("record", {})
        image_path = record.get("_resolved_image_path")
        if image_path:
            image_paths.append(Path(str(image_path)))

    plan = call_openai_multimodal_json(
        base_url=base_url,
        api_key_env=api_key_env,
        model=model,
        system_prompt=SYSTEM_PROMPT,
        user_text=build_user_text(
            structured_record=structured_record,
            requirement_packet=requirement_packet,
            retrieval_packet=retrieval_packet,
        ),
        image_paths=image_paths,
        temperature=0.1,
        max_tokens=12000,
        timeout=timeout,
        retries=retries,
        retry_sleep=retry_sleep,
    )
    validate_plan(plan, sample_id)
    return plan
