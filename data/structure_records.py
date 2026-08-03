from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any
from urllib import error, request


SYSTEM_PROMPT = """You are a careful art-caption compression assistant.

You will receive raw caption records or raw EmoArt annotation records for artistic images.
Your job is to compress each record into a concise, retrieval-friendly structured record.

Top priority:
1. Do not lose explicit information from the caption.
2. Do not hallucinate or infer unstated details.
3. Prefer short phrases or clear words, not long rewritten prose.
4. Preserve fine-grained details when they are explicitly present.
5. Keep the original record order unchanged.

Rules:
- Output valid JSON only.
- Return one JSON array.
- One output object per input object.
- Keep `id` exactly equal to the input identifier.
- If the input is a caption record, keep `original_caption` exactly equal to the input caption.
- If the input is an annotation record, set `original_caption` to the first-section description text exactly.
- Use short phrases, keywords, or concise fragments.
- If a field is not explicitly supported by the caption, return an empty list.
- Do not convert specific phrases into overly broad abstractions if detail would be lost.
  Example: prefer `orchid grasses` over just `orchid`; prefer `red seals` over just `seal`.
- Preserve relation information as short relation phrases, not isolated prepositions.
  Example: use `travelers near stone embankment`, not just `near`.
- For annotation records, you may use:
  - `request_id`
  - `description.first_section.description`
  - `description.second_section.visual_attributes`
  - `description.second_section.emotional_impact`
  - `description.third_section`
  - `image_path`
  But do not invent information beyond these fields.

Target schema:
{
  "id": "",
  "original_caption": "",
  "style": [],
  "main_subjects": [],
  "scene_or_setting": [],
  "spatial_relations": [],
  "brushstroke_or_texture": [],
  "color_or_tone": [],
  "composition_or_layout": [],
  "light_or_shadow": [],
  "line_quality": [],
  "mood_or_atmosphere": [],
  "material_or_surface": [],
  "compressed_caption": ""
}

Field guidance:
- `style`: named style or genre terms such as `Ink and wash painting`.
- `main_subjects`: concrete objects, figures, plants, structures, landscape elements.
- `scene_or_setting`: scene type, format, or setting phrases such as `riverside landscape`, `album page`, `fan-shaped composition`.
- `spatial_relations`: short relation phrases such as `boats on rippling water`, `travelers near stone embankment`, `calligraphy beside bamboo`.
- `brushstroke_or_texture`: explicit brushwork, washes, texture, stroke quality.
- `color_or_tone`: explicit palette, tone, chroma, monochrome, sepia, muted, beige, brown-gray, etc.
- `composition_or_layout`: explicit layout/composition terms, including foreground/background, vertical layout, layered cliffs, fan-shaped layout.
- `light_or_shadow`: explicit lighting or contrast only.
- `line_quality`: explicit line-related descriptors only.
- `mood_or_atmosphere`: explicit mood words or short atmosphere phrases only.
- `material_or_surface`: paper, scroll, fan, album page, mounting sheet, etc.
- `compressed_caption`: one compact retrieval-oriented line that preserves the main information without unnecessary prose.
"""


def build_user_prompt(batch: list[dict[str, Any]]) -> str:
    return (
        "Compress the following raw caption records into the target schema.\n\n"
        "Important reminders:\n"
        "- Do not omit explicit details.\n"
        "- Prefer concise phrases or clear words.\n"
        "- Do not infer unstated content.\n"
        "- Preserve relation phrases.\n"
        "- Keep each output aligned to the input record.\n"
        "- For annotation records, treat first_section.description as the main original caption text, and use the visual_attributes/emotional_impact/emotion labels as auxiliary evidence.\n\n"
        "Records:\n"
        f"{json.dumps(batch, ensure_ascii=False, indent=2)}"
    )


def parse_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\[.*\])\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON array.")
    payload = cleaned[start : end + 1]
    parsed = json.loads(payload)
    if not isinstance(parsed, list):
        raise ValueError("Model output JSON is not a list.")
    return parsed


def chunked(records: list[dict[str, Any]], batch_size: int) -> list[list[dict[str, Any]]]:
    return [records[i : i + batch_size] for i in range(0, len(records), batch_size)]


def validate_record_shape(record: dict[str, Any]) -> None:
    required = [
        "id",
        "original_caption",
        "style",
        "main_subjects",
        "scene_or_setting",
        "spatial_relations",
        "brushstroke_or_texture",
        "color_or_tone",
        "composition_or_layout",
        "light_or_shadow",
        "line_quality",
        "mood_or_atmosphere",
        "material_or_surface",
        "compressed_caption",
    ]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Compressed record missing required keys: {missing}")


def call_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    batch: list[dict[str, Any]],
    temperature: float,
    max_retries: int,
    retry_sleep: float,
) -> list[dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(batch)},
                ],
            }
            body = json.dumps(payload).encode("utf-8")
            endpoint = base_url.rstrip("/") + "/chat/completions"
            req = request.Request(
                endpoint,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                method="POST",
            )
            with request.urlopen(req, timeout=300) as resp:
                response_json = json.loads(resp.read().decode("utf-8"))
            content = response_json["choices"][0]["message"]["content"] or ""
            parsed = parse_json_array(content)
            if len(parsed) != len(batch):
                raise ValueError(
                    f"Compressed batch size mismatch: expected {len(batch)}, got {len(parsed)}"
                )
            for item in parsed:
                if not isinstance(item, dict):
                    raise ValueError("Compressed batch contains a non-object item.")
                validate_record_shape(item)
            return parsed
        except (ValueError, KeyError, json.JSONDecodeError, error.URLError) as exc:
            last_error = exc
            if attempt == max_retries:
                break
            time.sleep(retry_sleep)
    assert last_error is not None
    raise last_error


def validate_batch_alignment(batch_index: int, batch: list[dict[str, Any]], compressed: list[dict[str, Any]]) -> None:
    if len(compressed) != len(batch):
        raise ValueError(
            f"Compressed batch size mismatch: expected {len(batch)}, got {len(compressed)}"
        )
    for src_record, out_record in zip(batch, compressed, strict=True):
        expected_id = str(src_record.get("sample_id", src_record.get("request_id", "")))
        got_id = str(out_record.get("id", ""))
        if got_id != expected_id:
            raise ValueError(
                f"ID mismatch in batch {batch_index}: expected {expected_id}, got {got_id}"
            )
        expected_caption = str(src_record.get("caption", src_record.get("original_description", "")))
        got_caption = str(out_record.get("original_caption", ""))
        if got_caption != expected_caption:
            raise ValueError(
                f"Caption mismatch in batch {batch_index} for {expected_id}"
            )


def append_error(error_log_path: Path | None, payload: dict[str, Any], lock: Lock) -> None:
    if error_log_path is None:
        return
    line = json.dumps(payload, ensure_ascii=False)
    with lock:
        with error_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def batch_checkpoint_path(checkpoint_dir: Path, batch_index: int) -> Path:
    return checkpoint_dir / f"batch_{batch_index:03d}.json"


def load_existing_batch(
    checkpoint_dir: Path | None,
    batch_index: int,
    batch: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    if checkpoint_dir is None:
        return None
    path = batch_checkpoint_path(checkpoint_dir, batch_index)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Checkpoint is not a JSON array: {path}")
    for item in data:
        if not isinstance(item, dict):
            raise ValueError(f"Checkpoint batch contains non-object item: {path}")
        validate_record_shape(item)
    validate_batch_alignment(batch_index, batch, data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=8.0)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=30)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set OPENAI_API_KEY.")

    records = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise SystemExit(f"Input must be a JSON array: {args.input}")

    normalized_inputs: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise SystemExit("Input array must contain JSON objects.")
        if "sample_id" in record and "caption" in record:
            normalized_inputs.append(record)
            continue
        if "request_id" in record and "description" in record:
            description = record.get("description", {})
            if not isinstance(description, dict):
                raise SystemExit("Annotation record has invalid description field.")
            first_section = description.get("first_section", {})
            second_section = description.get("second_section", {})
            third_section = description.get("third_section", {})
            normalized_inputs.append(
                {
                    "request_id": record["request_id"],
                    "original_description": first_section.get("description", ""),
                    "visual_attributes": second_section.get("visual_attributes", {}),
                    "emotional_impact": second_section.get("emotional_impact", ""),
                    "emotion_labels": third_section,
                    "image_path": record.get("image_path", ""),
                }
            )
            continue
        raise SystemExit(f"Unsupported record schema in {args.input}: {list(record.keys())}")

    batches = chunked(normalized_inputs, args.batch_size)

    if args.checkpoint_dir is not None:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    io_lock = Lock()
    error_log_path = (args.checkpoint_dir / "errors.jsonl") if args.checkpoint_dir is not None else None

    completed_by_index: dict[int, list[dict[str, Any]]] = {}
    work_items: list[tuple[int, list[dict[str, Any]]]] = []
    skipped_existing = 0
    for batch_index, batch in enumerate(batches, start=1):
        existing = load_existing_batch(args.checkpoint_dir, batch_index, batch)
        if existing is not None:
            completed_by_index[batch_index] = existing
            skipped_existing += 1
            continue
        work_items.append((batch_index, batch))

    print(f"total_records={len(normalized_inputs)}")
    print(f"total_batches={len(batches)}")
    print(f"workers={args.workers}")
    print(f"max_retries={args.max_retries}")
    print(f"skipped_existing={skipped_existing}")
    if error_log_path is not None:
        print(f"error_log={error_log_path}")

    def process_batch(batch_index: int, batch: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                print(
                    f"[{batch_index}/{len(batches)}] attempt={attempt} compressing {len(batch)} records..."
                )
                compressed = call_model(
                    api_key=args.api_key,
                    base_url=args.base_url,
                    model=args.model,
                    batch=batch,
                    temperature=args.temperature,
                    max_retries=1,
                    retry_sleep=args.retry_sleep,
                )
                validate_batch_alignment(batch_index, batch, compressed)
                if args.checkpoint_dir is not None:
                    batch_path = batch_checkpoint_path(args.checkpoint_dir, batch_index)
                    with io_lock:
                        batch_path.write_text(json.dumps(compressed, ensure_ascii=False, indent=2))
                return batch_index, compressed
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                append_error(
                    error_log_path,
                    {
                        "batch_index": batch_index,
                        "attempt": attempt,
                        "error": repr(exc),
                    },
                    io_lock,
                )
                time.sleep(args.retry_sleep)
        assert last_error is not None
        raise last_error

    completed = 0
    failed = 0
    if work_items:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_meta = {
                executor.submit(process_batch, batch_index, batch): (batch_index, batch)
                for batch_index, batch in work_items
            }
            for future in as_completed(future_to_meta):
                batch_index, batch = future_to_meta[future]
                try:
                    done_index, compressed = future.result()
                    completed_by_index[done_index] = compressed
                    completed += 1
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    print(f"FAILED batch={batch_index} size={len(batch)}: {exc}")

    missing = [i for i in range(1, len(batches) + 1) if i not in completed_by_index]
    if missing:
        raise SystemExit(f"Missing completed batches: {missing[:10]}{'...' if len(missing) > 10 else ''}")

    compressed_all: list[dict[str, Any]] = []
    for batch_index in range(1, len(batches) + 1):
        compressed_all.extend(completed_by_index[batch_index])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(compressed_all, ensure_ascii=False, indent=2))
    print(f"completed={completed} failed={failed}")
    print(f"Wrote {len(compressed_all)} compressed records to {args.output}")


if __name__ == "__main__":
    main()
