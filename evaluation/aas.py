from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import threading
import traceback
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AAS with an official-like multimodal judge.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Run directory containing per-sample metadata.json.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where result json files are written.")
    parser.add_argument("--sample-id", action="append", default=[], help="Optional sample ids to restrict evaluation.")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="Judge provider backend.")
    parser.add_argument("--judge-model", default=os.environ.get("OPENAI_JUDGE_MODEL", "gpt-5.4"), help="Multimodal judge model name.")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"), help="OpenAI-compatible base url.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable containing the API key.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel sample workers.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-request timeout in seconds.")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate samples with existing output json.")
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_existing_results(output_dir: Path) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("*_aas.json")):
        try:
            results.append(load_json(path))
        except Exception:
            continue
    return results


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON object in response: {text[:400]}")
    candidate = match.group(0).strip()

    # Common Gemini / VLM malformed JSON repairs:
    # 1. missing comma between adjacent lines of array/object members
    # 2. trailing commas before ] or }
    # 3. smart quotes are not expected here, but normalize if present
    repaired = candidate.replace("“", '"').replace("”", '"').replace("’", "'")
    repaired = re.sub(r'"\s*\n\s*"', '",\n  "', repaired)
    repaired = re.sub(r'(\]|\}|"|\d)\s*\n\s*"', r'\1,\n  "', repaired)
    repaired = re.sub(r",\s*([\]}])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Could not parse judge JSON even after repair. Original excerpt: {candidate[:500]}"
        ) from exc


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def image_to_data_uri(image_path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def split_prompt_sections(prompt_text: str) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []

    for line in prompt_text.splitlines():
        stripped = line.strip()
        if stripped.endswith(":") and stripped[:-1] in {
            "Original task",
            "Style preservation",
            "Reference usage",
            "Negative constraints",
            "Candidate directive",
        }:
            if current_key is not None:
                sections[current_key] = "\n".join(current_lines).strip()
            current_key = stripped[:-1]
            current_lines = []
        else:
            current_lines.append(line)

    if current_key is not None:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


def collect_samples(run_dir: Path, requested_ids: Iterable[str]) -> List[Tuple[str, Path]]:
    requested = set(requested_ids)
    pairs: List[Tuple[str, Path]] = []
    for meta_path in sorted(run_dir.glob("*/metadata.json")):
        sample_id = meta_path.parent.name
        if requested and sample_id not in requested:
            continue
        pairs.append((sample_id, meta_path))
    return pairs


def extract_candidate_view(metadata: Dict[str, Any], metadata_path: Path) -> Tuple[Path, str]:
    """Support both GPT-image-2 run metadata and HiDream run metadata."""
    candidate = (metadata.get("candidates") or [None])[0]
    if isinstance(candidate, dict):
        image_value = candidate.get("image_path")
        prompt_text = candidate.get("prompt_text", "")
        if isinstance(image_value, str) and image_value.strip():
            image_path = Path(image_value)
            if not image_path.is_absolute():
                image_path = ROOT / image_value
            return image_path, str(prompt_text or "")

    output_image = metadata.get("output_image")
    prompt_text = metadata.get("prompt", "")
    if isinstance(output_image, str) and output_image.strip():
        image_path = Path(output_image)
        if not image_path.is_absolute():
            image_path = ROOT / output_image
        return image_path, str(prompt_text or "")

    fallback_image = metadata_path.parent / "candidate_01.png"
    if fallback_image.exists():
        return fallback_image, str(prompt_text or "")

    raise ValueError(f"No candidates found in {metadata_path}")


def build_content_prompt(sample_id: str, caption: str, style_section: str) -> str:
    return f"""You are evaluating CONTENT ALIGNMENT for an artistic text-to-image result.

Task:
- Judge whether the generated image correctly realizes the major visual content described in the caption.
- Focus on main subjects, required scene type, major relations, and obvious omissions or contradictions.
- Do NOT score artistic style here unless it affects content readability.

Benchmark inspirations:
- TIFA / DSG / LongT2IExpert-style structured checking
- Think in terms of entities, salient attributes, and relations

Sample id: {sample_id}

Caption:
{caption}

Additional style context:
{style_section or "(none)"}

Scoring rubric:
- 9-10: almost all major required visual content is present and correctly related
- 7-8: mostly correct, with minor missing or weakly rendered details
- 5-6: partial realization; some major content is missing or ambiguous
- 3-4: substantial content mismatch
- 0-2: severe mismatch or unrelated image

Return STRICT JSON with this schema:
{{
  "score_0_to_10": number,
  "aligned_entities": [string],
  "missing_entities": [string],
  "relation_issues": [string],
  "content_summary": string
}}"""


def build_style_prompt(sample_id: str, caption: str, style_section: str, negative_section: str) -> str:
    return f"""You are evaluating STYLE ALIGNMENT for an artistic text-to-image result.

Task:
- Judge whether the image realizes the requested artistic style identity.
- Focus on style family, medium logic, period-consistent visual language, and style drift.
- Ignore content omissions unless they directly prevent judging style.

Benchmark inspirations:
- EvalMuse-40K fine-grained style axis
- FineGRAIN style drift / blending style failure modes

Sample id: {sample_id}

Caption:
{caption}

Style preservation guidance:
{style_section or "(none)"}

Negative constraints:
{negative_section or "(none)"}

Scoring rubric:
- 9-10: style identity is strong, specific, and faithful
- 7-8: clear style identity with mild drift
- 5-6: partially correct but generic or unstable style
- 3-4: weak style match or adjacent-style drift
- 0-2: style largely wrong

Return STRICT JSON with this schema:
{{
  "score_0_to_10": number,
  "matched_style_cues": [string],
  "style_drift_cues": [string],
  "medium_or_material_notes": [string],
  "style_summary": string
}}"""


def build_attribute_prompt(sample_id: str, caption: str, style_section: str, negative_section: str) -> str:
    return f"""You are evaluating ATTRIBUTE ALIGNMENT for an artistic text-to-image result.

Task:
- Judge whether the image realizes the caption-requested artistic attributes when applicable.
- Prioritize composition, brushwork, line quality, palette/color handling, lighting, and mood-carrying visual attributes.
- Only judge attributes that are actually requested or strongly implied by the caption/style guidance.

Benchmark inspirations:
- LongT2IExpert structured attribute checking
- FineGRAIN attribute binding failures such as color binding, texture binding, and visual relation misbinding

Sample id: {sample_id}

Caption:
{caption}

Style preservation guidance:
{style_section or "(none)"}

Negative constraints:
{negative_section or "(none)"}

Scoring rubric:
- 9-10: requested attributes are clearly and faithfully realized
- 7-8: mostly correct with small misses
- 5-6: mixed realization; some requested attributes are weak or absent
- 3-4: multiple key attributes are wrong or missing
- 0-2: attributes mostly fail

Return STRICT JSON with this schema:
{{
  "score_0_to_10": number,
  "requested_attributes": [string],
  "well_realized_attributes": [string],
  "failed_attributes": [string],
  "attribute_summary": string
}}"""


def call_openai_chat_completion(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_text: str,
    image_path: Path,
    timeout: int,
) -> Dict[str, Any]:
    endpoint = f"{normalize_base_url(base_url)}/chat/completions"
    data_uri = image_to_data_uri(image_path)

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
    }

    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_gemini_generate_content(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt_text: str,
    image_path: Path,
    timeout: int,
) -> Dict[str, Any]:
    endpoint = f"{normalize_base_url(base_url)}/models/{model}:generateContent?key={api_key}"
    mime, _ = mimetypes.guess_type(str(image_path))
    mime = mime or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")

    payload = {
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": mime,
                            "data": encoded,
                        }
                    },
                ],
            }
        ],
    }
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_judge_content(response_payload: Dict[str, Any]) -> Dict[str, Any]:
    content = None

    choices = response_payload.get("choices") or []
    if choices:
        message = choices[0].get("message", {})
        content = message.get("content", "")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
            content = "\n".join(parts).strip()

    if content is None:
        candidates = response_payload.get("candidates") or []
        if candidates:
            parts = ((candidates[0].get("content") or {}).get("parts")) or []
            text_parts = [part.get("text", "") for part in parts if isinstance(part, dict) and part.get("text")]
            content = "\n".join(text_parts).strip()

    if not isinstance(content, str) or not content.strip():
        raise ValueError(f"Empty judge content: {response_payload}")
    return {
        "raw_text": content,
        "parsed_json": extract_json_object(content),
    }


def evaluate_single_candidate(
    *,
    sample_id: str,
    metadata_path: Path,
    output_dir: Path,
    provider: str,
    judge_model: str,
    base_url: str,
    api_key: str,
    timeout: int,
    overwrite: bool,
) -> Dict[str, Any]:
    out_path = output_dir / f"{sample_id}_aas.json"
    if out_path.exists() and not overwrite:
        return load_json(out_path)

    metadata = load_json(metadata_path)
    image_path, prompt_text = extract_candidate_view(metadata, metadata_path)
    sections = split_prompt_sections(prompt_text)
    caption = sections.get("Original task", prompt_text).strip()
    style_section = sections.get("Style preservation", "").strip()
    negative_section = sections.get("Negative constraints", "").strip()

    content_prompt = build_content_prompt(sample_id, caption, style_section)
    style_prompt = build_style_prompt(sample_id, caption, style_section, negative_section)
    attribute_prompt = build_attribute_prompt(sample_id, caption, style_section, negative_section)

    call_fn = call_openai_chat_completion if provider == "openai" else call_gemini_generate_content

    content_resp = call_fn(
        base_url=base_url,
        api_key=api_key,
        model=judge_model,
        prompt_text=content_prompt,
        image_path=image_path,
        timeout=timeout,
    )
    style_resp = call_fn(
        base_url=base_url,
        api_key=api_key,
        model=judge_model,
        prompt_text=style_prompt,
        image_path=image_path,
        timeout=timeout,
    )
    attribute_resp = call_fn(
        base_url=base_url,
        api_key=api_key,
        model=judge_model,
        prompt_text=attribute_prompt,
        image_path=image_path,
        timeout=timeout,
    )

    content_result = parse_judge_content(content_resp)
    style_result = parse_judge_content(style_resp)
    attribute_result = parse_judge_content(attribute_resp)

    content_score = float(content_result["parsed_json"]["score_0_to_10"])
    style_score = float(style_result["parsed_json"]["score_0_to_10"])
    attribute_score = float(attribute_result["parsed_json"]["score_0_to_10"])
    aas_mean_10 = (content_score + style_score + attribute_score) / 3.0

    result = {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "metadata_path": str(metadata_path),
        "provider": provider,
        "judge_model": judge_model,
        "base_url": base_url,
        "caption": caption,
        "prompt_sections": sections,
        "scores": {
            "content_0_to_10": content_score,
            "style_0_to_10": style_score,
            "attribute_0_to_10": attribute_score,
            "aas_mean_0_to_10": aas_mean_10,
            "aas_mean_0_to_1": aas_mean_10 / 10.0,
        },
        "content_judge": content_result,
        "style_judge": style_result,
        "attribute_judge": attribute_result,
    }
    write_json(out_path, result)
    return result


def write_error_result(
    *,
    output_dir: Path,
    sample_id: str,
    metadata_path: Path,
    provider: str,
    judge_model: str,
    base_url: str,
    exc: Exception,
) -> Path:
    out_path = output_dir / f"{sample_id}_error.json"
    payload = {
        "sample_id": sample_id,
        "metadata_path": str(metadata_path),
        "provider": provider,
        "judge_model": judge_model,
        "base_url": base_url,
        "error_type": exc.__class__.__name__,
        "error_message": str(exc),
        "traceback": traceback.format_exc(),
    }
    write_json(out_path, payload)
    return out_path


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {
            "sample_count": 0,
            "means": {},
        }

    def mean(key: str) -> float:
        values = [float(r["scores"][key]) for r in results]
        return sum(values) / len(values)

    return {
        "sample_count": len(results),
        "means": {
            "content_0_to_10": mean("content_0_to_10"),
            "style_0_to_10": mean("style_0_to_10"),
            "attribute_0_to_10": mean("attribute_0_to_10"),
            "aas_mean_0_to_10": mean("aas_mean_0_to_10"),
            "aas_mean_0_to_1": mean("aas_mean_0_to_1"),
        },
    }


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    if args.provider == "gemini" and args.base_url == "https://api.openai.com/v1":
        args.base_url = DEFAULT_GEMINI_BASE_URL

    samples = collect_samples(args.run_dir, args.sample_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()

    def worker(sample: Tuple[str, Path]) -> Dict[str, Any]:
        sample_id, metadata_path = sample
        try:
            return evaluate_single_candidate(
                sample_id=sample_id,
                metadata_path=metadata_path,
                output_dir=args.output_dir,
                provider=args.provider,
                judge_model=args.judge_model,
                base_url=args.base_url,
                api_key=api_key,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            error_path = write_error_result(
                output_dir=args.output_dir,
                sample_id=sample_id,
                metadata_path=metadata_path,
                provider=args.provider,
                judge_model=args.judge_model,
                base_url=args.base_url,
                exc=exc,
            )
            print(f"[error] {sample_id} -> {error_path}: {exc}")
            return {}

    if args.workers <= 1:
        for sample in samples:
            result = worker(sample)
            if result:
                results.append(result)
                print(f"Wrote {args.output_dir / (result['sample_id'] + '_aas.json')}")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {executor.submit(worker, sample): sample for sample in samples}
            for future in as_completed(future_map):
                result = future.result()
                if result:
                    with lock:
                        results.append(result)
                    print(f"Wrote {args.output_dir / (result['sample_id'] + '_aas.json')}")

    # Always summarize from all successful outputs on disk so interrupted/resumed
    # runs still produce a valid summary even when some samples failed.
    summary_results = load_existing_results(args.output_dir)
    summary = aggregate_results(sorted(summary_results, key=lambda r: r["sample_id"]))
    summary_path = args.output_dir / "summary.json"
    write_json(summary_path, summary)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
