from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def join_field_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "; ".join(parts)
    if value is None:
        return ""
    return str(value).strip()


def normalize_id(record: dict[str, Any]) -> str:
    for key in ("sample_id", "id", "request_id"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(f"Cannot normalize id from record keys: {list(record.keys())}")


def build_id_map(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for record in records:
        out[normalize_id(record)] = record
    return out


def image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def extract_message_text(message_content: Any) -> str:
    if isinstance(message_content, str):
        return message_content
    if isinstance(message_content, list):
        parts: list[str] = []
        for item in message_content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(part for part in parts if part).strip()
    return str(message_content)


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", cleaned, re.DOTALL)
    if fenced:
        cleaned = fenced.group(1)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output does not contain a JSON object.")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is not an object.")
    return parsed


def call_openai_multimodal_json(
    *,
    base_url: str,
    api_key_env: str,
    model: str,
    system_prompt: str,
    user_text: str,
    image_paths: list[Path],
    temperature: float = 0.1,
    max_tokens: int = 12000,
    timeout: int = 900,
    retries: int = 3,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {api_key_env}")

    content: list[dict[str, Any]] = [{"type": "text", "text": user_text}]
    for image_path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": image_to_data_uri(image_path)}})

    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
    }
    raw = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=raw,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            text = extract_message_text(result["choices"][0]["message"]["content"])
            return parse_json_object(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt >= retries:
                raise
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            print(f"[retry] multimodal plan attempt {attempt}/{retries} failed: {exc}")
            time.sleep(retry_sleep)
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected multimodal request failure")
