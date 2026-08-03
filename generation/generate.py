from __future__ import annotations

import argparse
import base64
import json
import os
import mimetypes
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one image with gpt-image-2 from a prompt file.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prompt-bundle-file", type=Path, default=None)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--image-name", default="generated.png")
    parser.add_argument("--response-name", default="response.json")
    parser.add_argument("--request-prompt-name", default="request_prompt.txt")
    parser.add_argument("--reference-paths-name", default="reference_paths_used.json")
    parser.add_argument("--request-mode-name", default="request_mode.json")
    parser.add_argument("--references-dir-name", default="references")
    parser.add_argument("--size", default=None, help="Optional size like 1024x1024. Omit by default so the API decides.")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--use-references", action="store_true")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--fallback-image-root", type=Path, default=None)
    parser.add_argument("--ink-annotation-json", type=Path, default=None)
    parser.add_argument("--china-annotation-json", type=Path, default=None)
    parser.add_argument(
        "--annotation-json",
        type=Path,
        action="append",
        default=[],
        help="Raw annotation JSON used to resolve reference image paths; repeat for multiple pools.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_payload(model: str, prompt: str, size: str | None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
    }
    if size:
        payload["size"] = size
    return payload


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_prompt_bundle_file(
    prompt_file: Path,
    explicit_bundle_file: Path | None,
) -> Path:
    if explicit_bundle_file and explicit_bundle_file.exists():
        return explicit_bundle_file

    candidates = [
        prompt_file.with_name("prompt_bundle.json"),
        prompt_file.parent / "prompt_bundle.json",
        prompt_file.parent.parent / "prompt_bundle.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing prompt bundle file. Tried:\n" + "\n".join(str(path) for path in candidates)
    )


def build_annotation_map(*annotation_paths: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for path in annotation_paths:
        if not path:
            continue
        records = load_json(path)
        for record in records:
            request_id = record.get("request_id") or record.get("id")
            image_path = record.get("image_path")
            if request_id and image_path and request_id not in mapping:
                mapping[request_id] = image_path
    return mapping


def collect_reference_ids(bundle: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("ink_topk", "china_topk"):
        for item in bundle.get(key) or []:
            ref_id = item.get("reference_id")
            if isinstance(ref_id, str) and ref_id and ref_id not in out:
                out.append(ref_id)
    return out


def normalize_relative_image_path(raw_path: str) -> str:
    text = raw_path.replace("\\", "/")
    if text.startswith("Images/"):
        text = text[len("Images/") :]
    return text


def resolve_reference_paths(
    *,
    prompt_bundle_file: Path,
    image_root: Path,
    fallback_image_root: Path | None,
    annotation_map: Dict[str, str],
) -> List[Path]:
    bundle = load_json(prompt_bundle_file)
    reference_paths: List[Path] = []
    missing: List[str] = []
    for ref_id in collect_reference_ids(bundle):
        raw_image_path = annotation_map.get(ref_id)
        if not raw_image_path:
            missing.append(f"{ref_id} (missing annotation image_path)")
            continue
        normalized = normalize_relative_image_path(raw_image_path)
        path = image_root / normalized
        if not path.exists() and fallback_image_root is not None:
            alt_path = fallback_image_root / normalized
            if alt_path.exists():
                path = alt_path
        if not path.exists():
            missing.append(f"{ref_id} -> {path}")
            continue
        reference_paths.append(path)
    if missing:
        raise FileNotFoundError("Missing reference images:\n" + "\n".join(missing[:20]))
    return reference_paths


def collect_local_reference_paths(prompt_file: Path, references_dir_name: str) -> List[Path]:
    references_dir = prompt_file.parent / references_dir_name
    if not references_dir.exists() or not references_dir.is_dir():
        return []

    reference_paths = [path for path in sorted(references_dir.iterdir()) if path.is_file()]
    if not reference_paths:
        raise FileNotFoundError(f"Reference directory exists but is empty: {references_dir}")
    return reference_paths


def encode_multipart_formdata(
    *,
    fields: Iterable[Tuple[str, str]],
    files: Iterable[Tuple[str, Path]],
) -> Tuple[bytes, str]:
    boundary = f"----CodexFormBoundary{uuid.uuid4().hex}"
    body = bytearray()
    boundary_bytes = boundary.encode("utf-8")

    for name, value in fields:
        body.extend(b"--" + boundary_bytes + b"\r\n")
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    for field_name, file_path in files:
        body.extend(b"--" + boundary_bytes + b"\r\n")
        body.extend(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"\r\n'.encode("utf-8")
        )
        body.extend(f"Content-Type: {guess_mime_type(file_path)}\r\n\r\n".encode("utf-8"))
        body.extend(file_path.read_bytes())
        body.extend(b"\r\n")

    body.extend(b"--" + boundary_bytes + b"--\r\n")
    return bytes(body), f"multipart/form-data; boundary={boundary}"


def extract_error_text(exc: Exception) -> str:
    parts = [str(exc)]
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            body = ""
        if body:
            parts.append(body)
    return "\n".join(part for part in parts if part)


def is_safety_error(exc: Exception) -> bool:
    text = extract_error_text(exc).lower()
    if isinstance(exc, urllib.error.HTTPError) and exc.code not in (400, 403):
        return False
    cues = (
        "safety",
        "policy",
        "content_filter",
        "content filter",
        "content_policy",
        "content policy",
        "violat",
        "flagged",
        "moderation",
        "unsafe",
    )
    return any(cue in text for cue in cues)


def request_generation(
    *,
    base_url: str,
    api_key: str,
    payload: Dict[str, Any],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    raw = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/images/generations",
        data=raw,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if is_safety_error(exc):
                print(f"[safety-stop] {extract_error_text(exc)[:1000]}")
                raise
            if attempt >= retries:
                raise
            print(f"[retry] attempt {attempt}/{retries} failed: {exc}")
            time.sleep(retry_sleep)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected generation failure")


def request_generation_with_references(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    reference_paths: List[Path],
    size: str | None,
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> Dict[str, Any]:
    fields: List[Tuple[str, str]] = [
        ("model", model),
        ("prompt", prompt),
    ]
    if size:
        fields.append(("size", size))
    files = [("image[]", path) for path in reference_paths]
    raw, content_type = encode_multipart_formdata(fields=fields, files=files)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/images/edits",
        data=raw,
        headers={
            "Content-Type": content_type,
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if is_safety_error(exc):
                print(f"[safety-stop] {extract_error_text(exc)[:1000]}")
                raise
            if attempt >= retries:
                raise
            print(f"[retry] attempt {attempt}/{retries} failed: {exc}")
            time.sleep(retry_sleep)
    if last_error is not None:
        raise last_error
    raise RuntimeError("Unexpected generation failure")


def main() -> None:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    prompt = args.prompt_file.read_text(encoding="utf-8")
    ensure_dir(args.output_dir)

    image_path = args.output_dir / args.image_name
    response_path = args.output_dir / args.response_name
    if image_path.exists() and response_path.exists() and not args.overwrite:
        print(image_path)
        return

    print(f"[generate] model={args.model}")
    print(f"[generate] prompt_file={args.prompt_file}")
    print(f"[generate] size={'<api-default>' if not args.size else args.size}")

    request_prompt_path = args.output_dir / args.request_prompt_name
    request_prompt_path.write_text(prompt, encoding="utf-8")

    if args.use_references:
        prompt_bundle_file: Path | None = None
        local_reference_paths = collect_local_reference_paths(args.prompt_file, args.references_dir_name)
        if local_reference_paths:
            reference_paths = local_reference_paths
            request_mode = {
                "endpoint": "/images/edits",
                "reference_count": len(reference_paths),
                "prompt_file": str(args.prompt_file),
                "reference_source": "local_references_dir",
                "references_dir": str(args.prompt_file.parent / args.references_dir_name),
            }
        else:
            prompt_bundle_file = resolve_prompt_bundle_file(args.prompt_file, args.prompt_bundle_file)
            if not args.image_root:
                raise SystemExit("--image-root is required when --use-references is set and local references/ is absent")
            annotation_paths = list(args.annotation_json)
            annotation_paths.extend(
                path for path in (args.ink_annotation_json, args.china_annotation_json) if path is not None
            )
            if not annotation_paths:
                raise SystemExit(
                    "At least one --annotation-json is required when --use-references is set "
                    "and local references/ is absent."
                )

            annotation_map = build_annotation_map(*annotation_paths)
            reference_paths = resolve_reference_paths(
                prompt_bundle_file=prompt_bundle_file,
                image_root=args.image_root,
                fallback_image_root=args.fallback_image_root,
                annotation_map=annotation_map,
            )
            request_mode = {
                "endpoint": "/images/edits",
                "reference_count": len(reference_paths),
                "prompt_file": str(args.prompt_file),
                "prompt_bundle_file": str(prompt_bundle_file),
                "reference_source": "prompt_bundle_and_annotations",
            }
        (args.output_dir / args.reference_paths_name).write_text(
            json.dumps([str(path) for path in reference_paths], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (args.output_dir / args.request_mode_name).write_text(
            json.dumps(request_mode, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if prompt_bundle_file is not None:
            print(f"[generate] prompt_bundle_file={prompt_bundle_file}")
        print(f"[generate] reference_count={len(reference_paths)}")
        data = request_generation_with_references(
            base_url=args.base_url,
            api_key=api_key,
            model=args.model,
            prompt=prompt,
            reference_paths=reference_paths,
            size=args.size,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )
    else:
        payload = build_payload(args.model, prompt, args.size)
        (args.output_dir / args.request_mode_name).write_text(
            json.dumps(
                {
                    "endpoint": "/images/generations",
                    "reference_count": 0,
                    "prompt_file": str(args.prompt_file),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        data = request_generation(
            base_url=args.base_url,
            api_key=api_key,
            payload=payload,
            timeout=args.timeout,
            retries=args.retries,
            retry_sleep=args.retry_sleep,
        )

    response_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    image_bytes = base64.b64decode(data["data"][0]["b64_json"])
    image_path.write_bytes(image_bytes)
    print(image_path)


if __name__ == "__main__":
    main()
