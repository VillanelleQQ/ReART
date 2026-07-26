from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from common import load_json


def guess_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    return mime or "application/octet-stream"


def encode_multipart_formdata(fields, files):
    boundary = f"----RefineBoundary{uuid.uuid4().hex}"
    body = bytearray()
    boundary_bytes = boundary.encode("utf-8")

    for name, value in fields:
        body.extend(b"--" + boundary_bytes + b"\r\n")
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(str(value).encode("utf-8"))
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


def request_edit(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    image_paths: list[Path],
    timeout: int,
    retries: int,
    retry_sleep: float,
) -> dict:
    fields = [("model", model), ("prompt", prompt)]
    files = [("image[]", path) for path in image_paths]
    raw, content_type = encode_multipart_formdata(fields, files)
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/images/edits",
        data=raw,
        headers={
            "Content-Type": content_type,
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
            if attempt >= retries:
                raise
            print(f"[retry] refine edit attempt {attempt}/{retries} failed: {exc}")
            time.sleep(retry_sleep)
    if last_error:
        raise last_error
    raise RuntimeError("Unexpected refine edit failure")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run gpt-image-2 refine edit from a refine packet directory.")
    parser.add_argument("--packet-dir", type=Path, required=True)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--output-subdir", default="gpt-image-2-refine")
    args = parser.parse_args()

    packet_dir = args.packet_dir.resolve()
    output_dir = packet_dir / args.output_subdir
    prompt_file = packet_dir / "prompt_zh_refine.txt"
    bundle_file = packet_dir / "prompt_bundle_refine.json"
    refs_file = packet_dir / "reference_paths_used.json"
    if not prompt_file.exists() or not bundle_file.exists() or not refs_file.exists():
        raise SystemExit("Missing refine packet files.")

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise SystemExit(f"Missing API key env var: {args.api_key_env}")

    prompt = prompt_file.read_text(encoding="utf-8")
    bundle = load_json(bundle_file)
    current_image_path = Path(str(bundle["current_image_path"]))
    reference_paths = [Path(x) for x in load_json(refs_file)]
    image_paths = [current_image_path] + reference_paths
    output_dir.mkdir(parents=True, exist_ok=True)

    response = request_edit(
        base_url=args.base_url,
        api_key=api_key,
        model=args.model,
        prompt=prompt,
        image_paths=image_paths,
        timeout=args.timeout,
        retries=args.retries,
        retry_sleep=args.retry_sleep,
    )
    (output_dir / "response.json").write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
    image_bytes = base64.b64decode(response["data"][0]["b64_json"])
    (output_dir / "refined.png").write_bytes(image_bytes)
    (output_dir / "request_mode.json").write_text(
        json.dumps(
            {
                "endpoint": "/images/edits",
                "input_image_count": len(image_paths),
                "current_image_path": str(current_image_path),
                "reference_paths": [str(path) for path in reference_paths],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir / "refined.png")


if __name__ == "__main__":
    main()
