from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch generate gpt-image-2 images from prompt.txt files.")
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--single-script", type=Path, default=Path(__file__).with_name("generate.py"))
    parser.add_argument("--prompt-name", default="prompt.txt")
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=3.0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--model", default="gpt-image-2")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--size", default=None)
    parser.add_argument("--use-references", action="store_true")
    parser.add_argument("--image-root", type=Path, default=None)
    parser.add_argument("--fallback-image-root", type=Path, default=None)
    parser.add_argument("--ink-annotation-json", type=Path, default=None)
    parser.add_argument("--china-annotation-json", type=Path, default=None)
    parser.add_argument("--annotation-json", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-subdir", default="gpt-image-2")
    parser.add_argument("--image-name", default=None, help="Default: {sample_id}.png")
    parser.add_argument("--response-name", default="response.json")
    parser.add_argument(
        "--sample-dir-levels-up",
        type=int,
        default=1,
        help="How many directory levels to go up from the prompt file to recover the sample dir. "
        "Default 1 means sample_dir/prompt.txt. Use 2 for sample_dir/<subdir>/prompt.txt.",
    )
    return parser.parse_args()


def collect_samples(
    generation_root: Path,
    requested: List[str],
    prompt_name: str,
    sample_dir_levels_up: int,
) -> List[Path]:
    requested_set = set(requested)
    sample_dirs: List[Path] = []
    pattern = f"*/{prompt_name}"
    for prompt_file in sorted(generation_root.glob(pattern)):
        if sample_dir_levels_up < 1:
            raise ValueError("--sample-dir-levels-up must be >= 1")
        sample_dir = prompt_file
        for _ in range(sample_dir_levels_up):
            sample_dir = sample_dir.parent
        if requested_set and sample_dir.name not in requested_set:
            continue
        sample_dirs.append(sample_dir)
    return sample_dirs


def build_paths(
    sample_dir: Path,
    output_subdir: str,
    image_name: str | None,
    response_name: str,
    prompt_name: str,
) -> Dict[str, Path]:
    sample_id = sample_dir.name
    out_dir = sample_dir / output_subdir
    img_name = image_name or f"{sample_id}.png"
    return {
        "prompt_file": sample_dir / prompt_name,
        "out_dir": out_dir,
        "image_path": out_dir / img_name,
        "response_path": out_dir / response_name,
    }


def run_one(sample_dir: Path, args: argparse.Namespace) -> Dict[str, str]:
    sample_id = sample_dir.name
    paths = build_paths(sample_dir, args.output_subdir, args.image_name, args.response_name, args.prompt_name)
    error_path = paths["out_dir"] / "error.json"

    if (
        not args.overwrite
        and paths["image_path"].exists()
        and paths["response_path"].exists()
    ):
        return {"sample_id": sample_id, "status": "skipped", "message": str(paths["image_path"])}

    cmd = [
        sys.executable,
        str(args.single_script),
        "--prompt-file",
        str(paths["prompt_file"]),
        "--output-dir",
        str(paths["out_dir"]),
        "--model",
        args.model,
        "--api-key-env",
        args.api_key_env,
        "--timeout",
        str(args.timeout),
        "--retries",
        str(args.retries),
        "--retry-sleep",
        str(args.retry_sleep),
        "--response-name",
        args.response_name,
        "--image-name",
        paths["image_path"].name,
    ]
    if args.base_url:
        cmd.extend(["--base-url", args.base_url])
    if args.size:
        cmd.extend(["--size", args.size])
    if args.overwrite:
        cmd.append("--overwrite")
    if args.use_references:
        cmd.append("--use-references")
        if args.image_root:
            cmd.extend(["--image-root", str(args.image_root)])
        if args.fallback_image_root:
            cmd.extend(["--fallback-image-root", str(args.fallback_image_root)])
        if args.ink_annotation_json:
            cmd.extend(["--ink-annotation-json", str(args.ink_annotation_json)])
        if args.china_annotation_json:
            cmd.extend(["--china-annotation-json", str(args.china_annotation_json)])
        for annotation_path in args.annotation_json:
            cmd.extend(["--annotation-json", str(annotation_path)])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        if error_path.exists():
            error_path.unlink()
        return {"sample_id": sample_id, "status": "success", "message": proc.stdout.strip() or str(paths["image_path"])}
    paths["out_dir"].mkdir(parents=True, exist_ok=True)
    error_payload = {
        "sample_id": sample_id,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    error_path.write_text(json.dumps(error_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sample_id": sample_id,
        "status": "failed",
        "message": (proc.stderr.strip() or proc.stdout.strip() or f"returncode={proc.returncode}")[:4000],
    }


def main() -> None:
    args = parse_args()
    samples = collect_samples(
        args.generation_root,
        args.sample_id,
        args.prompt_name,
        args.sample_dir_levels_up,
    )
    print(
        f"[batch] sample_count={len(samples)} workers={args.workers} "
        f"output_subdir={args.output_subdir} prompt_name={args.prompt_name}"
    )

    results: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {executor.submit(run_one, sample_dir, args): sample_dir.name for sample_dir in samples}
        for future in as_completed(future_map):
            result = future.result()
            results.append(result)
            print(f"[{result['status']}] {result['sample_id']} :: {result['message']}")

    summary = {
        "sample_count": len(results),
        "success": sum(1 for r in results if r["status"] == "success"),
        "skipped": sum(1 for r in results if r["status"] == "skipped"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "results": sorted(results, key=lambda x: x["sample_id"]),
    }
    summary_path = args.generation_root / f"batch_generate_summary_{args.output_subdir}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[batch] wrote {summary_path}")


if __name__ == "__main__":
    main()
