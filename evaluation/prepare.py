from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build official_like_eval_run metadata.json files for batch AAS evaluation.")
    parser.add_argument("--test-json", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, default=None)
    parser.add_argument(
        "--image-path",
        type=Path,
        default=None,
        help="Evaluate one explicit image. Requires one --sample-id and --flat-output-root.",
    )
    parser.add_argument("--image-subdir", default="gpt-image-2")
    parser.add_argument("--image-name", default=None, help="Default: {sample_id}.png")
    parser.add_argument("--output-subdir", default="official_like_eval_run")
    parser.add_argument(
        "--style-preservation-text",
        default=(
            "Preserve the requested artistic style identity exactly as specified in the original task. "
            "Keep the image visibly within the required style family, medium logic, period-consistent visual language, "
            "brushwork or line treatment, palette tendencies, composition cues, support or format cues, and overall atmosphere. "
            "Do not drift into a generic modern digital illustration look unless the task itself asks for that."
        ),
    )
    parser.add_argument(
        "--negative-constraints-text",
        default=(
            "Avoid unrelated scene drift. Avoid replacing the requested style, composition, support, or format with a generic alternative. "
            "Avoid unnecessary saturation, over-clean digital polish, attribute omissions, or stylistic blending that weakens the requested identity."
        ),
    )
    parser.add_argument(
        "--flat-output-root",
        type=Path,
        default=None,
        help="Optional flat run root like .../official_like_eval_run_all . If set, write {flat_output_root}/{sample_id}/metadata.json for compatibility with evaluate_aas_official_like.py.",
    )
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_test_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_json(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        sample_key = None
        if isinstance(row.get("id"), str):
            sample_key = row["id"]
        elif isinstance(row.get("sample_id"), str):
            sample_key = row["sample_id"]
        if sample_key:
            out[sample_key] = row
    return out


def build_prompt(caption: str, style_preservation_text: str, negative_constraints_text: str) -> str:
    return (
        "Original task:\n"
        f"{caption}\n\n"
        "Style preservation:\n"
        f"{style_preservation_text}\n\n"
        "Reference usage:\n"
        "Judge alignment against the original task itself. Do not assume additional content beyond the stated caption.\n\n"
        "Negative constraints:\n"
        f"{negative_constraints_text}\n"
    )


def main() -> None:
    args = parse_args()
    test_map = load_test_map(args.test_json)
    requested = set(args.sample_id)

    if args.image_path is not None:
        if len(args.sample_id) != 1 or args.flat_output_root is None:
            raise SystemExit("--image-path requires exactly one --sample-id and --flat-output-root.")
        sample_id = args.sample_id[0]
        record = test_map.get(sample_id)
        if record is None:
            raise SystemExit(f"Missing test record: {sample_id}")
        if not args.image_path.exists():
            raise SystemExit(f"Missing image: {args.image_path}")
        out_dir = args.flat_output_root / sample_id
        out_path = out_dir / "metadata.json"
        ensure_dir(out_dir)
        caption = str(record.get("original_caption") or record.get("caption") or "").strip()
        metadata = {
            "output_image": str(args.image_path.resolve()),
            "prompt": build_prompt(caption, args.style_preservation_text, args.negative_constraints_text),
        }
        out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {sample_id} :: {out_path}")
        return

    if args.generation_root is None:
        raise SystemExit("Pass --generation-root for a batch or --image-path for one image.")

    sample_dirs = sorted(
        p for p in args.generation_root.iterdir()
        if p.is_dir() and (not requested or p.name in requested)
    )
    print(f"[build] sample_count={len(sample_dirs)}")

    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        record = test_map.get(sample_id)
        if not record:
            print(f"[skip] {sample_id} :: missing test record")
            continue

        image_name = args.image_name or f"{sample_id}.png"
        image_path = sample_dir / args.image_subdir / image_name
        if not image_path.exists():
            print(f"[skip] {sample_id} :: missing image {image_path}")
            continue

        if args.flat_output_root is not None:
            out_dir = args.flat_output_root / sample_id
        else:
            out_dir = sample_dir / args.output_subdir / sample_id
        out_path = out_dir / "metadata.json"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {sample_id} :: exists")
            continue

        ensure_dir(out_dir)
        caption = str(record.get("original_caption") or record.get("caption") or "").strip()
        metadata = {
            "output_image": str(image_path),
            "prompt": build_prompt(
                caption,
                args.style_preservation_text,
                args.negative_constraints_text,
            ),
        }
        out_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[ok] {sample_id} :: {out_path}")


if __name__ == "__main__":
    main()
