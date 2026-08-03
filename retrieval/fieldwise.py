from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


FIELD_ORDER = [
    "main_subjects",
    "spatial_relations",
    "scene_or_setting",
    "composition_or_layout",
    "material_or_surface",
    "brushstroke_or_texture",
    "line_quality",
    "color_or_tone",
    "mood_or_atmosphere",
]

DEFAULT_WEIGHTS = {
    "main_subjects": 0.25,
    "spatial_relations": 0.10,
    "scene_or_setting": 0.15,
    "composition_or_layout": 0.20,
    "material_or_surface": 0.05,
    "brushstroke_or_texture": 0.10,
    "line_quality": 0.05,
    "color_or_tone": 0.07,
    "mood_or_atmosphere": 0.03,
}

LAYOUT_HEAVY_WEIGHTS = {
    "main_subjects": 0.22,
    "spatial_relations": 0.08,
    "scene_or_setting": 0.18,
    "composition_or_layout": 0.24,
    "material_or_surface": 0.08,
    "brushstroke_or_texture": 0.08,
    "line_quality": 0.04,
    "color_or_tone": 0.05,
    "mood_or_atmosphere": 0.03,
}

LAYOUT_CUES = (
    "fan-shaped",
    "folding fan",
    "folded fan",
    "album page",
    "album leaf",
    "album spread",
    "open album",
    "open page",
    "open spread",
    "vertical scroll",
    "hanging scroll",
    "two-page",
    "left-right",
    "panel",
    "upper and lower",
    "vertical arrangement",
    "mounting",
    "handscroll",
)


def choose_device(force_device: str | None) -> str:
    if force_device:
        return force_device
    return "cuda" if torch.cuda.is_available() else "cpu"


def join_field_text(value: Any) -> str:
    if isinstance(value, list):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return "; ".join(parts)
    if value is None:
        return ""
    return str(value).strip()


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class HFEmbedder:
    def __init__(self, model_name: str, device: str, max_length: int) -> None:
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int) -> torch.Tensor:
        if not texts:
            raise ValueError("encode() received an empty text list")
        all_vecs: list[torch.Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
            outputs = self.model(**tokenized)
            if hasattr(outputs, "last_hidden_state"):
                pooled = mean_pool(outputs.last_hidden_state, tokenized["attention_mask"])
            elif isinstance(outputs, tuple) and outputs:
                pooled = mean_pool(outputs[0], tokenized["attention_mask"])
            else:
                raise ValueError("Model output does not contain last_hidden_state")
            pooled = F.normalize(pooled, p=2, dim=1)
            all_vecs.append(pooled.cpu())
        return torch.cat(all_vecs, dim=0)


def load_json_array(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected JSON array in {path}")
    return [x for x in data if isinstance(x, dict)]


def is_abstract_record(record: dict[str, Any]) -> bool:
    style_text = " ".join(str(x) for x in record.get("style", []))
    scene_text = " ".join(str(x) for x in record.get("scene_or_setting", []))
    caption_text = str(record.get("compressed_caption", ""))
    merged = f"{style_text} {scene_text} {caption_text}".lower()
    return "abstract" in merged


def has_strong_layout_requirement(record: dict[str, Any]) -> bool:
    texts = []
    for field in ("scene_or_setting", "composition_or_layout", "material_or_surface"):
        texts.append(join_field_text(record.get(field, [])))
    merged = " ".join(texts).lower()
    return any(cue in merged for cue in LAYOUT_CUES)


def renormalize_weights(weights: dict[str, float], active_fields: list[str]) -> dict[str, float]:
    subset = {field: weights[field] for field in active_fields}
    total = sum(subset.values())
    if total <= 0:
        return {field: 0.0 for field in active_fields}
    return {field: value / total for field, value in subset.items()}


def precompute_field_embeddings(
    records: list[dict[str, Any]],
    fields: list[str],
    embedder: HFEmbedder,
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for field in fields:
        texts = [join_field_text(record.get(field, [])) for record in records]
        non_empty_indices = [idx for idx, text in enumerate(texts) if text]
        if non_empty_indices:
            encoded = embedder.encode([texts[idx] for idx in non_empty_indices], batch_size=batch_size)
            embeddings = torch.zeros((len(records), encoded.shape[1]), dtype=encoded.dtype)
            for row_idx, record_idx in enumerate(non_empty_indices):
                embeddings[record_idx] = encoded[row_idx]
        else:
            embeddings = torch.zeros((len(records), 1), dtype=torch.float32)
        result[field] = {
            "texts": texts,
            "embeddings": embeddings,
            "non_empty": non_empty_indices,
        }
        print(f"precomputed field={field} non_empty={len(non_empty_indices)}/{len(records)}")
    return result


def compute_field_similarity(
    query_text: str,
    query_embedding: torch.Tensor | None,
    ref_embeddings: torch.Tensor,
    ref_texts: list[str],
) -> torch.Tensor:
    if not query_text or query_embedding is None:
        return torch.full((len(ref_texts),), float("nan"))
    query_vec = query_embedding
    if query_vec.dim() == 2:
        query_vec = query_vec[0]
    scores = torch.mv(ref_embeddings, query_vec)
    empty_mask = torch.tensor([not text for text in ref_texts], dtype=torch.bool)
    scores[empty_mask] = float("nan")
    return scores


def topk_indices(values: torch.Tensor, top_k: int) -> list[int]:
    safe = values.clone()
    safe[torch.isnan(safe)] = -1e9
    k = min(top_k, safe.shape[0])
    if k <= 0:
        return []
    _, indices = torch.topk(safe, k=k, largest=True)
    return [int(i) for i in indices.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-abstract", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="BAAI/bge-m3")
    parser.add_argument("--device", default=None)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-k-main", type=int, default=30)
    parser.add_argument("--candidate-k-scene", type=int, default=20)
    parser.add_argument("--candidate-k-comp", type=int, default=20)
    parser.add_argument("--candidate-k-surface", type=int, default=10)
    args = parser.parse_args()

    device = choose_device(args.device)
    print(f"device={device}")
    if device != "cuda":
        print("CUDA not available; running on CPU.")

    test_records = load_json_array(args.test)
    if args.limit is not None:
        test_records = test_records[: args.limit]
    ref_records = load_json_array(args.reference)
    ref_abstract_records = load_json_array(args.reference_abstract) if args.reference_abstract else None

    embedder = HFEmbedder(args.model_name, device=device, max_length=args.max_length)

    print("Precomputing reference embeddings...")
    ref_field_store = precompute_field_embeddings(ref_records, FIELD_ORDER, embedder, args.embed_batch_size)
    abs_field_store = None
    if ref_abstract_records is not None:
        print("Precomputing abstract reference embeddings...")
        abs_field_store = precompute_field_embeddings(
            ref_abstract_records, FIELD_ORDER, embedder, args.embed_batch_size
        )

    results: list[dict[str, Any]] = []

    for idx, test_record in enumerate(test_records, start=1):
        use_abstract = is_abstract_record(test_record) and ref_abstract_records is not None and abs_field_store is not None
        active_refs = ref_abstract_records if use_abstract else ref_records
        active_store = abs_field_store if use_abstract else ref_field_store
        assert active_store is not None

        query_texts = {field: join_field_text(test_record.get(field, [])) for field in FIELD_ORDER}
        query_embeddings: dict[str, torch.Tensor | None] = {}
        for field, text in query_texts.items():
            if text:
                query_embeddings[field] = embedder.encode([text], batch_size=1)
            else:
                query_embeddings[field] = None

        candidate_indices: set[int] = set()
        recall_plan = [
            ("main_subjects", args.candidate_k_main),
            ("scene_or_setting", args.candidate_k_scene),
            ("composition_or_layout", args.candidate_k_comp),
            ("material_or_surface", args.candidate_k_surface),
        ]
        for field, topk in recall_plan:
            scores = compute_field_similarity(
                query_texts[field],
                query_embeddings[field],
                active_store[field]["embeddings"],
                active_store[field]["texts"],
            )
            candidate_indices.update(topk_indices(scores, topk))

        if not candidate_indices:
            candidate_indices = set(range(len(active_refs)))

        weights = LAYOUT_HEAVY_WEIGHTS if has_strong_layout_requirement(test_record) else DEFAULT_WEIGHTS
        ranked: list[dict[str, Any]] = []

        for ref_idx in sorted(candidate_indices):
            ref_record = active_refs[ref_idx]
            field_scores: dict[str, float | None] = {}
            active_fields: list[str] = []

            for field in FIELD_ORDER:
                score_tensor = compute_field_similarity(
                    query_texts[field],
                    query_embeddings[field],
                    active_store[field]["embeddings"][ref_idx : ref_idx + 1],
                    [active_store[field]["texts"][ref_idx]],
                )
                score = float(score_tensor[0].item()) if not torch.isnan(score_tensor[0]) else None
                if score is not None and query_texts[field]:
                    active_fields.append(field)
                field_scores[field] = score

            norm_weights = renormalize_weights(weights, active_fields)
            final_score = 0.0
            for field in active_fields:
                value = field_scores[field]
                assert value is not None
                # cosine is in [-1, 1], map to [0, 1] for easier reading
                normalized = (value + 1.0) / 2.0
                field_scores[field] = round(normalized, 4)
                final_score += norm_weights[field] * normalized

            ranked.append(
                {
                    "reference_id": ref_record["id"],
                    "reference_caption": ref_record.get("original_caption", ""),
                    "final_score": round(final_score, 4),
                    "field_scores": field_scores,
                }
            )

        ranked.sort(key=lambda x: x["final_score"], reverse=True)
        top_items = ranked[: args.top_k]
        for rank, item in enumerate(top_items, start=1):
            item["rank"] = rank

        results.append(
            {
                "sample_id": test_record.get("sample_id", test_record.get("id")),
                "caption": test_record.get("original_caption", ""),
                "used_reference_pool": "abstract" if use_abstract else "normal",
                f"top{args.top_k}": top_items,
            }
        )
        print(f"[{idx}/{len(test_records)}] sample={results[-1]['sample_id']} candidates={len(candidate_indices)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {len(results)} retrieval results to {args.output}")


if __name__ == "__main__":
    main()
