from typing import Any, Dict, List, Optional
import torch
from transformers import AutoProcessor
from .rerank_utils import build_rerank_attention_mask_indices
from .token_pruning_utils import normalize_modalities, normalize_token_counts, select_keep_indices_by_rank

def compact_processor_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in kwargs.items() if value is not None}

def _has_side_visual(example: Dict[str, Any], side: str) -> bool:
    return (
        example.get(f"{side}_image_bytes") is not None
        or bool(example.get(f"{side}_video_frame_paths"))
    )

def _ordered_sides_for_example(example: Dict[str, Any], rerank_format: str) -> List[str]:
    query_has_visual = _has_side_visual(example, "query")
    document_has_visual = _has_side_visual(example, "document")
    if rerank_format == "query_first":
        return ["query", "document"]
    if rerank_format == "document_first":
        return ["document", "query"]
    if rerank_format == "image_first":
        return ["query", "document"] if query_has_visual and not document_has_visual else ["document", "query"]
    if rerank_format == "text_first":
        return ["query", "document"] if document_has_visual else ["document", "query"]
    raise ValueError(f"Unsupported rerank_format: {rerank_format}")

def _side_keep_plan(
    example: Dict[str, Any],
    side: str,
    prune_k: Optional[float],
) -> List[tuple[str, List[int]]]:
    modalities = normalize_modalities(example.get(f"{side}_visual_modalities"))
    token_counts = normalize_token_counts(example.get(f"{side}_visual_token_counts"))
    ranked_tokens = example.get(f"{side}_visual_ranked_tokens")
    if not modalities and not token_counts:
        if _has_side_visual(example, side):
            raise ValueError(
                f"Missing token-pruning rank columns for visual {side} side. "
                "Run precompute_token_ranks.py first."
            )
        return []
    if len(modalities) != len(token_counts):
        raise ValueError(
            f"{side} visual metadata mismatch: "
            f"{len(modalities)} modalities vs {len(token_counts)} token counts."
        )
    keep_entries = select_keep_indices_by_rank(ranked_tokens, token_counts, prune_k)
    if len(keep_entries) != len(modalities):
        raise ValueError(f"{side} keep entries do not match media count.")
    return list(zip(modalities, keep_entries))

def _ordered_media_keep_plan(
    example: Dict[str, Any],
    rerank_format: str,
    prune_k: Optional[float],
) -> Dict[str, List[List[int]]]:
    per_modality = {"image": [], "video": []}
    for side in _ordered_sides_for_example(example, rerank_format):
        for modality, keep_indices in _side_keep_plan(example, side, prune_k):
            if modality not in per_modality:
                raise ValueError(f"Unsupported visual modality in token ranks: {modality}")
            per_modality[modality].append(keep_indices)
    return per_modality

def _grid_token_lengths(grid_thw: Optional[torch.Tensor], spatial_merge_size: int) -> List[int]:
    if grid_thw is None or grid_thw.numel() == 0:
        return []
    grid = grid_thw.to(dtype=torch.long).reshape(-1, 3)
    lengths = grid.prod(-1) // int(spatial_merge_size) ** 2
    return [int(length) for length in lengths.tolist()]

def _split_media_positions_by_sample(
    input_ids: torch.Tensor,
    grid_thw: Optional[torch.Tensor],
    token_id: int,
    spatial_merge_size: int,
) -> List[List[torch.Tensor]]:
    batch_size = input_ids.shape[0]
    if grid_thw is None or grid_thw.numel() == 0:
        return [[] for _ in range(batch_size)]

    split_sizes = _grid_token_lengths(grid_thw, spatial_merge_size)
    split_idx = 0
    all_spans: List[List[torch.Tensor]] = []
    for batch_idx in range(batch_size):
        positions = torch.nonzero(input_ids[batch_idx] == int(token_id), as_tuple=False).squeeze(-1)
        positions = positions.reshape(-1)
        consumed = 0
        spans: List[torch.Tensor] = []
        while consumed < int(positions.numel()):
            if split_idx >= len(split_sizes):
                raise ValueError(f"Token metadata ended early for token_id={token_id}.")
            split_size = int(split_sizes[split_idx])
            spans.append(positions[consumed:consumed + split_size])
            consumed += split_size
            split_idx += 1
        all_spans.append(spans)

    if split_idx != len(split_sizes):
        raise ValueError(f"Unused visual metadata: consumed {split_idx} of {len(split_sizes)} grids.")
    return all_spans

def _sanitize_keep_indices(keep_indices: List[int], upper_bound: int) -> torch.Tensor:
    values = sorted({int(idx) for idx in keep_indices if 0 <= int(idx) < int(upper_bound)})
    if not values and upper_bound > 0:
        values = [0]
    return torch.tensor(values, dtype=torch.long)

def _remap_span(span: tuple[int, int], keep_mask: torch.Tensor) -> tuple[int, int]:
    start, end = int(span[0]), int(span[1])
    prefix = keep_mask.to(torch.long).cumsum(dim=0)
    new_start = int(prefix[start - 1].item()) if start > 0 else 0
    new_end = int(prefix[end - 1].item()) if end > 0 else 0
    return new_start, new_end

def _processor_token_id(processor: AutoProcessor, attr_name: str, token: str) -> int:
    value = getattr(processor, attr_name, None)
    if value is not None:
        return int(value)
    return int(processor.tokenizer.convert_tokens_to_ids(token))

def apply_token_pruning_to_batch(
    batch_full: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    early_segment_spans: List[tuple[int, int]],
    features: List[Dict[str, Any]],
    *,
    prune_k: Optional[float],
    rerank_format: str,
    processor: AutoProcessor,
    k: int,
    k_selection: Optional[str],
) -> Dict[str, Any]:
    input_ids = batch_full["input_ids"]
    attention_mask = batch_full["attention_mask"]
    if prune_k is None:
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "rerank_mask_indices": build_rerank_attention_mask_indices(
                early_segment_spans,
                attention_mask,
                k=k,
                k_selection=k_selection,
            ),
        }

    image_processor = getattr(processor, "image_processor", None)
    spatial_merge_size = int(getattr(image_processor, "merge_size", 2))
    image_token_id = _processor_token_id(processor, "image_token_id", "<|image_pad|>")
    video_token_id = _processor_token_id(processor, "video_token_id", "<|video_pad|>")

    image_spans = _split_media_positions_by_sample(
        input_ids,
        batch_full.get("image_grid_thw"),
        image_token_id,
        spatial_merge_size,
    )
    video_spans = _split_media_positions_by_sample(
        input_ids,
        batch_full.get("video_grid_thw"),
        video_token_id,
        spatial_merge_size,
    )

    row_keep_masks: List[torch.Tensor] = []
    image_keep_indices: List[List[int]] = []
    video_keep_indices: List[List[int]] = []

    for row_idx, example in enumerate(features):
        keep_mask = attention_mask[row_idx].bool().clone()
        media_keep = _ordered_media_keep_plan(example, rerank_format, prune_k)

        if len(media_keep["image"]) != len(image_spans[row_idx]):
            raise ValueError(
                f"Image keep/media count mismatch for sample {row_idx}: "
                f"{len(media_keep['image'])} ranks vs {len(image_spans[row_idx])} placeholders."
            )
        if len(media_keep["video"]) != len(video_spans[row_idx]):
            raise ValueError(
                f"Video keep/media count mismatch for sample {row_idx}: "
                f"{len(media_keep['video'])} ranks vs {len(video_spans[row_idx])} placeholders."
            )

        for span_positions, keep_list in zip(image_spans[row_idx], media_keep["image"]):
            keep_tensor = _sanitize_keep_indices(keep_list, int(span_positions.numel()))
            keep_mask[span_positions] = False
            keep_mask[span_positions[keep_tensor]] = True
            image_keep_indices.append(keep_tensor.tolist())

        for span_positions, keep_list in zip(video_spans[row_idx], media_keep["video"]):
            keep_tensor = _sanitize_keep_indices(keep_list, int(span_positions.numel()))
            keep_mask[span_positions] = False
            keep_mask[span_positions[keep_tensor]] = True
            video_keep_indices.append(keep_tensor.tolist())

        row_keep_masks.append(keep_mask)

    new_lengths = [int(mask.sum().item()) for mask in row_keep_masks]
    max_new_len = max(new_lengths) if new_lengths else input_ids.shape[1]
    tokenizer = processor.tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    new_input_ids = input_ids.new_full((input_ids.shape[0], max_new_len), int(pad_token_id))
    new_attention_mask = attention_mask.new_zeros((input_ids.shape[0], max_new_len))
    new_labels = labels.new_full((input_ids.shape[0], max_new_len), -100)
    new_early_segment_spans: List[tuple[int, int]] = []

    for row_idx, keep_mask in enumerate(row_keep_masks):
        keep_positions = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
        new_len = int(keep_positions.numel())
        new_input_ids[row_idx, :new_len] = input_ids[row_idx, keep_positions]
        new_attention_mask[row_idx, :new_len] = 1
        new_labels[row_idx, :new_len] = labels[row_idx, keep_positions]
        new_early_segment_spans.append(_remap_span(early_segment_spans[row_idx], keep_mask))

    batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels,
        "rerank_mask_indices": build_rerank_attention_mask_indices(
            new_early_segment_spans,
            new_attention_mask,
            k=k,
            k_selection=k_selection,
        ),
    }
    if image_keep_indices:
        batch["image_keep_indices"] = image_keep_indices
    if video_keep_indices:
        batch["video_keep_indices"] = video_keep_indices
    return batch
