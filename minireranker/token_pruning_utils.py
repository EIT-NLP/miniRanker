import math
from typing import Any, List, Optional, Sequence


def has_video(row: dict) -> bool:
    return bool(row.get("query_video_frame_paths")) or bool(row.get("document_video_frame_paths"))


def normalize_modalities(value: Any) -> List[str]:
    if value is None:
        return []
    return [str(item) for item in value]


def normalize_token_counts(value: Any) -> List[int]:
    if value is None:
        return []
    return [int(item) for item in value]


def normalize_ranked_tokens(value: Any) -> List[List[int]]:
    if value is None:
        return []
    ranked = []
    for item in value:
        if item is None or len(item) != 2:
            continue
        ranked.append([int(item[0]), int(item[1])])
    return ranked


def resolve_keep_count(total_tokens: int, prune_k: Optional[float]) -> int:
    if total_tokens <= 0:
        return 0
    if prune_k is None:
        return total_tokens
    prune_k = float(prune_k)
    if prune_k <= 0:
        raise ValueError(f"prune_k must be positive, got {prune_k}.")
    if prune_k <= 1:
        keep_count = int(math.ceil(total_tokens * prune_k))
    else:
        keep_count = int(prune_k)
    return max(1, min(total_tokens, keep_count))


def select_keep_indices_by_rank(
    ranked_tokens: Any,
    token_counts: Any,
    prune_k: Optional[float],
) -> List[List[int]]:
    counts = normalize_token_counts(token_counts)
    total_tokens = sum(counts)
    if total_tokens <= 0:
        return []

    keep_count = resolve_keep_count(total_tokens, prune_k)
    ranked = normalize_ranked_tokens(ranked_tokens)
    selected = [set() for _ in counts]

    for media_idx, token_idx in ranked:
        if sum(len(media) for media in selected) >= keep_count:
            break
        if media_idx < 0 or media_idx >= len(counts):
            continue
        if token_idx < 0 or token_idx >= counts[media_idx]:
            continue
        selected[media_idx].add(token_idx)
        if sum(len(media) for media in selected) >= keep_count:
            break

    for media_idx, count in enumerate(counts):
        if count <= 0 or selected[media_idx]:
            continue
        fallback = None
        for ranked_media_idx, ranked_token_idx in ranked:
            if ranked_media_idx == media_idx and 0 <= ranked_token_idx < count:
                fallback = ranked_token_idx
                break
        selected[media_idx].add(0 if fallback is None else fallback)

    return [sorted(media) for media in selected]


def full_keep_indices(token_counts: Sequence[int]) -> List[List[int]]:
    return [list(range(int(count))) for count in token_counts]
