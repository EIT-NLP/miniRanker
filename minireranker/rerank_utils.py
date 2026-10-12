import io
import os
from collections.abc import Sequence
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info


DEFAULT_SYSTEM_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
)
ALLOWED_RERANK_FORMATS = {"query_first", "document_first", "image_first", "text_first"}
ALLOWED_K_SELECTIONS = {"last", "step"}
SEGMENT_HEADERS = {
    "query": "<Query>:",
    "document": "<Document>:",
}


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def ensure_pil_or_none(value: Any) -> Optional[Image.Image]:
    if value is None:
        return None

    if isinstance(value, Image.Image):
        return value.convert("RGB")

    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(value)).convert("RGB")

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path") is not None:
            return Image.open(value["path"]).convert("RGB")

    if isinstance(value, str):
        return Image.open(value).convert("RGB")

    return None


def normalize_video_paths(value: Any, video_root: Optional[str], max_frames: int) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        paths = [value] if value.strip() else []
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, memoryview)):
        paths = [item for item in value if item is not None and str(item).strip()]
    else:
        return []

    resolved = []
    for item in paths:
        if isinstance(item, Image.Image):
            resolved.append(item)
            continue
        path = str(item).strip()
        if path.startswith(("http://", "https://")) or os.path.isabs(path):
            resolved.append(path)
        elif video_root is not None:
            resolved.append(os.path.join(video_root, path))
        else:
            resolved.append(path)

    missing_paths = [
        path
        for path in resolved
        if isinstance(path, str)
        and not path.startswith(("http://", "https://"))
        and not os.path.exists(path)
    ]
    if missing_paths:
        raise FileNotFoundError(f"Missing video frame: {missing_paths[0]}")

    if max_frames is not None and max_frames > 0 and len(resolved) > max_frames:
        if max_frames == 1:
            return [resolved[0]]
        last_idx = len(resolved) - 1
        indices = [int(i * last_idx / (max_frames - 1)) for i in range(max_frames)]
        resolved = [resolved[idx] for idx in indices]
    return resolved


def _has_content(value: Any) -> bool:
    return value is not None and (not isinstance(value, list) or len(value) > 0)


def _append_multimodal_block(
    user_content: List[Dict[str, Any]],
    header: str,
    text: str,
    image: Optional[Image.Image],
    video: Optional[List[Any]],
    min_pixels: int,
    image_max_pixels: int,
    video_frame_max_pixels: int,
    video_total_pixels: int,
    fps: int,
) -> None:
    user_content.append({"type": "text", "text": header})

    if not text and image is None and not video:
        user_content.append({"type": "text", "text": "NULL"})
        return

    if video:
        user_content.append(
            {
                "type": "video",
                "video": video,
                "min_pixels": min_pixels,
                "max_pixels": video_frame_max_pixels,
                "total_pixels": video_total_pixels,
                "fps": fps,
                "sample_fps": fps,
            }
        )

    if image is not None:
        user_content.append(
            {
                "type": "image",
                "image": image,
                "min_pixels": min_pixels,
                "max_pixels": image_max_pixels,
            }
        )

    if text:
        user_content.append({"type": "text", "text": text})


def _get_prompt_user_content(messages: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_messages = messages[:-1] if messages and messages[-1].get("role") == "assistant" else messages
    if len(prompt_messages) < 2 or prompt_messages[1].get("role") != "user":
        raise ValueError("Expected prompt messages with a user turn when building segment span views.")

    return prompt_messages, prompt_messages[1].get("content", [])


def _clone_messages_with_user_content(
    prompt_messages: List[Dict[str, Any]],
    content_slice: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    cloned_messages = [dict(message) for message in prompt_messages]
    cloned_messages[1] = dict(cloned_messages[1])
    cloned_messages[1]["content"] = list(content_slice)
    return cloned_messages


def _find_segment_bounds(
    user_content: List[Dict[str, Any]],
    segment_name: str,
) -> tuple[int, int]:
    if segment_name not in SEGMENT_HEADERS:
        raise ValueError(f"segment_name must be one of {sorted(SEGMENT_HEADERS)}, got {segment_name}")

    segment_start_idx = None
    segment_end_idx = None
    target_header = SEGMENT_HEADERS[segment_name]
    all_headers = tuple(SEGMENT_HEADERS.values())

    for idx, item in enumerate(user_content):
        if item.get("type") != "text":
            continue
        text = item.get("text", "").lstrip()
        if segment_start_idx is None and text.startswith(target_header):
            segment_start_idx = idx
            continue
        if segment_start_idx is not None and text.startswith(all_headers):
            segment_end_idx = idx
            break

    if segment_start_idx is None:
        raise ValueError(f"Unable to locate {target_header} header when building segment span views.")
    if segment_end_idx is None:
        segment_end_idx = len(user_content)

    return segment_start_idx, segment_end_idx


def build_segment_span_message_views(
    messages: List[Dict[str, Any]],
    segment_name: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_messages, user_content = _get_prompt_user_content(messages)
    segment_start_idx, segment_end_idx = _find_segment_bounds(user_content, segment_name)
    segment_content_start_idx = min(segment_start_idx + 1, len(user_content))
    return (
        _clone_messages_with_user_content(prompt_messages, user_content[:segment_content_start_idx]),
        _clone_messages_with_user_content(prompt_messages, user_content[:segment_end_idx]),
    )


def build_early_segment_span_message_views(
    messages: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_messages, user_content = _get_prompt_user_content(messages)
    first_segment_name = None
    first_segment_start_idx = None

    for idx, item in enumerate(user_content):
        if item.get("type") != "text":
            continue
        text = item.get("text", "").lstrip()
        for segment_name, header in SEGMENT_HEADERS.items():
            if text.startswith(header):
                first_segment_name = segment_name
                first_segment_start_idx = idx
                break
        if first_segment_name is not None:
            break

    if first_segment_name is None or first_segment_start_idx is None:
        raise ValueError("Unable to locate query/document headers when building early segment span views.")

    _, segment_end_idx = _find_segment_bounds(user_content, first_segment_name)
    segment_content_start_idx = min(first_segment_start_idx + 1, len(user_content))
    return (
        first_segment_name,
        _clone_messages_with_user_content(prompt_messages, user_content[:segment_content_start_idx]),
        _clone_messages_with_user_content(prompt_messages, user_content[:segment_end_idx]),
    )


def build_processor_vision_kwargs(messages, processor) -> Dict[str, Any]:
    images, videos, video_kwargs = process_vision_info(
        messages,
        return_video_kwargs=True,
        return_video_metadata=True,
        image_patch_size=processor.image_processor.patch_size,
    )

    video_metadata = None
    if videos:
        videos, video_metadata = zip(*videos)
        videos = list(videos)
        video_metadata = list(video_metadata)
    else:
        videos = None

    kwargs = {
        "images": images or None,
        "videos": videos,
        "video_metadata": video_metadata,
    }
    kwargs.update(video_kwargs)
    return kwargs


def _count_visible_tokens(
    processor,
    view_messages: List[Dict[str, Any]],
) -> int:
    text = processor.apply_chat_template(
        view_messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=False,
    )
    encoded = processor(
        text=[text],
        padding=True,
        truncation=False,
        return_tensors="pt",
        do_resize=False,
        **{
            key: value
            for key, value in build_processor_vision_kwargs(view_messages, processor).items()
            if value is not None
        },
    )

    token_count = int(encoded["attention_mask"][0].sum().item())
    final_message_suffix = "<|im_end|>\n"
    if text.endswith(final_message_suffix):
        suffix_token_count = len(
            processor.tokenizer(
                final_message_suffix,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]
        )
        token_count -= suffix_token_count
    return token_count


def compute_segment_span_from_messages(
    processor,
    messages: List[Dict[str, Any]],
    segment_name: str,
) -> tuple[int, int]:
    before_segment_messages, through_segment_messages = build_segment_span_message_views(messages, segment_name)
    segment_start = _count_visible_tokens(processor, before_segment_messages)
    segment_end_exclusive = _count_visible_tokens(processor, through_segment_messages)
    if segment_end_exclusive <= segment_start:
        raise ValueError(
            "Computed invalid segment span: "
            f"segment={segment_name}, start={segment_start}, end={segment_end_exclusive}"
        )
    return segment_start, segment_end_exclusive


def compute_document_span_from_messages(
    processor,
    messages: List[Dict[str, Any]],
) -> tuple[int, int]:
    return compute_segment_span_from_messages(processor, messages, "document")


def compute_early_segment_span_from_messages(
    processor,
    messages: List[Dict[str, Any]],
) -> tuple[tuple[int, int], str]:
    segment_name, before_segment_messages, through_segment_messages = build_early_segment_span_message_views(messages)
    segment_start = _count_visible_tokens(processor, before_segment_messages)
    segment_end_exclusive = _count_visible_tokens(processor, through_segment_messages)
    if segment_end_exclusive <= segment_start:
        raise ValueError(
            "Computed invalid early segment span: "
            f"segment={segment_name}, start={segment_start}, end={segment_end_exclusive}"
        )
    return (segment_start, segment_end_exclusive), segment_name


def compute_segment_mask_indices(
    segment_span: tuple[int, int],
    k: int,
    k_selection: Optional[str],
) -> tuple[Optional[List[int]], Optional[List[int]]]:
    if k_selection is None or k < 0:
        return None, None
    if k_selection not in ALLOWED_K_SELECTIONS:
        raise ValueError(f"k_selection must be one of {sorted(ALLOWED_K_SELECTIONS)}, got {k_selection}")

    start, end = int(segment_span[0]), int(segment_span[1])
    if end <= start:
        return None, None

    if k_selection == "last":
        if end - k > start:
            keep_start = max(end - k, start)
            keep_indices = list(range(keep_start, end))
            mask_indices = list(range(start, keep_start))
            return keep_indices, mask_indices
        return None, None

    if k <= 0:
        raise ValueError("k must be > 0 when k_selection='step'")

    if end - start > k:
        keep_indices = list(range(start + k - 1, end, k))
        if keep_indices[-1] != end - 1:
            keep_indices.append(end - 1)
        keep_set = set(keep_indices)
        mask_indices = [idx for idx in range(start, end) if idx not in keep_set]
        return keep_indices, mask_indices
    if end - start > 1:
        keep_indices = [end - 1]
        mask_indices = list(range(start, end - 1))
        return keep_indices, mask_indices
    return None, None


def compute_document_mask_indices(
    document_span: tuple[int, int],
    k: int,
    k_selection: Optional[str],
) -> tuple[Optional[List[int]], Optional[List[int]]]:
    return compute_segment_mask_indices(document_span, k, k_selection)


def build_rerank_attention_mask_indices(
    early_segment_spans: List[tuple[int, int]],
    attention_mask: torch.Tensor,
    k: int = -1,
    k_selection: Optional[str] = None,
) -> Dict[str, Dict[str, torch.Tensor]]:
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must have shape [batch, seq_len], got {tuple(attention_mask.shape)}")
    if len(early_segment_spans) != attention_mask.shape[0]:
        raise ValueError(
            "early_segment_spans length must match batch size, got "
            f"{len(early_segment_spans)} vs {attention_mask.shape[0]}"
        )

    batch_size, seq_len = attention_mask.shape
    valid_mask = attention_mask.bool()
    full_q = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=attention_mask.device)
    full_k = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=attention_mask.device)
    selective_q = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=attention_mask.device)
    selective_k = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=attention_mask.device)

    for sample_idx, segment_span in enumerate(early_segment_spans):
        start, end = int(segment_span[0]), int(segment_span[1])
        valid_length = int(valid_mask[sample_idx].sum().item())
        if start < 0 or end < start or end > valid_length:
            raise ValueError(
                "Invalid early segment span for attention mask indices: "
                f"sample={sample_idx}, span=[{start}, {end}), valid_length={valid_length}"
            )
        if end == start or end >= valid_length:
            continue

        full_q[sample_idx, end:valid_length] = valid_mask[sample_idx, end:valid_length]
        full_k[sample_idx, start:end] = True

        _, mask_indices = compute_segment_mask_indices(segment_span, k, k_selection)
        if not mask_indices:
            continue
        selective_q[sample_idx, end:valid_length] = valid_mask[sample_idx, end:valid_length]
        selective_k[sample_idx, mask_indices] = True

    return {
        "full": {"q": full_q, "k": full_k},
        "selective": {"q": selective_q, "k": selective_k},
    }


def build_rerank_messages(
    instruction: Any,
    query_text: Any = "",
    document_text: Any = "",
    query_image: Any = None,
    document_image: Any = None,
    query_video: Any = None,
    document_video: Any = None,
    video_root: Optional[str] = None,
    system_prompt: Optional[str] = None,
    label: Optional[Any] = None,
    rerank_format: str = "query_first",
    max_frames: int = 64,
    min_pixels: int = 4 * 32 * 32,
    image_max_pixels: int = 1800 * 32 * 32,
    video_frame_max_pixels: int = 768 * 32 * 32,
    video_total_pixels: int = 10 * 768 * 32 * 32,
    fps: int = 1,
) -> List[Dict[str, Any]]:
    if rerank_format not in ALLOWED_RERANK_FORMATS:
        raise ValueError(f"rerank_format must be one of {sorted(ALLOWED_RERANK_FORMATS)}, got {rerank_format}")

    instruction = normalize_text(instruction)
    query_text = normalize_text(query_text)
    document_text = normalize_text(document_text)
    label = None if label is None else normalize_text(label).lower()

    query_image = ensure_pil_or_none(query_image)
    document_image = ensure_pil_or_none(document_image)
    query_video = normalize_video_paths(query_video, video_root, max_frames)
    document_video = normalize_video_paths(document_video, video_root, max_frames)

    query_has_visual = query_image is not None or _has_content(query_video)
    document_has_visual = document_image is not None or _has_content(document_video)

    query_content: List[Dict[str, Any]] = []
    document_content: List[Dict[str, Any]] = []
    _append_multimodal_block(
        query_content,
        header="\n<Query>:",
        text=query_text,
        image=query_image,
        video=query_video,
        min_pixels=min_pixels,
        image_max_pixels=image_max_pixels,
        video_frame_max_pixels=video_frame_max_pixels,
        video_total_pixels=video_total_pixels,
        fps=fps,
    )
    _append_multimodal_block(
        document_content,
        header="\n<Document>:",
        text=document_text,
        image=document_image,
        video=document_video,
        min_pixels=min_pixels,
        image_max_pixels=image_max_pixels,
        video_frame_max_pixels=video_frame_max_pixels,
        video_total_pixels=video_total_pixels,
        fps=fps,
    )

    user_content: List[Dict[str, Any]] = [{"type": "text", "text": f"<Instruct>: {instruction}"}]
    if rerank_format == "query_first":
        user_content.extend(query_content)
        user_content.extend(document_content)
    elif rerank_format == "document_first":
        user_content.extend(document_content)
        user_content.extend(query_content)
    elif rerank_format == "image_first":
        if query_has_visual and not document_has_visual:
            user_content.extend(query_content)
            user_content.extend(document_content)
        else:
            user_content.extend(document_content)
            user_content.extend(query_content)
    elif rerank_format == "text_first":
        if document_has_visual:
            user_content.extend(query_content)
            user_content.extend(document_content)
        else:
            user_content.extend(document_content)
            user_content.extend(query_content)

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt or DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": label}]})
    return messages


def build_rerank_messages_from_example(
    example: Dict[str, Any],
    system_prompt: Optional[str] = None,
    rerank_format: str = "query_first",
    video_root: Optional[str] = None,
    max_frames: int = 64,
    min_pixels: int = 4 * 32 * 32,
    image_max_pixels: int = 1800 * 32 * 32,
    video_frame_max_pixels: int = 768 * 32 * 32,
    video_total_pixels: int = 10 * 768 * 32 * 32,
    fps: int = 1,
    include_label: bool = True,
) -> List[Dict[str, Any]]:
    return build_rerank_messages(
        instruction=example.get("instruction", ""),
        query_text=example.get("query_text", ""),
        document_text=example.get("document_text", ""),
        query_image=example.get("query_image_bytes"),
        document_image=example.get("document_image_bytes"),
        query_video=example.get("query_video_frame_paths"),
        document_video=example.get("document_video_frame_paths"),
        video_root=video_root,
        system_prompt=system_prompt,
        label=example.get("label", "") if include_label else None,
        rerank_format=rerank_format,
        max_frames=max_frames,
        min_pixels=min_pixels,
        image_max_pixels=image_max_pixels,
        video_frame_max_pixels=video_frame_max_pixels,
        video_total_pixels=video_total_pixels,
        fps=fps,
    )
