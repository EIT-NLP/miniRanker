# Custom extensions to Qwen3-VL in Hugging Face Transformers.
# Copyright 2025 The Qwen Team and The HuggingFace Inc. team.
# Licensed under the Apache License, Version 2.0; see LICENSE.

from typing import Any, Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from transformers import AutoConfig
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLTextAttention,
    Qwen3VLTextConfig,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import TransformersKwargs, is_torchdynamo_compiling
from transformers.utils.generic import check_model_inputs
from typing_extensions import Unpack

_UNSET = object()
MASK_NONE = "none"
MASK_FULL = "full"
MASK_SELECTIVE = "selective"
ALLOWED_K_SELECTIONS = {"last", "step"}


def _as_long_tensor(value: Any, device: Optional[torch.device] = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    tensor = tensor.to(dtype=torch.long)
    if device is not None:
        tensor = tensor.to(device)
    return tensor.reshape(-1)


def _safe_sorted_unique(indices: torch.Tensor, upper_bound: int) -> torch.Tensor:
    if indices.numel() == 0:
        return indices.to(dtype=torch.long)
    indices = indices.to(dtype=torch.long).reshape(-1)
    indices = indices[(indices >= 0) & (indices < int(upper_bound))]
    if indices.numel() == 0 and upper_bound > 0:
        return torch.zeros(1, dtype=torch.long, device=indices.device)
    return torch.unique(indices, sorted=True)


def _normalize_keep_entries(
    keep_indices: Any,
    num_items: int,
    device: torch.device,
) -> List[Optional[torch.Tensor]]:
    if num_items <= 0:
        return []
    if keep_indices is None:
        return [None] * num_items
    if isinstance(keep_indices, torch.Tensor):
        if keep_indices.ndim == 1:
            if num_items != 1:
                raise ValueError(f"Expected {num_items} keep-index entries, got a 1D tensor.")
            return [_as_long_tensor(keep_indices, device=device)]
        if keep_indices.ndim == 2:
            if keep_indices.shape[0] != num_items:
                raise ValueError(f"Expected {num_items} keep-index rows, got {keep_indices.shape[0]}.")
            return [_as_long_tensor(row, device=device) for row in keep_indices]
        raise ValueError(f"Unsupported keep-index tensor shape: {tuple(keep_indices.shape)}.")
    if isinstance(keep_indices, (list, tuple)):
        if num_items == 1 and (len(keep_indices) == 0 or not isinstance(keep_indices[0], (list, tuple, torch.Tensor))):
            return [_as_long_tensor(keep_indices, device=device)]
        if len(keep_indices) != num_items:
            raise ValueError(f"Expected {num_items} keep-index entries, got {len(keep_indices)}.")
        return [_as_long_tensor(row, device=device) for row in keep_indices]
    raise TypeError(f"Unsupported keep-index type: {type(keep_indices)}")


def _media_grid_lengths(grid_thw: Optional[torch.Tensor], spatial_merge_size: int) -> List[int]:
    if grid_thw is None or grid_thw.numel() == 0:
        return []
    grid = grid_thw.detach().to(dtype=torch.long).reshape(-1, 3)
    lengths = grid.prod(-1) // int(spatial_merge_size) ** 2
    return [int(length) for length in lengths.tolist()]


def _prune_tensor_by_keep(tensor: torch.Tensor, keep_indices: Optional[torch.Tensor]) -> torch.Tensor:
    if keep_indices is None:
        return tensor
    keep_indices = _safe_sorted_unique(keep_indices, tensor.shape[0]).to(tensor.device)
    return tensor.index_select(0, keep_indices)


def _prune_flat_tensor_by_lengths(
    tensor: torch.Tensor,
    keep_entries: Sequence[Optional[torch.Tensor]],
    original_lengths: Sequence[int],
) -> torch.Tensor:
    if not original_lengths or int(sum(original_lengths)) != int(tensor.shape[0]):
        if len(keep_entries) == 1:
            return _prune_tensor_by_keep(tensor, keep_entries[0])
        return tensor

    pieces = []
    cursor = 0
    for length, keep_indices in zip(original_lengths, keep_entries):
        length = int(length)
        pieces.append(_prune_tensor_by_keep(tensor[cursor:cursor + length], keep_indices))
        cursor += length
    if not pieces:
        return tensor.new_zeros((0, *tensor.shape[1:]))
    return torch.cat(pieces, dim=0)


def _is_tensor_sequence(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and all(isinstance(item, torch.Tensor) for item in value)
    )


def _prune_visual_output(
    output: Any,
    keep_entries: Sequence[Optional[torch.Tensor]],
    original_lengths: Sequence[int],
) -> Any:
    if output is None:
        return None
    if isinstance(output, torch.Tensor):
        return _prune_flat_tensor_by_lengths(output, keep_entries, original_lengths)
    if _is_tensor_sequence(output):
        is_media_sequence = len(output) == len(keep_entries)
        if is_media_sequence and original_lengths:
            is_media_sequence = all(
                int(tensor.shape[0]) == int(length)
                for tensor, length in zip(output, original_lengths)
            )
        if is_media_sequence:
            pruned = [
                _prune_tensor_by_keep(tensor, keep_indices)
                for tensor, keep_indices in zip(output, keep_entries)
            ]
        else:
            pruned = [
                _prune_flat_tensor_by_lengths(tensor, keep_entries, original_lengths)
                for tensor in output
            ]
        return type(output)(pruned)
    if isinstance(output, tuple):
        return tuple(_prune_visual_output(item, keep_entries, original_lengths) for item in output)
    if isinstance(output, list):
        return [_prune_visual_output(item, keep_entries, original_lengths) for item in output]
    if isinstance(output, dict):
        return {key: _prune_visual_output(value, keep_entries, original_lengths) for key, value in output.items()}
    return output


def _grid_keep_coords(
    grid_thw: Sequence[int],
    keep_indices: Optional[torch.Tensor],
    spatial_merge_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t, h, w = [int(x) for x in grid_thw]
    rows = h // int(spatial_merge_size)
    cols = w // int(spatial_merge_size)
    total = int(t) * rows * cols
    if keep_indices is None:
        keep_indices = torch.arange(total, dtype=torch.long, device=device)
    else:
        keep_indices = _safe_sorted_unique(keep_indices, total).to(device)
    frame_area = rows * cols
    frame_idx = keep_indices // frame_area
    rem = keep_indices % frame_area
    h_idx = rem // cols
    w_idx = rem % cols
    return torch.stack([frame_idx, h_idx, w_idx], dim=0).to(dtype=dtype)


def _expand_video_keep_indices_by_frame(
    video_grid_thw: Optional[torch.Tensor],
    video_keep_indices: Any,
    spatial_merge_size: int,
    device: torch.device,
) -> List[Optional[torch.Tensor]]:
    if video_grid_thw is None or video_grid_thw.numel() == 0:
        return []
    original_grids = video_grid_thw.detach().to(dtype=torch.long).reshape(-1, 3)
    keep_entries = _normalize_keep_entries(video_keep_indices, original_grids.shape[0], device)
    frame_entries: List[Optional[torch.Tensor]] = []
    for grid, keep_indices in zip(original_grids, keep_entries):
        t, h, w = [int(x) for x in grid.tolist()]
        frame_area = (h // int(spatial_merge_size)) * (w // int(spatial_merge_size))
        if keep_indices is None:
            for _ in range(t):
                frame_entries.append(None)
            continue
        keep_indices = _safe_sorted_unique(keep_indices, t * frame_area).to(device)
        for frame_idx in range(t):
            start = frame_idx * frame_area
            end = start + frame_area
            frame_keep = keep_indices[(keep_indices >= start) & (keep_indices < end)] - start
            if frame_keep.numel() == 0 and frame_area > 0:
                frame_keep = torch.zeros(1, dtype=torch.long, device=device)
            frame_entries.append(frame_keep)
    return frame_entries


def _expand_mask_to_batch(mask: torch.Tensor, batch_size: int) -> Optional[torch.Tensor]:
    if mask is None:
        return None

    if mask.shape[0] == batch_size:
        return mask

    if mask.shape[0] == 0:
        return None

    if batch_size % mask.shape[0] == 0:
        repeat_factor = batch_size // mask.shape[0]
        return mask.repeat_interleave(repeat_factor, dim=0)

    return mask[:batch_size]


def _normalize_rerank_mask_indices(rerank_mask_indices):
    if rerank_mask_indices is None:
        return None

    if not isinstance(rerank_mask_indices, dict):
        raise ValueError("rerank_mask_indices must be a dict with full/selective mask entries.")

    normalized = {}
    for mask_mode in (MASK_FULL, MASK_SELECTIVE):
        mask_spec = rerank_mask_indices.get(mask_mode)
        if mask_spec is None:
            continue
        if not isinstance(mask_spec, dict):
            raise ValueError(f"rerank_mask_indices[{mask_mode!r}] must be a dict with q/k tensors.")
        q_mask = mask_spec.get("q")
        k_mask = mask_spec.get("k")
        if q_mask is None or k_mask is None:
            continue
        if not isinstance(q_mask, torch.Tensor) or not isinstance(k_mask, torch.Tensor):
            raise ValueError(f"rerank_mask_indices[{mask_mode!r}] q/k entries must be tensors.")
        if q_mask.ndim != 2 or k_mask.ndim != 2:
            raise ValueError(
                f"rerank_mask_indices[{mask_mode!r}] q/k tensors must have shape [batch, seq_len], "
                f"got q={tuple(q_mask.shape)}, k={tuple(k_mask.shape)}"
            )
        if q_mask.shape[0] != k_mask.shape[0]:
            raise ValueError(
                f"rerank_mask_indices[{mask_mode!r}] q/k batch sizes differ: "
                f"{q_mask.shape[0]} != {k_mask.shape[0]}"
            )
        normalized[mask_mode] = {
            "q": q_mask.bool(),
            "k": k_mask.bool(),
        }
    return normalized


def _resolve_rerank_mask_mode(
    layer_idx: int,
    no_interaction_layer: int,
    early_exit_layer: int,
    k: int,
    k_selection: Optional[str],
) -> str:
    if layer_idx > early_exit_layer:
        return MASK_NONE
    if layer_idx <= no_interaction_layer:
        return MASK_FULL
    if k >= 0 and k_selection is not None:
        return MASK_SELECTIVE
    return MASK_NONE


def _get_mask_pair(rerank_mask_indices, mask_mode: str, batch_size: int, device: torch.device):
    if rerank_mask_indices is None or mask_mode == MASK_NONE:
        return None, None

    mask_spec = rerank_mask_indices.get(mask_mode)
    if mask_spec is None:
        return None, None

    q_mask = _expand_mask_to_batch(mask_spec.get("q"), batch_size)
    k_mask = _expand_mask_to_batch(mask_spec.get("k"), batch_size)
    if q_mask is None or k_mask is None:
        return None, None
    return q_mask.to(device=device, dtype=torch.bool), k_mask.to(device=device, dtype=torch.bool)


def eager_attention_forward_last_k_vis(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    rerank_mask_indices=None,
    rerank_mask_mode: str = MASK_NONE,
    **kwargs: Unpack[TransformersKwargs],
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if rerank_mask_indices is not None and rerank_mask_mode != MASK_NONE:
        batch_size = attn_weights.shape[0]
        query_mask, key_mask = _get_mask_pair(
            rerank_mask_indices=rerank_mask_indices,
            mask_mode=rerank_mask_mode,
            batch_size=batch_size,
            device=query.device,
        )
        if query_mask is not None and key_mask is not None:
            query_length = attn_weights.shape[-2]
            key_length = attn_weights.shape[-1]
            # Mask indices are built for the prefill sequence. Decode steps have
            # q_len=1 and are intentionally left untouched for this reranking use case.
            if query_mask.shape[-1] == query_length and key_mask.shape[-1] >= key_length:
                key_mask = key_mask[:, :key_length]
                mask_value = torch.finfo(attn_weights.dtype).min
                for sample_idx in range(batch_size):
                    query_indices = torch.nonzero(query_mask[sample_idx], as_tuple=False).flatten()
                    key_indices = torch.nonzero(key_mask[sample_idx], as_tuple=False).flatten()
                    if query_indices.numel() == 0 or key_indices.numel() == 0:
                        continue
                    attn_weights[sample_idx, :, query_indices[:, None], key_indices[None, :]] = mask_value
            elif query_mask.shape[-1] != query_length:
                pass
            else:
                if key_mask.shape[-1] != 0:
                    raise ValueError(
                        "rerank key mask is shorter than attention key length: "
                        f"{key_mask.shape[-1]} < {key_length}"
                    )

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class CustomQwen3VLTextAttention(Qwen3VLTextAttention):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int) -> None:
        super().__init__(config, layer_idx)
        self.rerank_mask_indices = None
        self.k = -1
        self.k_selection = None
        self.no_interaction_layer = -1
        self.early_exit_layer = config.num_hidden_layers - 1

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward_last_k_vis
        rerank_mask_mode = _resolve_rerank_mask_mode(
            layer_idx=self.layer_idx,
            no_interaction_layer=self.no_interaction_layer,
            early_exit_layer=self.early_exit_layer,
            k=self.k,
            k_selection=self.k_selection,
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            rerank_mask_indices=self.rerank_mask_indices,
            rerank_mask_mode=rerank_mask_mode,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class CustomQwen3VLTextDecoderLayer(Qwen3VLTextDecoderLayer):
    def __init__(self, config: Qwen3VLTextConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = CustomQwen3VLTextAttention(config=config, layer_idx=layer_idx)


class CustomQwen3VLTextModel(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [CustomQwen3VLTextDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.early_exit_layer = config.num_hidden_layers - 1

    def _resolve_early_exit_layer(self, early_exit_layer: Optional[int]) -> int:
        if early_exit_layer is None:
            early_exit_layer = self.config.num_hidden_layers - 1

        if early_exit_layer < 0 or early_exit_layer >= self.config.num_hidden_layers:
            raise ValueError(
                f"early_exit_layer must be in [0, {self.config.num_hidden_layers - 1}], got {early_exit_layer}"
            )

        return int(early_exit_layer)

    def set_early_exit_layer(self, early_exit_layer: Optional[int]) -> None:
        self.early_exit_layer = self._resolve_early_exit_layer(early_exit_layer)

    @check_model_inputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        early_exit_layer: Optional[int] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        target_layer = self._resolve_early_exit_layer(
            early_exit_layer if early_exit_layer is not None else self.early_exit_layer
        )

        if use_cache and past_key_values is None and not torch.jit.is_tracing():
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(3, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = position_ids[0]

        attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

            if layer_idx == target_layer:
                break

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Custom_Qwen3_VLModel(Qwen3VLModel):
    def __init__(self, config):
        super().__init__(config)
        self.language_model = CustomQwen3VLTextModel._from_config(config.text_config)
        self.rerank_mask_indices = None
        self.k = -1
        self.k_selection = None
        self.early_exit_layer = config.text_config.num_hidden_layers - 1
        self.no_interaction_layer = -1
        early_exit_layer = getattr(config, "early_exit_layer", None)
        no_interaction_layer = getattr(config, "no_interaction_layer", getattr(config, "full_document_mask_layer", None))
        k = getattr(config, "k", getattr(config, "last_k", -1))
        k_selection = getattr(config, "k_selection", None)
        if k_selection is None and getattr(config, "last_k", -1) >= 0:
            k_selection = "last"
        self.set_early_exit_layer(early_exit_layer)
        self.set_no_interaction_layer(no_interaction_layer)
        self.set_k_config(k, k_selection)

    def _resolve_early_exit_layer(self, early_exit_layer: Optional[int]) -> int:
        num_hidden_layers = self.config.text_config.num_hidden_layers
        if early_exit_layer is None:
            early_exit_layer = num_hidden_layers - 1
        if early_exit_layer < 0 or early_exit_layer >= num_hidden_layers:
            raise ValueError(
                f"early_exit_layer must be in [0, {num_hidden_layers - 1}], got {early_exit_layer}"
            )
        return int(early_exit_layer)

    def _resolve_no_interaction_layer(
        self,
        no_interaction_layer: Optional[int],
        early_exit_layer: Optional[int] = None,
    ) -> int:
        num_hidden_layers = self.config.text_config.num_hidden_layers
        resolved_early_exit_layer = self._resolve_early_exit_layer(
            early_exit_layer if early_exit_layer is not None else self.early_exit_layer
        )

        if no_interaction_layer is None:
            return -1
        if no_interaction_layer < -1 or no_interaction_layer >= num_hidden_layers:
            raise ValueError(
                "no_interaction_layer must be in [-1, "
                f"{num_hidden_layers - 1}], got {no_interaction_layer}"
            )
        if no_interaction_layer > resolved_early_exit_layer:
            raise ValueError(
                "no_interaction_layer must be smaller than or equal to early_exit_layer, got "
                f"{no_interaction_layer} > {resolved_early_exit_layer}"
            )
        return int(no_interaction_layer)

    def _resolve_full_document_mask_layer(
        self,
        full_document_mask_layer: Optional[int],
        early_exit_layer: Optional[int] = None,
    ) -> int:
        return self._resolve_no_interaction_layer(full_document_mask_layer, early_exit_layer)

    def _sync_attention_schedule(self) -> None:
        for layer in self.language_model.layers:
            layer.self_attn.k = self.k
            layer.self_attn.k_selection = self.k_selection
            layer.self_attn.no_interaction_layer = self.no_interaction_layer
            layer.self_attn.early_exit_layer = self.early_exit_layer

    def _requires_rerank_mask_indices(self) -> bool:
        return self.no_interaction_layer >= 0 or (self.k >= 0 and self.k_selection is not None)

    def _resolve_k_config(self, k: Optional[int], k_selection: Optional[str]) -> tuple[int, Optional[str]]:
        if k is None:
            k = -1
        k = int(k)
        if k_selection is not None and k_selection not in ALLOWED_K_SELECTIONS:
            raise ValueError(f"k_selection must be one of {sorted(ALLOWED_K_SELECTIONS)}, got {k_selection}")
        if k_selection is None and k != -1:
            raise ValueError(f"k_selection must be set when k is set. Got k={k}, k_selection=None")
        if k_selection is not None and k < 0:
            raise ValueError(f"k must be >= 0 when k_selection is set. Got k={k}, k_selection={k_selection}")
        if k_selection == "step" and k <= 0:
            raise ValueError("k must be > 0 when k_selection='step'")
        return k, k_selection

    def set_k_config(self, k: Optional[int], k_selection: Optional[str]) -> None:
        self.k, self.k_selection = self._resolve_k_config(k, k_selection)
        self.config.k = self.k
        self.config.k_selection = self.k_selection
        if hasattr(self.config, "last_k"):
            delattr(self.config, "last_k")
        self._sync_attention_schedule()

    def set_last_k(self, last_k: int) -> None:
        self.set_k_config(last_k, "last" if last_k is not None and last_k >= 0 else None)

    def set_early_exit_layer(self, early_exit_layer: Optional[int]) -> None:
        resolved_early_exit_layer = self._resolve_early_exit_layer(early_exit_layer)
        self.early_exit_layer = resolved_early_exit_layer
        self.config.early_exit_layer = resolved_early_exit_layer
        self.language_model.set_early_exit_layer(resolved_early_exit_layer)
        if self.no_interaction_layer >= 0:
            self.no_interaction_layer = self._resolve_no_interaction_layer(
                self.no_interaction_layer,
                early_exit_layer=resolved_early_exit_layer,
            )
            self.config.no_interaction_layer = self.no_interaction_layer
            self.config.full_document_mask_layer = self.no_interaction_layer
        self._sync_attention_schedule()

    def set_no_interaction_layer(self, no_interaction_layer: Optional[int]) -> None:
        resolved_no_interaction_layer = self._resolve_no_interaction_layer(
            no_interaction_layer,
            early_exit_layer=self.early_exit_layer,
        )
        self.no_interaction_layer = resolved_no_interaction_layer
        self.config.no_interaction_layer = resolved_no_interaction_layer
        self.config.full_document_mask_layer = resolved_no_interaction_layer
        self._sync_attention_schedule()

    def set_full_document_mask_layer(self, full_document_mask_layer: Optional[int]) -> None:
        self.set_no_interaction_layer(full_document_mask_layer)

    def _set_attention_rerank_mask_indices(self, rerank_mask_indices) -> None:
        self.rerank_mask_indices = rerank_mask_indices
        for layer in self.language_model.layers:
            layer.self_attn.rerank_mask_indices = rerank_mask_indices

    def set_rerank_mask_indices(self, rerank_mask_indices) -> None:
        self._set_attention_rerank_mask_indices(_normalize_rerank_mask_indices(rerank_mask_indices))

    def set_document_span(self, document_span) -> None:
        raise ValueError("document_span is deprecated; pass rerank_mask_indices with external Q/K masks instead.")

    def _build_position_ids_and_image_spans(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_keep_indices: Any = None,
        video_keep_indices: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[dict[str, int]]]]:
        if input_ids is None:
            raise ValueError("input_ids must be provided to build position ids")

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_keep_entries = _normalize_keep_entries(
            image_keep_indices,
            0 if image_grid_thw is None else image_grid_thw.reshape(-1, 3).shape[0],
            input_ids.device,
        )
        video_keep_entries = _expand_video_keep_indices_by_frame(
            video_grid_thw,
            video_keep_indices,
            spatial_merge_size,
            input_ids.device,
        )
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        batch_size, seq_len = input_ids.shape
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(attention_mask.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(seq_len, device=input_ids.device).view(1, 1, -1).expand(3, batch_size, -1)
                )
                mrope_position_deltas = torch.zeros(
                    [batch_size, 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas, [[] for _ in range(batch_size)]

        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        vision_end_token_id = self.config.vision_end_token_id

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(
            3,
            batch_size,
            seq_len,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )

        image_index, video_index = 0, 0
        mrope_position_deltas = []
        attention_mask = attention_mask.to(input_ids.device)

        for batch_idx, sample_input_ids in enumerate(input_ids):
            sample_input_ids = sample_input_ids[attention_mask[batch_idx] == 1]
            vision_start_indices = torch.argwhere(sample_input_ids == vision_start_token_id).squeeze(1)
            vision_tokens = sample_input_ids[vision_start_indices + 1]
            image_nums = int((vision_tokens == image_token_id).sum().item())
            video_nums = int((vision_tokens == video_token_id).sum().item())

            input_tokens = sample_input_ids.tolist()
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            pre_img_idx = None

            for _ in range(image_nums + video_nums):
                if image_token_id in input_tokens and remain_images > 0:
                    ed_image = input_tokens.index(image_token_id, st)
                else:
                    ed_image = len(input_tokens) + 1
                if video_token_id in input_tokens and remain_videos > 0:
                    ed_video = input_tokens.index(video_token_id, st)
                else:
                    ed_video = len(input_tokens) + 1

                if ed_image < ed_video:
                    t, h, w = image_grid_thw[image_index]
                    keep_indices = image_keep_entries[image_index] if image_keep_entries else None
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t, h, w = video_grid_thw[video_index]
                    keep_indices = video_keep_entries[video_index] if video_keep_entries else None
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video

                llm_grid_t = t.item()
                llm_grid_h = h.item() // spatial_merge_size
                llm_grid_w = w.item() // spatial_merge_size
                total_visual_tokens = int(llm_grid_t * llm_grid_h * llm_grid_w)
                keep_indices = (
                    None
                    if keep_indices is None
                    else _safe_sorted_unique(keep_indices, total_visual_tokens).to(sample_input_ids.device)
                )
                text_len = ed - st

                if pre_img_idx is None:
                    pre_img_idx = text_len

                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                if text_len == 2 and input_tokens[st] == vision_end_token_id:
                    text_positions = torch.tensor(
                        [st_idx, pre_img_idx - 1],
                        device=sample_input_ids.device,
                        dtype=sample_input_ids.dtype,
                    ).view(1, -1).expand(3, -1)
                else:
                    text_positions = (
                        torch.arange(text_len, device=sample_input_ids.device, dtype=sample_input_ids.dtype)
                        .view(1, -1)
                        .expand(3, -1)
                        + st_idx
                    )
                llm_pos_ids_list.append(text_positions)

                visual_positions = _grid_keep_coords(
                    [llm_grid_t, h.item(), w.item()],
                    keep_indices,
                    spatial_merge_size,
                    device=sample_input_ids.device,
                    dtype=sample_input_ids.dtype,
                )
                llm_pos_ids_list.append(visual_positions + pre_img_idx)

                st = ed + visual_positions.shape[1]

            if st < len(input_tokens):
                st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
                text_len = len(input_tokens) - st
                llm_pos_ids_list.append(
                    torch.arange(text_len, device=sample_input_ids.device, dtype=sample_input_ids.dtype)
                    .view(1, -1)
                    .expand(3, -1)
                    + st_idx
                )

            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if llm_positions.shape[1] != int((attention_mask[batch_idx] == 1).sum().item()):
                raise ValueError(
                    "Pruned M-RoPE length mismatch: "
                    f"positions={llm_positions.shape[1]}, "
                    f"tokens={int((attention_mask[batch_idx] == 1).sum().item())}"
                )
            position_ids[..., batch_idx, attention_mask[batch_idx] == 1] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(input_ids[batch_idx]))

        mrope_position_deltas = torch.tensor(
            mrope_position_deltas,
            device=input_ids.device,
            dtype=input_ids.dtype,
        ).unsqueeze(1)
        return position_ids, mrope_position_deltas, [[] for _ in range(batch_size)]

    def get_rope_index(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_keep_indices: Any = None,
        video_keep_indices: Any = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids, mrope_position_deltas, _ = self._build_position_ids_and_image_spans(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_keep_indices=image_keep_indices,
            video_keep_indices=video_keep_indices,
            attention_mask=attention_mask,
        )
        self._set_attention_rerank_mask_indices(self.rerank_mask_indices)
        return position_ids, mrope_position_deltas

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_keep_indices: Any = None,
        video_keep_indices: Any = None,
        cache_position: Optional[torch.LongTensor] = None,
        rerank_mask_indices=_UNSET,
        document_span=_UNSET,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLModelOutputWithPast]:
        if document_span is not _UNSET:
            self.set_document_span(document_span)
        if rerank_mask_indices is not _UNSET:
            self.set_rerank_mask_indices(rerank_mask_indices)

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_lengths = _media_grid_lengths(image_grid_thw, self.config.vision_config.spatial_merge_size)
            image_keep_entries = _normalize_keep_entries(image_keep_indices, len(image_lengths), inputs_embeds.device)
            image_embeds = _prune_visual_output(image_embeds, image_keep_entries, image_lengths)
            deepstack_image_embeds = _prune_visual_output(deepstack_image_embeds, image_keep_entries, image_lengths)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_lengths = _media_grid_lengths(video_grid_thw, self.config.vision_config.spatial_merge_size)
            video_keep_entries = _normalize_keep_entries(video_keep_indices, len(video_lengths), inputs_embeds.device)
            video_embeds = _prune_visual_output(video_embeds, video_keep_entries, video_lengths)
            deepstack_video_embeds = _prune_visual_output(deepstack_video_embeds, video_keep_entries, video_lengths)
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            image_mask = image_mask[..., 0]
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            video_mask = video_mask[..., 0]
            visual_pos_masks = video_mask
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            attention_mask_tensor = (
                attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
            )
            if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                if attention_mask_tensor.dtype.is_floating_point:
                    attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                    attention_mask_tensor = (1.0 - attention_mask_tensor).int()

            prefill_compiled_stage = is_torchdynamo_compiling() and (
                (input_ids is not None and input_ids.shape[1] != 1)
                or (inputs_embeds is not None and inputs_embeds.shape[1] != 1)
            )
            prefill_noncompiled_stage = not is_torchdynamo_compiling() and (
                (cache_position is not None and cache_position[0] == 0)
                or (past_key_values is None or past_key_values.get_seq_length() == 0)
            )

            if (prefill_compiled_stage or prefill_noncompiled_stage) or self.rope_deltas is None:
                if self._requires_rerank_mask_indices() and self.rerank_mask_indices is None:
                    raise ValueError(
                        "rerank_mask_indices must be provided when no_interaction_layer >= 0 or k/k_selection is set"
                    )

                position_ids, rope_deltas, _ = self._build_position_ids_and_image_spans(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_keep_indices=image_keep_indices,
                    video_keep_indices=video_keep_indices,
                    attention_mask=attention_mask_tensor,
                )
                self.rope_deltas = rope_deltas
                self._set_attention_rerank_mask_indices(self.rerank_mask_indices)
            else:
                batch_size, seq_length, _ = inputs_embeds.shape
                delta = (
                    (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
                    if cache_position is not None
                    else 0
                )
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                if cache_position is not None:
                    delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
                self._set_attention_rerank_mask_indices(self.rerank_mask_indices)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            early_exit_layer=self.early_exit_layer,
            **kwargs,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
        )


class CrossAttnMasking_Qwen3_VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        if not hasattr(config, "k"):
            config.k = getattr(config, "last_k", -1)
        if not hasattr(config, "k_selection"):
            config.k_selection = "last" if getattr(config, "last_k", -1) >= 0 else None
        if not hasattr(config, "early_exit_layer"):
            config.early_exit_layer = config.text_config.num_hidden_layers - 1
        if not hasattr(config, "no_interaction_layer"):
            config.no_interaction_layer = getattr(config, "full_document_mask_layer", -1)
        config.full_document_mask_layer = config.no_interaction_layer
        if not hasattr(config, "rerank_format"):
            config.rerank_format = "query_first"
        if hasattr(config, "last_k"):
            delattr(config, "last_k")

        super().__init__(config)
        self.model = Custom_Qwen3_VLModel(config)
        self.set_k_config(config.k, config.k_selection)
        self.set_early_exit_layer(config.early_exit_layer)
        self.set_no_interaction_layer(config.no_interaction_layer)

    def set_k_config(self, k: Optional[int], k_selection: Optional[str]) -> None:
        self.model.set_k_config(k, k_selection)
        self.config.k = self.model.k
        self.config.k_selection = self.model.k_selection
        if hasattr(self.config, "last_k"):
            delattr(self.config, "last_k")

    def set_last_k(self, last_k: int) -> None:
        self.set_k_config(last_k, "last" if last_k is not None and last_k >= 0 else None)

    def set_early_exit_layer(self, early_exit_layer: Optional[int]) -> None:
        resolved_early_exit_layer = self.model._resolve_early_exit_layer(early_exit_layer)
        self.config.early_exit_layer = resolved_early_exit_layer
        self.model.set_early_exit_layer(resolved_early_exit_layer)

    def set_no_interaction_layer(self, no_interaction_layer: Optional[int]) -> None:
        resolved_no_interaction_layer = self.model._resolve_no_interaction_layer(
            no_interaction_layer,
            early_exit_layer=self.config.early_exit_layer,
        )
        self.config.no_interaction_layer = resolved_no_interaction_layer
        self.config.full_document_mask_layer = resolved_no_interaction_layer
        self.model.set_no_interaction_layer(resolved_no_interaction_layer)

    def set_full_document_mask_layer(self, full_document_mask_layer: Optional[int]) -> None:
        self.set_no_interaction_layer(full_document_mask_layer)

    @check_model_inputs
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        image_keep_indices: Any = None,
        video_keep_indices: Any = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        rerank_mask_indices=_UNSET,
        document_span=_UNSET,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        if document_span is not _UNSET:
            self.model.set_document_span(document_span)
        if rerank_mask_indices is not _UNSET:
            self.model.set_rerank_mask_indices(rerank_mask_indices)

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            image_keep_indices=image_keep_indices,
            video_keep_indices=video_keep_indices,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            rerank_mask_indices=rerank_mask_indices,
            **kwargs,
        )

        hidden_states = outputs[0]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(self, *args, rerank_mask_indices=_UNSET, document_span=_UNSET, **kwargs):
        model_inputs = super().prepare_inputs_for_generation(*args, **kwargs)
        if rerank_mask_indices is not _UNSET:
            model_inputs["rerank_mask_indices"] = rerank_mask_indices
        if document_span is not _UNSET:
            model_inputs["document_span"] = document_span
        return model_inputs

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        k = kwargs.pop("k", None)
        k_selection = kwargs.pop("k_selection", None)
        last_k = kwargs.pop("last_k", None)
        early_exit_layer = kwargs.pop("early_exit_layer", None)
        no_interaction_layer = kwargs.pop("no_interaction_layer", None)
        full_document_mask_layer = kwargs.pop("full_document_mask_layer", None)
        rerank_format = kwargs.pop("rerank_format", None)
        config = kwargs.get("config")

        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
            kwargs["config"] = config

        legacy_last_k = getattr(config, "last_k", -1)
        if not hasattr(config, "k"):
            config.k = legacy_last_k
        if not hasattr(config, "k_selection"):
            config.k_selection = "last" if legacy_last_k >= 0 else None
        if not hasattr(config, "early_exit_layer"):
            config.early_exit_layer = config.text_config.num_hidden_layers - 1
        if not hasattr(config, "no_interaction_layer"):
            config.no_interaction_layer = getattr(config, "full_document_mask_layer", -1)
        config.full_document_mask_layer = config.no_interaction_layer
        if not hasattr(config, "rerank_format"):
            config.rerank_format = "query_first"

        if last_k is not None:
            k = last_k if k is None else k
            k_selection = "last" if k_selection is None and k is not None and k >= 0 else k_selection
        if k is not None:
            config.k = k
        if k_selection is not None:
            config.k_selection = k_selection
        if early_exit_layer is not None:
            config.early_exit_layer = early_exit_layer
        if no_interaction_layer is None:
            no_interaction_layer = full_document_mask_layer
        elif full_document_mask_layer is not None and int(no_interaction_layer) != int(full_document_mask_layer):
            raise ValueError(
                "no_interaction_layer and legacy full_document_mask_layer disagree: "
                f"{no_interaction_layer} != {full_document_mask_layer}"
            )
        if no_interaction_layer is not None:
            config.no_interaction_layer = no_interaction_layer
            config.full_document_mask_layer = no_interaction_layer
        if full_document_mask_layer is not None:
            config.full_document_mask_layer = config.no_interaction_layer
        if rerank_format is not None:
            config.rerank_format = rerank_format
        if hasattr(config, "last_k"):
            delattr(config, "last_k")

        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.set_k_config(model.config.k, model.config.k_selection)
        model.set_early_exit_layer(model.config.early_exit_layer)
        model.set_no_interaction_layer(model.config.no_interaction_layer)
        return model
