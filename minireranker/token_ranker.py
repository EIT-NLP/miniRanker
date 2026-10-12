from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F


from ._vendor.qwen3_vl_embedding import Qwen3VLEmbedder



@dataclass
class VisualTokenRanking:
    modalities: List[str]
    token_counts: List[int]
    ranked_tokens: List[List[int]]

    @classmethod
    def empty(cls) -> "VisualTokenRanking":
        return cls(modalities=[], token_counts=[], ranked_tokens=[])


def _stable_descending_indices(scores: torch.Tensor) -> torch.Tensor:
    try:
        return torch.argsort(scores, descending=True, stable=True)
    except TypeError:
        pairs = sorted(
            enumerate(scores.detach().cpu().tolist()),
            key=lambda item: (-float(item[1]), item[0]),
        )
        return torch.tensor([idx for idx, _ in pairs], dtype=torch.long, device=scores.device)


class Qwen3VLVisualTokenRanker:
    def __init__(
        self,
        model_name_or_path: str,
        prune_method: str = "attention",
        prune_layer: Any = "all",
        normalize_embedding: bool = True,
        video_frame_max_pixels: Optional[int] = None,
        **kwargs: Any,
    ):
        if prune_method not in {"attention", "hidden_state_similarity"}:
            raise ValueError("prune_method must be `attention` or `hidden_state_similarity`.")
        self.prune_method = prune_method
        self.prune_layer = "all" if prune_layer == "all" else int(prune_layer)
        self.normalize_embedding = normalize_embedding
        self.video_frame_max_pixels = None if video_frame_max_pixels is None else int(video_frame_max_pixels)
        print(f"[token-ranker] loading embedder from: {model_name_or_path}", flush=True)
        self.embedder = Qwen3VLEmbedder(model_name_or_path=model_name_or_path, **kwargs)
        if prune_method == "attention":
            self.embedder.model.set_attn_implementation("eager")
        print("[token-ranker] embedder loaded", flush=True)

    def _apply_video_frame_max_pixels(self, conversation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.video_frame_max_pixels is None:
            return conversation
        for message in conversation:
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    item["max_pixels"] = self.video_frame_max_pixels
                    item["min_pixels"] = getattr(self.embedder, "min_pixels", 4096)
                    item["total_pixels"] = getattr(self.embedder, "total_pixels", 7864320)
                    item["fps"] = getattr(self.embedder, "fps", 1)
                    item["sample_fps"] = item["fps"]
        return conversation

    @staticmethod
    def _merge_visual_embeddings(
        image_mask: Optional[torch.Tensor],
        video_mask: Optional[torch.Tensor],
        deepstack_image_embeds: Optional[List[torch.Tensor]],
        deepstack_video_embeds: Optional[List[torch.Tensor]],
    ):
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None:
            image_mask = image_mask[..., 0]
        if video_mask is not None:
            video_mask = video_mask[..., 0]

        if image_mask is not None and video_mask is not None:
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_counts = image_mask.sum(dim=1).tolist()
            video_counts = video_mask.sum(dim=1).tolist()
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                row_embeds = []
                img_offset = 0
                vid_offset = 0
                hidden_dim = img_embed.shape[-1]
                for row_idx, visual_mask in enumerate(visual_pos_masks):
                    row_visual_count = int(visual_mask.sum().item())
                    row_embed = img_embed.new_zeros((row_visual_count, hidden_dim))
                    row_image_count = int(image_counts[row_idx])
                    row_video_count = int(video_counts[row_idx])
                    row_image_mask = image_mask[row_idx, visual_mask]
                    row_video_mask = video_mask[row_idx, visual_mask]
                    if row_image_count:
                        row_embed[row_image_mask] = img_embed[img_offset:img_offset + row_image_count]
                    if row_video_count:
                        row_embed[row_video_mask] = vid_embed[vid_offset:vid_offset + row_video_count]
                    row_embeds.append(row_embed)
                    img_offset += row_image_count
                    vid_offset += row_video_count
                deepstack_visual_embeds.append(torch.cat(row_embeds, dim=0))
        elif image_mask is not None:
            visual_pos_masks = image_mask
            deepstack_visual_embeds = list(deepstack_image_embeds)
        elif video_mask is not None:
            visual_pos_masks = video_mask
            deepstack_visual_embeds = list(deepstack_video_embeds)
        return visual_pos_masks, deepstack_visual_embeds

    @staticmethod
    def _extract_media_spans(
        input_ids: torch.Tensor,
        grid_thw: Optional[torch.Tensor],
        token_id: int,
        spatial_merge_size: int,
        modality: str,
    ):
        batch_size = input_ids.shape[0]
        if grid_thw is None or grid_thw.numel() == 0:
            return [[] for _ in range(batch_size)]
        split_sizes = (grid_thw.prod(-1) // int(spatial_merge_size) ** 2).tolist()
        split_idx = 0
        all_spans = []
        for batch_idx in range(batch_size):
            positions = torch.nonzero(input_ids[batch_idx] == int(token_id), as_tuple=False).squeeze(-1)
            positions = positions.reshape(-1)
            spans = []
            consumed = 0
            while consumed < int(positions.numel()):
                if split_idx >= len(split_sizes):
                    raise ValueError(f"{modality} metadata ended early for sample {batch_idx}.")
                split_size = int(split_sizes[split_idx])
                spans.append((modality, positions[consumed:consumed + split_size]))
                consumed += split_size
                split_idx += 1
            all_spans.append(spans)
        if split_idx != len(split_sizes):
            raise ValueError(f"Unused {modality} metadata: consumed {split_idx} of {len(split_sizes)} grids.")
        return all_spans

    def _prepare_sequence(self, processed_inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        model = self.embedder.model.model
        input_ids = processed_inputs["input_ids"]
        attention_mask = processed_inputs.get("attention_mask")
        image_grid_thw = processed_inputs.get("image_grid_thw")
        video_grid_thw = processed_inputs.get("video_grid_thw")
        pixel_values = processed_inputs.get("pixel_values")
        pixel_values_videos = processed_inputs.get("pixel_values_videos")

        inputs_embeds = model.get_input_embeddings()(input_ids)
        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = model.get_image_features(pixel_values, image_grid_thw)
            flat_image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=flat_image_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, flat_image_embeds)

        if pixel_values_videos is not None:
            video_embeds, deepstack_video_embeds = model.get_video_features(pixel_values_videos, video_grid_thw)
            flat_video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = model.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=flat_video_embeds,
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, flat_video_embeds)

        position_ids, rope_deltas = model.get_rope_index(
            input_ids=input_ids,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
        )
        model.rope_deltas = rope_deltas
        visual_pos_masks, deepstack_visual_embeds = self._merge_visual_embeddings(
            image_mask,
            video_mask,
            deepstack_image_embeds,
            deepstack_video_embeds,
        )
        spatial_merge_size = model.visual.spatial_merge_size
        image_spans = self._extract_media_spans(
            input_ids,
            image_grid_thw,
            model.config.image_token_id,
            spatial_merge_size,
            "image",
        )
        video_spans = self._extract_media_spans(
            input_ids,
            video_grid_thw,
            model.config.video_token_id,
            spatial_merge_size,
            "video",
        )
        media_spans = []
        for row_images, row_videos in zip(image_spans, video_spans):
            row = [*row_images, *row_videos]
            row.sort(key=lambda item: int(item[1][0].item()) if item[1].numel() else -1)
            media_spans.append(row)
        return {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
            "position_ids": position_ids,
            "visual_pos_masks": visual_pos_masks,
            "deepstack_visual_embeds": deepstack_visual_embeds,
            "media_spans": media_spans,
            "model": model,
        }

    @staticmethod
    def _last_valid_indices(attention_mask: Optional[torch.Tensor], seq_len: int, batch_size: int) -> torch.Tensor:
        if attention_mask is None:
            return torch.full((batch_size,), seq_len - 1, dtype=torch.long)
        flipped = attention_mask.to(torch.long).flip(dims=[1])
        last_one_positions = flipped.argmax(dim=1)
        return (attention_mask.shape[1] - last_one_positions - 1).to(torch.long)

    def _validate_prune_layer(self, num_layers: int) -> None:
        if self.prune_layer == "all":
            return
        if not (0 <= int(self.prune_layer) < num_layers):
            raise ValueError(f"prune_layer must be in [0, {num_layers - 1}] or `all`, got {self.prune_layer}.")

    def _select_hidden_states(self, outputs: Any, num_layers: int, layer_idx: int) -> torch.Tensor:
        if layer_idx == num_layers - 1:
            return outputs.last_hidden_state
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise ValueError("Embedder hidden states were not returned.")
        if len(hidden_states) == num_layers + 1:
            return hidden_states[layer_idx + 1]
        if len(hidden_states) == num_layers:
            return hidden_states[layer_idx]
        raise ValueError(f"Unexpected hidden-state tuple length {len(hidden_states)}.")

    def _similarity_scores(
        self,
        outputs: Any,
        prepared: Dict[str, Any],
        row_idx: int,
        last_idx: int,
        positions: torch.Tensor,
        num_layers: int,
    ) -> torch.Tensor:
        embedding_vector = outputs.last_hidden_state[row_idx, last_idx:last_idx + 1]
        if self.normalize_embedding:
            embedding_vector = F.normalize(embedding_vector, p=2, dim=-1)
        layer_ids = range(num_layers) if self.prune_layer == "all" else [int(self.prune_layer)]
        scores = None
        for layer_idx in layer_ids:
            hidden = self._select_hidden_states(outputs, num_layers, layer_idx)
            if layer_idx != num_layers - 1:
                hidden = prepared["model"].language_model.norm(hidden)
            visual_tokens = F.normalize(hidden[row_idx, positions], p=2, dim=-1)
            layer_scores = torch.matmul(visual_tokens, embedding_vector[0])
            scores = layer_scores if scores is None else scores + layer_scores
        return scores

    def _attention_scores(
        self,
        outputs: Any,
        row_idx: int,
        last_idx: int,
        positions: torch.Tensor,
        num_layers: int,
    ) -> torch.Tensor:
        if outputs.attentions is None:
            raise ValueError("Embedder attentions were not returned.")
        layer_ids = range(num_layers) if self.prune_layer == "all" else [int(self.prune_layer)]
        scores = None
        for layer_idx in layer_ids:
            layer_scores = outputs.attentions[layer_idx][row_idx, :, last_idx, positions].mean(dim=0)
            scores = layer_scores if scores is None else scores + layer_scores
        return scores

    @torch.no_grad()
    def build_rankings(
        self,
        samples: List[Dict[str, Any]],
        instructions: Optional[List[Optional[str]]] = None,
    ) -> List[VisualTokenRanking]:
        if not samples:
            return []

        normalized_instructions = [None] * len(samples) if instructions is None else instructions
        if len(normalized_instructions) != len(samples):
            raise ValueError("instructions must match samples length.")

        conversations = []
        for sample, instruction in zip(samples, normalized_instructions):
            sample_for_embedder = dict(sample)
            if instruction is not None and "instruction" not in sample_for_embedder:
                sample_for_embedder["instruction"] = instruction
            conversation = self.embedder.format_model_input(
                text=sample_for_embedder.get("text"),
                image=sample_for_embedder.get("image"),
                video=sample_for_embedder.get("video"),
                instruction=sample_for_embedder.get("instruction"),
                fps=sample_for_embedder.get("fps"),
                max_frames=sample_for_embedder.get("max_frames"),
            )
            conversations.append(self._apply_video_frame_max_pixels(conversation))

        processed_inputs = self.embedder._preprocess_inputs(conversations)
        processed_inputs = {
            key: value.to(self.embedder.model.device)
            for key, value in processed_inputs.items()
        }
        prepared = self._prepare_sequence(processed_inputs)
        media_spans = prepared["media_spans"]
        if not any(media_spans):
            return [VisualTokenRanking.empty() for _ in samples]

        num_layers = len(prepared["model"].language_model.layers)
        self._validate_prune_layer(num_layers)
        batch_size = prepared["inputs_embeds"].shape[0]
        last_indices = self._last_valid_indices(prepared["attention_mask"],
                                                prepared["inputs_embeds"].shape[1], batch_size)
        attention_sums = [None] * batch_size
        handles = []
        if self.prune_method == "attention":
            selected_layers = range(num_layers) if self.prune_layer == "all" else [int(self.prune_layer)]
            def capture_attention(module, args, output):
                weights = output[1]
                if weights is None:
                    raise RuntimeError("The embedder must use eager attention for attention-based ranking.")
                for row_idx, spans in enumerate(media_spans):
                    if not spans:
                        continue
                    positions = torch.cat([span for _, span in spans])
                    values = weights[row_idx, :, int(last_indices[row_idx]), positions].mean(dim=0).detach()
                    attention_sums[row_idx] = values if attention_sums[row_idx] is None else attention_sums[row_idx] + values
                return output[0], None
            for layer_idx in selected_layers:
                handles.append(prepared["model"].language_model.layers[layer_idx].self_attn.register_forward_hook(capture_attention))

        output_hidden_states = self.prune_method == "hidden_state_similarity"
        output_attentions = False
        try:
            outputs = prepared["model"].language_model(
                input_ids=None,
                inputs_embeds=prepared["inputs_embeds"],
                attention_mask=prepared["attention_mask"],
                position_ids=prepared["position_ids"],
                visual_pos_masks=prepared["visual_pos_masks"],
                deepstack_visual_embeds=prepared["deepstack_visual_embeds"],
                use_cache=False,
                output_hidden_states=output_hidden_states,
                output_attentions=output_attentions,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()

        batch_size = prepared["inputs_embeds"].shape[0]
        last_indices = self._last_valid_indices(
            prepared["attention_mask"],
            prepared["inputs_embeds"].shape[1],
            batch_size,
        ).to(prepared["inputs_embeds"].device)
        num_layers = len(prepared["model"].language_model.layers)
        self._validate_prune_layer(num_layers)

        rankings: List[VisualTokenRanking] = []
        for row_idx, row_media_spans in enumerate(media_spans):
            if not row_media_spans:
                rankings.append(VisualTokenRanking.empty())
                continue

            all_positions = torch.cat([span for _, span in row_media_spans], dim=0)
            last_idx = int(last_indices[row_idx].item())
            if self.prune_method == "hidden_state_similarity":
                scores = self._similarity_scores(
                    outputs, prepared, row_idx, last_idx, all_positions, num_layers
                )
            elif self.prune_method == "attention":
                scores = attention_sums[row_idx]
                if scores is None:
                    raise RuntimeError("No attention rows were captured for a visual input.")
            else:
                raise ValueError(
                    "prune_method must be `hidden_state_similarity` or `attention`, "
                    f"got {self.prune_method!r}."
                )

            global_to_media: List[List[int]] = []
            modalities: List[str] = []
            token_counts: List[int] = []
            for media_idx, (modality, span_positions) in enumerate(row_media_spans):
                modalities.append(str(modality))
                token_count = int(span_positions.numel())
                token_counts.append(token_count)
                for local_idx in range(token_count):
                    global_to_media.append([media_idx, local_idx])

            order = _stable_descending_indices(scores).detach().cpu().tolist()
            rankings.append(
                VisualTokenRanking(
                    modalities=modalities,
                    token_counts=token_counts,
                    ranked_tokens=[global_to_media[int(idx)] for idx in order],
                )
            )

        return rankings
