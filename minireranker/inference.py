"""Point-wise multimodal inference using the same preparation as training."""
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoConfig, AutoProcessor

from .batch_processing import apply_token_pruning_to_batch, compact_processor_kwargs
from .custom_modeling_qwen3_vl import CrossAttnMasking_Qwen3_VLForConditionalGeneration
from .data_utils import infer_video_root
from .rerank_utils import (
    build_processor_vision_kwargs, build_rerank_messages_from_example,
    compute_early_segment_span_from_messages, ensure_pil_or_none,
    normalize_text, normalize_video_paths,
)


class MiniReranker:
    """Return unnormalized yes-minus-no scores in document input order.

    Each query/document is a mapping with ``text``, one ``image`` (path,
    bytes, PIL image), and/or ``video`` (an ordered list of frame paths).
    Inference processes one pair at a time so padding cannot move its score
    token. Visual rankings can be supplied with process_examples() or
    computed online with a Qwen3-VL embedding checkpoint.
    """

    def __init__(
        self, model_name_or_path: str, *,
        embedder_model_name_or_path: str = "Qwen/Qwen3-VL-Embedding-2B",
        prune_k: Optional[float] = None,
        prune_method: str = "attention", prune_layer: Any = "all",
        rerank_format: Optional[str] = None,
        early_exit_layer: Optional[int] = None,
        no_interaction_layer: Optional[int] = None,
        video_root: Optional[str] = None,
        default_instruction: str = "Given a search query, retrieve relevant candidates that answer the query.",
        device: Optional[str] = None, torch_dtype=None,
        attn_implementation: str = "eager", vision_attn_implementation: str = "sdpa",
        **kwargs,
    ):
        config = AutoConfig.from_pretrained(model_name_or_path)
        if early_exit_layer is not None:
            config.early_exit_layer = int(early_exit_layer)
        if no_interaction_layer is not None:
            config.no_interaction_layer = int(no_interaction_layer)
            config.full_document_mask_layer = int(no_interaction_layer)
        if rerank_format is not None:
            config.rerank_format = rerank_format
        self.rerank_format = getattr(config, "rerank_format", "image_first")
        config.rerank_format = self.rerank_format
        self.prune_k = getattr(config, "prune_k", None) if prune_k is None else float(prune_k)
        if self.prune_k is not None and self.prune_k <= 0:
            raise ValueError("prune_k must be positive, or None to use the checkpoint setting.")
        self.vision = {
            "max_length": 10240, "min_pixels": 4096, "max_pixels": 1843200,
            "video_frame_max_pixels": 786432, "video_total_pixels": 7864320,
            "fps": 1, "max_video_frames": 64,
            **getattr(config, "minireranker_vision", {}),
        }
        for key in self.vision:
            if key in kwargs:
                self.vision[key] = kwargs.pop(key)
        config.text_config._attn_implementation = attn_implementation
        config.vision_config._attn_implementation = vision_attn_implementation
        config._attn_implementation = {
            "text_config": attn_implementation, "vision_config": vision_attn_implementation,
        }
        self.device = torch.device(device or (f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"))
        dtype = kwargs.pop("dtype", torch_dtype)
        self.dtype = dtype or (torch.bfloat16 if self.device.type == "cuda" else torch.float32)
        self.model = CrossAttnMasking_Qwen3_VLForConditionalGeneration.from_pretrained(
            model_name_or_path, config=config, dtype=self.dtype, **kwargs,
        )
        if "device_map" not in kwargs:
            self.model.to(self.device)
        self.model.eval()
        self.device = self.model.device
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, padding_side="right")
        self.processor.tokenizer.model_max_length = self.vision["max_length"]
        vocab = self.processor.tokenizer.get_vocab()
        if "yes" not in vocab or "no" not in vocab:
            raise ValueError("The checkpoint tokenizer must contain single-token 'yes' and 'no' labels.")
        weights = self.model.lm_head.weight
        self.score_linear = torch.nn.Linear(weights.shape[1], 1, bias=False).to(weights)
        with torch.no_grad():
            self.score_linear.weight[0].copy_(weights[vocab["yes"]] - weights[vocab["no"]])
        self.score_linear.eval()
        self.embedder_model_name_or_path = embedder_model_name_or_path
        self.prune_method, self.prune_layer = prune_method, prune_layer
        self.video_root = infer_video_root("data/miniReranker_736k", video_root)
        self.default_instruction = default_instruction
        self._ranker = None

    def _get_ranker(self):
        if self._ranker is None:
            from .token_ranker import Qwen3VLVisualTokenRanker
            self._ranker = Qwen3VLVisualTokenRanker(
                self.embedder_model_name_or_path, prune_method=self.prune_method,
                prune_layer=self.prune_layer, torch_dtype=self.dtype,
                max_length=self.vision["max_length"], min_pixels=self.vision["min_pixels"],
                max_pixels=self.vision["max_pixels"], fps=self.vision["fps"],
                max_frames=self.vision["max_video_frames"],
                video_frame_max_pixels=self.vision["video_frame_max_pixels"],
                total_pixels=self.vision["video_total_pixels"], device=str(self.device),
            )
        return self._ranker

    def _rank_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example = dict(example)
        if self.prune_k is None:
            return example
        for side in ("query", "document"):
            image = ensure_pil_or_none(example.get(f"{side}_image_bytes"))
            frames = normalize_video_paths(example.get(f"{side}_video_frame_paths"), self.video_root,
                                           self.vision["max_video_frames"])
            example[f"{side}_image_bytes"] = image
            example[f"{side}_video_frame_paths"] = frames
            if image is None and not frames:
                for suffix in ("modalities", "token_counts", "ranked_tokens"):
                    example[f"{side}_visual_{suffix}"] = []
                continue
            if all(example.get(f"{side}_visual_{suffix}") is not None
                   for suffix in ("modalities", "token_counts", "ranked_tokens")):
                continue
            sample = {
                "text": example.get(f"{side}_text", ""), "image": image, "video": frames,
                "fps": self.vision["fps"], "max_frames": self.vision["max_video_frames"],
            }
            if side == "query":
                sample["instruction"] = example.get("instruction", "")
            ranking = self._get_ranker().build_rankings([sample])[0]
            example[f"{side}_visual_modalities"] = ranking.modalities
            example[f"{side}_visual_token_counts"] = ranking.token_counts
            example[f"{side}_visual_ranked_tokens"] = ranking.ranked_tokens
        return example

    def prepare_inputs(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare one dataset-format pair; supervision labels are never included."""
        example = self._rank_example(example)
        messages = build_rerank_messages_from_example(
            example, include_label=False, rerank_format=self.rerank_format,
            video_root=self.video_root, max_frames=self.vision["max_video_frames"],
            min_pixels=self.vision["min_pixels"], image_max_pixels=self.vision["max_pixels"],
            video_frame_max_pixels=self.vision["video_frame_max_pixels"],
            video_total_pixels=self.vision["video_total_pixels"], fps=self.vision["fps"],
        )
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        span, _ = compute_early_segment_span_from_messages(self.processor, messages)
        full = self.processor(
            text=[text], padding=True, truncation=False, return_tensors="pt", do_resize=False,
            **compact_processor_kwargs(build_processor_vision_kwargs([messages], self.processor)),
        )
        if full["input_ids"].shape[1] > self.vision["max_length"]:
            raise ValueError("Input exceeds max_length; reduce visual budgets or input text length.")
        prepared = apply_token_pruning_to_batch(
            full, torch.full_like(full["input_ids"], -100), [span], [example],
            prune_k=self.prune_k, rerank_format=self.rerank_format, processor=self.processor,
            k=self.model.config.k, k_selection=self.model.config.k_selection,
        )
        prepared.pop("labels", None)
        for key, value in full.items():
            prepared.setdefault(key, value)
        return {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                for key, value in prepared.items()}

    @torch.no_grad()
    def compute_scores(self, inputs: Dict[str, Any]) -> List[float]:
        inputs = {key: value for key, value in inputs.items() if key != "labels"}
        hidden = self.model.model(**inputs, use_cache=False).last_hidden_state
        mask = inputs.get("attention_mask")
        if mask is None:
            last = hidden[:, -1]
        else:
            positions = mask.shape[1] - 1 - mask.flip(1).long().argmax(1)
            last = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions]
        return self.score_linear(last).squeeze(-1).float().cpu().tolist()

    def process_examples(self, examples: List[Dict[str, Any]]) -> List[float]:
        """Score Parquet-format pairs, reusing supplied visual token rankings."""
        return [self.compute_scores(self.prepare_inputs(example))[0] for example in examples]

    def process(self, inputs: Dict[str, Any]) -> List[float]:
        query = inputs.get("query", {})
        if isinstance(query, str):
            query = {"text": query}
        examples = []
        for document in inputs.get("documents", []):
            if isinstance(document, str):
                document = {"text": document}
            example = {"instruction": inputs.get("instruction") or self.default_instruction}
            for side, sample in (("query", query), ("document", document)):
                image = sample.get("image")
                if isinstance(image, (list, tuple)):
                    if len(image) > 1:
                        raise ValueError("Only one image per query/document is supported by the training schema.")
                    image = image[0] if image else None
                example[f"{side}_text"] = normalize_text(sample.get("text", ""))
                example[f"{side}_image_bytes"] = image or None
                example[f"{side}_video_frame_paths"] = sample.get("video") or []
            examples.append(example)
        return self.process_examples(examples)
