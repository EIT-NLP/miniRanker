from .batch_processing import compact_processor_kwargs, apply_token_pruning_to_batch
from .data_utils import infer_video_root
import glob
import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import torch
# torch.backends.cudnn.enabled = False
import torch.distributed as dist
import torch.nn.functional as F
from datasets import concatenate_datasets, load_dataset
from torch.utils.data import DataLoader, Sampler
from transformers import AutoConfig, AutoProcessor, HfArgumentParser, Trainer
from transformers.trainer import seed_worker
# qwen_vl_utils

# import flash_attn

try:
    from peft import LoraConfig, get_peft_model
    _PEFT_IMPORT_ERROR = None
except ImportError as exc:
    LoraConfig = None
    get_peft_model = None
    _PEFT_IMPORT_ERROR = exc

from .arguments import DataArguments, ModelArguments, TrainingArguments
from .custom_modeling_qwen3_vl import CrossAttnMasking_Qwen3_VLForConditionalGeneration
from .rerank_utils import (
    ALLOWED_K_SELECTIONS,
    ALLOWED_RERANK_FORMATS,
    DEFAULT_SYSTEM_PROMPT,
    build_processor_vision_kwargs,
    build_rerank_messages_from_example,
    build_rerank_attention_mask_indices,
    compute_early_segment_span_from_messages,
    normalize_text,
)
from .token_pruning_utils import (
    normalize_modalities,
    normalize_token_counts,
    select_keep_indices_by_rank,
)

def _normalize_subsets(subsets: Optional[Any]) -> Optional[List[str]]:
    if subsets is None:
        return None
    if isinstance(subsets, str):
        items = [item.strip() for item in subsets.split(",") if item.strip()]
        return items or None
    return list(subsets)

def collect_parquet_files(data_root: str, subsets: Optional[Any], pattern: str) -> List[str]:
    files = []
    subset_names = _normalize_subsets(subsets)
    if subset_names is None:
        subset_dirs = [p for p in glob.glob(os.path.join(data_root, "*")) if os.path.isdir(p)]
    else:
        subset_dirs = [os.path.join(data_root, subset_name) for subset_name in subset_names]

    for subset_dir in subset_dirs:
        files.extend(sorted(glob.glob(os.path.join(subset_dir, pattern))))
    return files

def freeze_vision_and_projector(model, freeze_projector):
    frozen, total = 0, 0
    for name, param in model.named_parameters():
        lname = name.lower()
        freeze_flag = False

        if "visual" in lname or "vision_tower" in lname or "vision_model" in lname or "vit" in lname:
            freeze_flag = True

        if freeze_projector and (
            "merger" in lname or "projector" in lname or "multi_modal_projector" in lname
        ):
            freeze_flag = True

        if freeze_flag:
            param.requires_grad = False
            frozen += param.numel()
        total += param.numel()

    print(f"Frozen params: {frozen / 1e6:.2f}M / {total / 1e6:.2f}M")

def get_model_config(model):
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        if hasattr(base_model, "config"):
            return base_model.config
    return model.config

def save_model_config(model, output_dir: str) -> None:
    get_model_config(model).save_pretrained(output_dir)

def apply_model_config_values(
    model,
    k: int,
    k_selection: Optional[str],
    early_exit_layer: int,
    no_interaction_layer: int,
    rerank_format: str,
) -> None:
    config = get_model_config(model)
    config.k = k
    config.k_selection = k_selection
    config.early_exit_layer = early_exit_layer
    config.no_interaction_layer = no_interaction_layer
    config.full_document_mask_layer = no_interaction_layer
    config.rerank_format = rerank_format
    if hasattr(config, "last_k"):
        delattr(config, "last_k")

    for subconfig_name in ("text_config", "vision_config"):
        subconfig = getattr(config, subconfig_name, None)
        if subconfig is None:
            continue
        for key in (
            "early_exit_layer",
            "no_interaction_layer",
            "full_document_mask_layer",
            "k",
            "k_selection",
            "rerank_format",
            "last_k",
        ):
            if hasattr(subconfig, key):
                delattr(subconfig, key)

def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def barrier_if_distributed() -> None:
    if is_distributed_initialized():
        dist.barrier()

def resolve_no_interaction_layer_arg(model_args: ModelArguments) -> Optional[int]:
    no_interaction_layer = model_args.no_interaction_layer
    legacy_layer = model_args.full_document_mask_layer
    if no_interaction_layer is not None and legacy_layer is not None and int(no_interaction_layer) != int(legacy_layer):
        raise ValueError(
            "no_interaction_layer and legacy full_document_mask_layer disagree: "
            f"{no_interaction_layer} != {legacy_layer}"
        )
    return no_interaction_layer if no_interaction_layer is not None else legacy_layer

def resolve_masking_schedule(config, model_args: ModelArguments) -> tuple[int, int]:
    num_hidden_layers = config.text_config.num_hidden_layers

    early_exit_layer = (
        num_hidden_layers - 1 if model_args.early_exit_layer is None else int(model_args.early_exit_layer)
    )
    if early_exit_layer < 0 or early_exit_layer >= num_hidden_layers:
        raise ValueError(
            f"early_exit_layer must be in [0, {num_hidden_layers - 1}], got {early_exit_layer}"
        )

    no_interaction_layer_arg = resolve_no_interaction_layer_arg(model_args)
    no_interaction_layer = (
        -1
        if no_interaction_layer_arg is None
        else int(no_interaction_layer_arg)
    )
    if no_interaction_layer < -1 or no_interaction_layer >= num_hidden_layers:
        raise ValueError(
            "no_interaction_layer must be in [-1, "
            f"{num_hidden_layers - 1}], got {no_interaction_layer}"
        )
    if no_interaction_layer > early_exit_layer:
        raise ValueError(
            "no_interaction_layer must be smaller than or equal to early_exit_layer, got "
            f"{no_interaction_layer} > {early_exit_layer}"
        )

    return early_exit_layer, no_interaction_layer

def resolve_k_config(model_args: ModelArguments) -> tuple[int, Optional[str]]:
    if model_args.last_k is not None and model_args.k == -1 and model_args.k_selection is None:
        model_args.k = int(model_args.last_k)
        model_args.k_selection = "last" if model_args.k >= 0 else None

    k = int(model_args.k)
    k_selection = model_args.k_selection
    if k_selection is not None and k_selection not in ALLOWED_K_SELECTIONS:
        raise ValueError(f"k_selection must be one of {sorted(ALLOWED_K_SELECTIONS)}, got {k_selection}")
    if k_selection is None and k != -1:
        raise ValueError(f"k_selection must be set when k is set. Got k={k}, k_selection=None")
    if k_selection is not None and k < 0:
        raise ValueError(f"k must be >= 0 when k_selection is set. Got k={k}, k_selection={k_selection}")
    if k_selection == "step" and k <= 0:
        raise ValueError("k must be > 0 when k_selection='step'")
    return k, k_selection

def resolve_rerank_format(data_args: DataArguments) -> str:
    rerank_format = data_args.rerank_format
    if rerank_format not in ALLOWED_RERANK_FORMATS:
        raise ValueError(f"rerank_format must be one of {sorted(ALLOWED_RERANK_FORMATS)}, got {rerank_format}")
    if rerank_format == "query_first" and data_args.query_first is False:
        rerank_format = "document_first"
    return rerank_format

def configure_attention_implementations(
    config,
    text_attn_implementation: str = "eager",
    vision_attn_implementation: str = "flash_attention_2",
    # vision_attn_implementation: str = "eager",
) -> None:
    # The text stack must stay on the custom eager path because rerank masks are
    # applied inside our attention function. The vision stack does not need that
    # custom mask, so it can safely use FlashAttention-2 for lower memory/latency.
    config._attn_implementation = text_attn_implementation
    config.text_config._attn_implementation = text_attn_implementation
    config.vision_config._attn_implementation = vision_attn_implementation

def get_batch_category(subset_name: Any) -> str:
    subset_name = normalize_text(subset_name)
    if subset_name in {"ViDoRe", "VisRAG"}:
        return "VisDoc"
    if subset_name.startswith("ShareGPTVideo"):
        return "video"
    return "MMEB"

def downsample_video_dataset(dataset, keep_ratio: float, seed: int = 42):
    if keep_ratio is None or keep_ratio >= 1.0:
        return dataset
    if keep_ratio <= 0:
        raise ValueError(f"video_sample_ratio must be in (0, 1], got {keep_ratio}")
    if "subset" not in getattr(dataset, "column_names", []):
        print("Skip video downsampling: dataset has no subset column")
        return dataset

    subset_names = dataset["subset"]
    video_indices = [
        idx for idx, subset_name in enumerate(subset_names)
        if get_batch_category(subset_name) == "video"
    ]
    if len(video_indices) == 0:
        print("Skip video downsampling: no video examples found")
        return dataset

    keep_n = max(1, int(len(video_indices) * keep_ratio))
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampled_video_indices = [
        video_indices[idx]
        for idx in torch.randperm(len(video_indices), generator=generator).tolist()[:keep_n]
    ]
    video_index_set = set(video_indices)
    non_video_indices = [
        idx for idx in range(len(dataset))
        if idx not in video_index_set
    ]
    sampled_indices = non_video_indices + sampled_video_indices
    print(
        "Video downsampling: "
        f"kept {keep_n}/{len(video_indices)} video examples "
        f"(ratio={keep_ratio}), total={len(sampled_indices)}/{len(dataset)}"
    )
    return dataset.select(sampled_indices).shuffle(seed=seed)

class CategoryBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset, batch_size: int, drop_last: bool, seed: int = 42):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.category_to_indices = self._build_category_indices(dataset)
        self.num_batches = self._count_batches()

    def _build_category_indices(self, dataset) -> Dict[str, List[int]]:
        if "subset" in getattr(dataset, "column_names", []):
            subset_names = dataset["subset"]
        else:
            subset_names = ["MMEB"] * len(dataset)

        category_to_indices = {"VisDoc": [], "video": [], "MMEB": []}
        for idx, subset_name in enumerate(subset_names):
            category_to_indices[get_batch_category(subset_name)].append(idx)
        return {key: value for key, value in category_to_indices.items() if len(value) > 0}

    def _count_batches(self) -> int:
        total = 0
        for indices in self.category_to_indices.values():
            full, remainder = divmod(len(indices), self.batch_size)
            total += full
            if remainder > 0 and not self.drop_last:
                total += 1
        return total

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        self.epoch += 1

        batches = []
        for indices in self.category_to_indices.values():
            order = torch.randperm(len(indices), generator=generator).tolist()
            shuffled = [indices[idx] for idx in order]
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start:start + self.batch_size]
                if len(batch) == self.batch_size or (len(batch) > 0 and not self.drop_last):
                    batches.append(batch)

        order = torch.randperm(len(batches), generator=generator).tolist()
        for batch_idx in order:
            yield batches[batch_idx]

    def __len__(self) -> int:
        return self.num_batches

    def summary(self) -> str:
        return ", ".join(
            f"{category}={len(indices)}"
            for category, indices in sorted(self.category_to_indices.items())
        )

class PerSampleLossTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._yes_loss_sum = 0.0
        self._yes_count = 0
        self._no_loss_sum = 0.0
        self._no_count = 0

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        labels = inputs.pop("labels")
        shift_labels = labels[..., 1:].contiguous()
        mask = shift_labels != -100
        batch_indices, target_positions = torch.nonzero(mask, as_tuple=True)
        if target_positions.numel() == 0:
            raise ValueError("No supervised answer tokens found in labels.")

        # Only "yes"/"no" answer positions are supervised. Computing logits for
        # every prompt/document/vision token creates a huge [B, L, vocab] tensor
        # and was the source of late-step OOMs, so we ask the model for just the
        # logit positions that correspond to non-ignored labels.
        unique_target_positions = torch.unique(target_positions)
        outputs = model(
            **inputs,
            logits_to_keep=unique_target_positions.to(inputs["input_ids"].device),
        )
        logits = outputs.logits
        target_offsets = torch.searchsorted(unique_target_positions, target_positions)
        target_logits = logits[batch_indices, target_offsets]
        target_labels = shift_labels[batch_indices, target_positions]

        loss_per_target = F.cross_entropy(
            target_logits.float(),
            target_labels,
            reduction="none",
        )
        loss_sum = torch.zeros(labels.shape[0], device=loss_per_target.device, dtype=loss_per_target.dtype)
        token_count = torch.zeros(labels.shape[0], device=loss_per_target.device, dtype=loss_per_target.dtype)
        loss_sum.scatter_add_(0, batch_indices, loss_per_target)
        token_count.scatter_add_(0, batch_indices, torch.ones_like(loss_per_target))
        loss_per_sample = loss_sum / token_count.clamp(min=1)
        active_samples = token_count > 0

        sample_labels = [
            self.processing_class.decode(shift_labels[i][mask[i]]).strip().lower()
            for i in range(len(loss_per_sample))
        ]

        if self.args.local_rank in [-1, 0]:
            detached_losses = loss_per_sample.detach().float().cpu().tolist()
            for lbl, loss_value in zip(sample_labels, detached_losses):
                if lbl.startswith("yes"):
                    self._yes_loss_sum += loss_value
                    self._yes_count += 1
                elif lbl.startswith("no"):
                    self._no_loss_sum += loss_value
                    self._no_count += 1

        loss = loss_per_sample[active_samples].mean()
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self.args.local_rank in [-1, 0]:
            if self._yes_count > 0:
                logs["train/yes_loss"] = self._yes_loss_sum / self._yes_count
            if self._no_count > 0:
                logs["train/no_loss"] = self._no_loss_sum / self._no_count

            self._yes_loss_sum = 0.0
            self._yes_count = 0
            self._no_loss_sum = 0.0
            self._no_count = 0

        super().log(logs, start_time=start_time)

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if hasattr(train_dataset, "column_names"):
            dataloader_dataset = self._remove_unused_columns(train_dataset, description="Training")
        else:
            dataloader_dataset = train_dataset
            data_collator = self._get_collator_with_removed_columns(data_collator, description="Training")

        batch_sampler = CategoryBatchSampler(
            train_dataset,
            batch_size=self._train_batch_size,
            drop_last=self.args.dataloader_drop_last,
            seed=self.args.seed,
        )
        if self.args.process_index == 0:
            print(
                "Using category-grouped batches: "
                f"{batch_sampler.summary()}, batches={len(batch_sampler)}"
            )

        dataloader_params = {
            "batch_sampler": batch_sampler,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }
        if self.args.dataloader_prefetch_factor is not None:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        if self.args.dataloader_num_workers > 0:
            dataloader_params["worker_init_fn"] = partial(
                seed_worker,
                num_workers=self.args.dataloader_num_workers,
                rank=self.args.process_index,
            )

        return self.accelerator.prepare(DataLoader(dataloader_dataset, **dataloader_params))

@dataclass
class Qwen3VLRerankCollator:
    processor: AutoProcessor
    max_length: int = 10240
    min_pixels: int = 4 * 32 * 32
    image_max_pixels: int = 1800 * 32 * 32
    video_frame_max_pixels: int = 768 * 32 * 32
    video_total_pixels: int = 10 * 768 * 32 * 32
    fps: int = 1
    max_video_frames: int = 64
    video_root: Optional[str] = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    rerank_format: str = "query_first"
    k: int = -1
    k_selection: Optional[str] = None
    prune_k: Optional[float] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompt_texts = []
        full_texts = []
        prompt_messages_list = []
        full_messages_list = []
        early_segment_spans = []

        for ex in features:
            full_messages = build_rerank_messages_from_example(
                ex,
                system_prompt=self.system_prompt,
                rerank_format=self.rerank_format,
                video_root=self.video_root,
                max_frames=self.max_video_frames,
                min_pixels=self.min_pixels,
                image_max_pixels=self.image_max_pixels,
                video_frame_max_pixels=self.video_frame_max_pixels,
                video_total_pixels=self.video_total_pixels,
                fps=self.fps,
                include_label=True,
            )
            prompt_messages = full_messages[:-1]
            prompt_messages_list.append(prompt_messages)
            full_messages_list.append(full_messages)

            prompt_text = self.processor.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_text = self.processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            early_segment_span, _ = compute_early_segment_span_from_messages(self.processor, prompt_messages)
            early_segment_spans.append(early_segment_span)

        batch_full = self.processor(
            text=full_texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            do_resize=False,
            **compact_processor_kwargs(build_processor_vision_kwargs(full_messages_list, self.processor)),
        )

        batch_prompt = self.processor(
            text=prompt_texts,
            padding=True,
            truncation=False,
            return_tensors="pt",
            do_resize=False,
            **compact_processor_kwargs(build_processor_vision_kwargs(prompt_messages_list, self.processor)),
        )

        input_ids = batch_full["input_ids"]
        if input_ids.shape[1] > self.max_length:
            raise ValueError(
                f"Batch sequence length {input_ids.shape[1]} exceeds max_length={self.max_length}. "
                "Automatic multimodal truncation is not applied because cutting vision tokens can "
                "desynchronize input_ids from pixel/video tensors."
            )
        attention_mask = batch_full["attention_mask"]
        labels = torch.full_like(input_ids, -100)

        prefix_lens = batch_prompt["attention_mask"].sum(dim=1).tolist()

        for idx, ex in enumerate(features):
            answer_text = normalize_text(ex.get("label", "")).lower()
            answer_ids = self.processor.tokenizer(
                answer_text,
                add_special_tokens=False,
                return_attention_mask=False,
            )["input_ids"]

            prefix_len = int(prefix_lens[idx])
            answer_len = len(answer_ids)
            end_pos = min(prefix_len + answer_len, input_ids.shape[1])
            labels[idx, prefix_len:end_pos] = input_ids[idx, prefix_len:end_pos]

        batch = apply_token_pruning_to_batch(
            batch_full,
            labels,
            early_segment_spans,
            features,
            prune_k=self.prune_k,
            rerank_format=self.rerank_format,
            processor=self.processor,
            k=self.k,
            k_selection=self.k_selection,
        )

        for key, value in batch_full.items():
            if key not in batch:
                batch[key] = value

        return batch

def save_final_training_checkpoint(trainer: Trainer) -> None:
    barrier_if_distributed()
    if not (trainer.args.should_save and trainer.is_world_process_zero()):
        barrier_if_distributed()
        return
    trainer._save_checkpoint(trainer.model, trial=None)
    save_model_config(trainer.model, os.path.join(trainer.args.output_dir, f"checkpoint-{trainer.state.global_step}"))
    barrier_if_distributed()

def train(attn_implementation="eager"):
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = AutoConfig.from_pretrained(model_args.model_path)
    model_args.k, model_args.k_selection = resolve_k_config(model_args)
    early_exit_layer, no_interaction_layer = resolve_masking_schedule(config, model_args)
    model_args.early_exit_layer = early_exit_layer
    model_args.no_interaction_layer = no_interaction_layer
    model_args.full_document_mask_layer = no_interaction_layer
    data_args.rerank_format = resolve_rerank_format(data_args)
    config.k = model_args.k
    config.k_selection = model_args.k_selection
    config.early_exit_layer = early_exit_layer
    config.no_interaction_layer = no_interaction_layer
    config.full_document_mask_layer = no_interaction_layer
    config.rerank_format = data_args.rerank_format
    config.prune_k = data_args.prune_k
    config.minireranker_vision = {key: getattr(data_args, key) for key in ("max_length", "min_pixels", "max_pixels", "video_frame_max_pixels", "video_total_pixels", "fps", "max_video_frames")}
    if hasattr(config, "last_k"):
        delattr(config, "last_k")
    configure_attention_implementations(
        config,
        text_attn_implementation=model_args.text_attn_implementation,
        vision_attn_implementation=model_args.vision_attn_implementation,
        # vision_attn_implementation="eager",
    )

    all_args = {
        **asdict(model_args),
        **asdict(data_args),
        **asdict(training_args),
    }
    if training_args.process_index == 0:
        os.makedirs(training_args.output_dir, exist_ok=True)
        with open(f"{training_args.output_dir}/args.json", "w") as f:
            json.dump(all_args, f, indent=2)
    barrier_if_distributed()

    parquet_files = collect_parquet_files(data_args.data_root, data_args.subsets, data_args.train_glob)
    if len(parquet_files) == 0:
        raise ValueError(f"No parquet found under {data_args.data_root} with pattern {data_args.train_glob}")

    print(f"Found {len(parquet_files)} parquet shards")

    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=data_args.cache_training_split or ".cache/datasets",
    )
    dataset = downsample_video_dataset(
        dataset,
        keep_ratio=data_args.video_sample_ratio,
        seed=training_args.seed,
    )

    if data_args.debug_sample_ratio is not None and data_args.debug_sample_ratio < 1.0:
        subset_names = sorted(set(dataset["subset"]))
        sampled_parts = []

        for subset_name in subset_names:
            subset_ds = dataset.filter(lambda x: x["subset"] == subset_name)
            orig_n = len(subset_ds)

            keep_n = max(1, int(orig_n * data_args.debug_sample_ratio))
            subset_ds = subset_ds.shuffle(seed=42).select(range(keep_n))

            positive_ds = subset_ds.filter(lambda x: x["candidate_type"] == "positive")
            hard_neg_ds = subset_ds.filter(lambda x: x["candidate_type"] == "hard_negative")
            rand_neg_ds = subset_ds.filter(lambda x: x["candidate_type"] == "random_negative")

            if len(hard_neg_ds) > 0 and data_args.negative_keep_ratio < 1.0:
                keep_hard = max(1, int(len(hard_neg_ds) * data_args.negative_keep_ratio))
                hard_neg_ds = hard_neg_ds.shuffle(seed=42).select(range(keep_hard))
            else:
                keep_hard = len(hard_neg_ds)

            if len(rand_neg_ds) > 0 and data_args.negative_keep_ratio < 1.0:
                keep_rand = max(1, int(len(rand_neg_ds) * data_args.negative_keep_ratio))
                rand_neg_ds = rand_neg_ds.shuffle(seed=42).select(range(keep_rand))
            else:
                keep_rand = len(rand_neg_ds)

            subset_ds = concatenate_datasets([positive_ds, hard_neg_ds, rand_neg_ds]).shuffle(seed=42)
            sampled_parts.append(subset_ds)

            print(
                f"{subset_name}: "
                f"subset_debug_keep={keep_n}/{orig_n}, "
                f"positive={len(positive_ds)}, "
                f"hard_negative={keep_hard}, "
                f"random_negative={keep_rand}, "
                f"final={len(subset_ds)}"
            )

        dataset = concatenate_datasets(sampled_parts).shuffle(seed=42)

    print(dataset)

    processor = AutoProcessor.from_pretrained(model_args.model_path)
    processor.tokenizer.model_max_length = data_args.max_length

    model = CrossAttnMasking_Qwen3_VLForConditionalGeneration.from_pretrained(
        model_args.model_path,
        config=config,
        k=model_args.k,
        k_selection=model_args.k_selection,
        early_exit_layer=model_args.early_exit_layer,
        no_interaction_layer=model_args.no_interaction_layer,
        rerank_format=data_args.rerank_format,
        dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
    )

    freeze_vision_and_projector(model, freeze_projector=model_args.freeze_projector)

    if training_args.use_lora:
        if _PEFT_IMPORT_ERROR is not None:
            raise RuntimeError(
                "peft is required when --use_lora True. Install peft in the active environment."
            ) from _PEFT_IMPORT_ERROR
        peft_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            lora_dropout=training_args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    apply_model_config_values(
        model,
        k=model_args.k,
        k_selection=model_args.k_selection,
        early_exit_layer=model_args.early_exit_layer,
        no_interaction_layer=model_args.no_interaction_layer,
        rerank_format=data_args.rerank_format,
    )
    model.config.use_cache = False
    get_model_config(model).use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    collator = Qwen3VLRerankCollator(
        processor=processor,
        max_length=data_args.max_length,
        min_pixels=data_args.min_pixels,
        image_max_pixels=data_args.max_pixels,
        video_frame_max_pixels=data_args.video_frame_max_pixels,
        video_total_pixels=data_args.video_total_pixels,
        fps=data_args.fps,
        max_video_frames=data_args.max_video_frames,
        video_root=infer_video_root(data_args.data_root, data_args.video_root),
        system_prompt=data_args.system_prompt,
        rerank_format=data_args.rerank_format,
        k=model_args.k,
        k_selection=model_args.k_selection,
        prune_k=data_args.prune_k,
    )

    trainer = PerSampleLossTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    trainer.train(resume_from_checkpoint=getattr(training_args, "resume_from_checkpoint", None))
    barrier_if_distributed()

    if trainer.args.should_save:
        if trainer.is_world_process_zero():
            apply_model_config_values(
                model,
                k=model_args.k,
                k_selection=model_args.k_selection,
                early_exit_layer=model_args.early_exit_layer,
                no_interaction_layer=model_args.no_interaction_layer,
                rerank_format=data_args.rerank_format,
            )

        trainer.save_model(training_args.output_dir)

        if trainer.is_world_process_zero():
            save_model_config(model, training_args.output_dir)
            processor.save_pretrained(training_args.output_dir)
    barrier_if_distributed()

if __name__ == "__main__":
    train(attn_implementation="eager")
