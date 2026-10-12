from dataclasses import dataclass
from typing import Optional

import transformers


@dataclass
class ModelArguments:
    model_path: str
    text_attn_implementation: str = "eager"
    vision_attn_implementation: str = "sdpa"
    freeze_projector: bool = False
    k: int = -1
    k_selection: Optional[str] = None
    last_k: Optional[int] = None
    early_exit_layer: Optional[int] = None
    no_interaction_layer: Optional[int] = None
    # Backward-compatible alias for older scripts/checkpoints.
    full_document_mask_layer: Optional[int] = None


@dataclass
class DataArguments:
    data_root: str
    cache_training_split: Optional[str] = None
    subsets: Optional[str] = None
    train_glob: str = "original-*.parquet"
    debug_sample_ratio: float = 1.0
    negative_keep_ratio: float = 1.0
    video_sample_ratio: float = 1.0
    prune_k: Optional[float] = None
    system_prompt: str = (
        'Judge whether the Document meets the requirements based on the Query and '
        'the Instruct provided. Note that the answer can only be "yes" or "no".'
    )
    max_length: int = 10240
    min_pixels: int = 4 * 32 * 32
    max_pixels: int = 1800 * 32 * 32
    video_frame_max_pixels: int = 768 * 32 * 32
    video_total_pixels: int = 10 * 768 * 32 * 32
    fps: int = 1
    max_video_frames: int = 64
    video_root: Optional[str] = None
    rerank_format: str = "query_first"
    query_first: bool = True


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str

    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
