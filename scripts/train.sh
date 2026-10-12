#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
SIZE="${MODEL_SIZE:-2B}"
case "$SIZE" in
  2B) EXIT_LAYER=16 ;;
  4B|8B) EXIT_LAYER=21 ;;
  *) echo "MODEL_SIZE must be 2B, 4B, or 8B" >&2; exit 2 ;;
esac
ARGS=(
  --model_path "${MODEL_PATH:-Qwen/Qwen3-VL-${SIZE}-Instruct}"
  --data_root "${DATA_ROOT:-./data/miniReranker_736k}"
  --output_dir "${OUTPUT_DIR:-./checkpoints/miniReranker-${SIZE}/lora}"
  --early_exit_layer "${EARLY_EXIT_LAYER:-$EXIT_LAYER}"
  --no_interaction_layer "${NO_INTERACTION_LAYER:-8}"
  --rerank_format image_first --prune_k "${PRUNE_K:-0.5}"
  --video_sample_ratio 1.0 --remove_unused_columns False
  --eval_strategy no --num_train_epochs 1 --save_total_limit 1 --save_steps 1000
  --report_to none --logging_steps 10 --logging_first_step True
  --per_device_train_batch_size "${BATCH_SIZE:-1}"
  --gradient_accumulation_steps 1 --dataloader_num_workers "${DATA_WORKERS:-2}"
  --bf16 True --learning_rate 1e-4 --weight_decay 0 --warmup_ratio 0.03
  --lr_scheduler_type cosine --use_lora True --lora_r 16 --lora_alpha 32 --lora_dropout 0.05
)
if [[ -n "${VIDEO_ROOT:-}" ]]; then ARGS+=(--video_root "$VIDEO_ROOT"); fi
if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then ARGS+=(--resume_from_checkpoint "$RESUME_FROM_CHECKPOINT"); fi
if [[ "${USE_DEEPSPEED:-0}" == 1 ]]; then ARGS+=(--deepspeed "$REPO_ROOT/configs/zero2.json"); fi
torchrun --standalone --nproc_per_node "${NPROC_PER_NODE:-1}" -m minireranker.train "${ARGS[@]}" "$@"
