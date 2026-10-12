#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${INPUT_ROOT:?Set INPUT_ROOT to the source Parquet dataset}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a separate destination}"
ARGS=(--input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT"
      --embedder_model_name_or_path "${EMBEDDER_MODEL_PATH:-Qwen/Qwen3-VL-Embedding-2B}"
      --prune_method attention --prune_layer all --video_sample_ratio 1.0 --rank_batch_size 1)
if [[ -n "${VIDEO_ROOT:-}" ]]; then ARGS+=(--video_root "$VIDEO_ROOT"); fi
python -m minireranker.precompute_token_ranks "${ARGS[@]}" "$@"
