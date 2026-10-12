# MMEB-v2 evaluation

Follow [Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) for
evaluation datasets, embedding retrieval predictions, task YAML files and metrics.
Use a separate clean clone at the pinned revision below. Install this repository's
evaluation dependencies in the same environment (`pip install -e '.[eval]'`),
and any additional dataset dependencies required by the upstream download instructions.

```bash
# Run from the miniRanker repository root.
MINIRANKER_ROOT="$PWD"
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git third_party/Qwen3-VL-Embedding
git -C third_party/Qwen3-VL-Embedding checkout 683585a8d1a4cdedd40cd797af9410ed87ede728
git -C third_party/Qwen3-VL-Embedding apply --check "$MINIRANKER_ROOT/patches/qwen3-vl-evaluation.patch"
git -C third_party/Qwen3-VL-Embedding apply "$MINIRANKER_ROOT/patches/qwen3-vl-evaluation.patch"

python -m minireranker.evaluation \
  --upstream_repo ./third_party/Qwen3-VL-Embedding \
  --model_name_or_path ./checkpoints/miniReranker-2B/merged \
  --embedder_model_name_or_path Qwen/Qwen3-VL-Embedding-2B \
  --prune_k 0.5 --prune_method attention --prune_layer all \
  --rerank_format image_first --shuffle_seed 42 \
  --dataset_config ./third_party/Qwen3-VL-Embedding/scripts/evaluation/mmeb_v2/image.yaml \
  --encode_output_path ./outputs/embedding_predictions \
  --data_basedir ./data/MMEB-v2 \
  --rerank_output_path ./outputs/rerank --output_dir ./outputs/eval_runtime \
  --topk 20 --per_device_eval_batch_size 1
```

Use the upstream YAML matching your benchmark domain. For multi-GPU evaluation,
launch the same module with `torchrun --standalone --nproc_per_node ...`.
The adapter loads the project model through Transformers, and applies the checkpoint's
early-exit, interaction-band and visual-preprocessing settings. Explicit CLI layer
and prompt overrides are supported.

## Two required corrections

1. Shuffle the candidates **within each query**, after selecting the original TopK
   and before scoring. Shuffling queries alone does not remove candidate-order bias.
   We use a seed derived from the configured seed and the global query index before
   distributed splitting, then stably sort by score. Genuine ties therefore use the
   shuffled candidate order rather than the embedding rank. IDs and scores stay aligned.
2. Return the raw `yes - no` logit difference. Sigmoid is mathematically monotonic,
   but finite-precision saturation can map distinct large or small logits to equal
   scores. Both the supplied model and the upstream scoring patch omit sigmoid.

```python
batch_scores = self.model(**inputs).last_hidden_state[:, -1]
scores = self.score_linear(batch_scores)
return scores.squeeze(-1).float().cpu().detach().tolist()
```

The candidate set remains the embedding model's original TopK. Ground-truth documents
are never inserted when retrieval misses them. This protocol can yield different scores
from scripts that forcibly add a correct answer to the candidate set.

Result JSON includes the seed, score type, effective model settings and fingerprints
of the checkpoint, task configuration, CLI settings and embedding predictions. Cached
results are reused only when these match. Missing candidate IDs follow the upstream
skip behavior; scores with NaN or infinity raise an error.
