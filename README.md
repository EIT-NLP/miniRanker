# miniReranker

**miniReranker: Efficient Multimodal Reranking through Visual Cache Reuse and Interaction Sparsity**

This repository provides Transformers-based multimodal reranker training and inference
with vision-first prompting, early exit, interaction bands, and embedder-guided visual
token pruning. The repository directory is named `miniRanker`; the method and Python
package are `miniReranker` and `minireranker`.

- [Paper](assets/paper.pdf)
- [Training dataset](https://huggingface.co/datasets/Mickey25/miniReranker_736k)
- [Dataset card and subset inventory](docs/dataset_card.md)
- [Evaluation setup](docs/evaluation.md)

## Installation

Use Python 3.11. The reproducibility environment pins PyTorch 2.8.0 and Transformers
4.57.6; a CUDA-capable GPU is recommended for the released Qwen3-VL checkpoints.
The default language attention is eager and vision attention is SDPA.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .                 # inference
pip install -e '.[train,data]'    # training and optional rank preprocessing
pip install -e '.[eval,dev]'      # evaluation adapter and tests
```

Only use trusted pretrained checkpoints. Models may be specified as a Hugging Face
ID or a local checkpoint directory. Credentials, if needed, belong in your environment
or normal Hugging Face login configuration.

## Prepare the data

The training dataset is available at [Mickey25/miniReranker_736k](https://huggingface.co/datasets/Mickey25/miniReranker_736k).
The inspected release has **736,327 pairs across 25 subsets and 66 shards**. Images
are embedded as bytes; videos require external frames. See the dataset card for exact
counts, fields, source terms and subset loading examples.

```bash
hf download Mickey25/miniReranker_736k --repo-type dataset --local-dir ./data/miniReranker_736k
```

Download [ShareGPTVideo train_300k](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k)
and extract it under `./data/ShareGPTVideo/train_300k/...`. Set `VIDEO_ROOT` to
`./data/ShareGPTVideo`. The distributed Parquet data already contains visual token
rankings; preprocessing is only needed when building or changing your own dataset.

## Train

```bash
DATA_ROOT=./data/miniReranker_736k VIDEO_ROOT=./data/ShareGPTVideo \
  bash scripts/train_2B.sh
```

The scripts use one GPU and batch size 1 by default. To reproduce the paper's training
configuration, use four GPUs, per-device batch size 4, and ZeRO-2:

```bash
NPROC_PER_NODE=4 BATCH_SIZE=4 USE_DEEPSPEED=1 \
  DATA_ROOT=./data/miniReranker_736k VIDEO_ROOT=./data/ShareGPTVideo \
  bash scripts/train_2B.sh
```

The paper used four A100 40GB GPUs for 2B and four H100 GPUs for 4B/8B. Training uses
one epoch, learning rate 1e-4, zero weight decay, cosine decay, warmup 0.03, bf16,
and LoRA r=16, alpha=32, dropout=0.05. The vision encoder is frozen. Batch size and
visual budgets affect memory use; one GPU is the starter configuration, not the paper's
training-throughput configuration.

| Script | Base model | `early_exit_layer` | `no_interaction_layer` | `prune_k` |
|---|---|---:|---:|---:|
| `train_2B.sh` | Qwen/Qwen3-VL-2B-Instruct | 16 | 8 | 0.5 |
| `train_4B.sh` | Qwen/Qwen3-VL-4B-Instruct | 21 | 8 | 0.5 |
| `train_8B.sh` | Qwen/Qwen3-VL-8B-Instruct | 21 | 8 | 0.5 |

Layer arguments use **zero-based indices**: exit at index 16 executes 17 decoder layers.
The model applies its full cross-segment mask from `no_interaction_layer` onward and
exits after `early_exit_layer`. `-1` disables the interaction restriction. The legacy
`full_document_mask_layer` checkpoint field remains supported.

`prune_k` is a keep budget: `(0, 1]` keeps that fraction (1.0 keeps all tokens),
values greater than 1 keep that many tokens, and omitting it from training disables
pruning. At least one token is retained per nonempty media input. `image_first` is
the vision-first format: query first when only the query is visual; otherwise document
first. Query-first and document-first formats are also supported.

Environment overrides include `MODEL_PATH`, `OUTPUT_DIR`, `EARLY_EXIT_LAYER`,
`NO_INTERACTION_LAYER`, `PRUNE_K`, `NPROC_PER_NODE`, `BATCH_SIZE`, `DATA_WORKERS`,
and `RESUME_FROM_CHECKPOINT`. Additional training arguments can follow the script:

```bash
bash scripts/train_2B.sh --subsets VisRAG,ViDoRe --max_steps 10
```

Model outputs default to `./checkpoints/miniReranker-2B/lora`. Merge LoRA before using
the standard inference example:

```bash
python -m minireranker.merge_lora \
  --lora-path ./checkpoints/miniReranker-2B/lora \
  --base-model-path Qwen/Qwen3-VL-2B-Instruct \
  --output-dir ./checkpoints/miniReranker-2B/merged
```

The merged checkpoint retains the architecture and preprocessing configuration.
To build rankings for a new Parquet dataset, set `INPUT_ROOT`, `OUTPUT_ROOT`,
`EMBEDDER_MODEL_PATH` and optionally `VIDEO_ROOT`, then run `scripts/precompute.sh`.
Use a separate output directory. The default retains all video rows; use
`--video_sample_ratio 0.2` explicitly only when creating the paper's sampled release.

## Inference

```python
from minireranker import MiniReranker

reranker = MiniReranker(
    "./checkpoints/miniReranker-2B/merged",
    embedder_model_name_or_path="Qwen/Qwen3-VL-Embedding-2B",
)
scores = reranker.process({
    "instruction": "Find the image that matches the description.",
    "query": {"text": "A dog playing outdoors"},
    "documents": [{"image": "./examples/dog.jpg"}, {"image": "./examples/cat.jpg"}],
})
```

Scores are raw `yes - no` logits, with larger values indicating greater relevance.
The list preserves document input order and is not a probability. Each side supports
text, one image (path/bytes/PIL image), and/or an ordered list of video frame paths.
For video, use `{"video": ["train_300k/<id>/c01_0001.jpeg", ...]}` and configure
`video_root`. Raw video files should be converted to frames before using this API.

The loader reads saved prompt, layer, pruning and preprocessing settings. For older
checkpoints without a saved pruning rate, pass `prune_k=0.5` explicitly. Visual rankings
are computed online when needed; text-only inputs do not load an embedder.
`process_examples(rows)` accepts the dataset's pair schema and reuses supplied rankings.

```bash
python examples/rerank.py --model ./checkpoints/miniReranker-2B/merged
```

## Evaluation

Follow the [Qwen3-VL-Embedding evaluation code](https://github.com/QwenLM/Qwen3-VL-Embedding)
for datasets, embedding predictions and metrics, then use the
[pinned-version adapter and patch](docs/evaluation.md).
Shuffle candidates within each query to avoid dependence on embedding order when
scores tie, and omit sigmoid to avoid finite-precision saturation. Preserve the original
retrieved TopK; do not insert ground truth when retrieval misses it.

## Tests and citation

```bash
python -m pytest
```

Please cite the accompanying paper:

```bibtex
@misc{fan2026minireranker,
  title={miniReranker: Efficient Multimodal Reranking through Visual Cache Reuse and Interaction Sparsity},
  author={Fan, Yingqi and Lu, Xuan and Zhao, Anhao and Tong, Junlong and Nie, Ping and Zou, Kai and Ma, Yunpu and Zhang, Wei and Shen, Xiaoyu},
  year={2026}
}
```

Code: [Apache-2.0](LICENSE). See [third-party notices](THIRD_PARTY_NOTICES.md) and
the [dataset source terms](docs/dataset_card.md#upload-preparation-and-source-terms).
