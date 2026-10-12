---
language:
- en
task_categories:
- text-retrieval
tags:
- multimodal
- reranking
- token-pruning
size_categories:
- 100K<n<1M
configs:
- config_name: default
  default: true
  data_files:
  - split: train
    path: "*/original-*.parquet"
- config_name: A-OKVQA
  data_files:
  - split: train
    path: "A-OKVQA/original-*.parquet"
- config_name: CIRR
  data_files:
  - split: train
    path: "CIRR/original-*.parquet"
- config_name: ChartQA
  data_files:
  - split: train
    path: "ChartQA/original-*.parquet"
- config_name: DocVQA
  data_files:
  - split: train
    path: "DocVQA/original-*.parquet"
- config_name: HatefulMemes
  data_files:
  - split: train
    path: "HatefulMemes/original-*.parquet"
- config_name: ImageNet_1K
  data_files:
  - split: train
    path: "ImageNet_1K/original-*.parquet"
- config_name: InfographicsVQA
  data_files:
  - split: train
    path: "InfographicsVQA/original-*.parquet"
- config_name: MSCOCO
  data_files:
  - split: train
    path: "MSCOCO/original-*.parquet"
- config_name: MSCOCO_i2t
  data_files:
  - split: train
    path: "MSCOCO_i2t/original-*.parquet"
- config_name: MSCOCO_t2i
  data_files:
  - split: train
    path: "MSCOCO_t2i/original-*.parquet"
- config_name: N24News
  data_files:
  - split: train
    path: "N24News/original-*.parquet"
- config_name: NIGHTS
  data_files:
  - split: train
    path: "NIGHTS/original-*.parquet"
- config_name: OK-VQA
  data_files:
  - split: train
    path: "OK-VQA/original-*.parquet"
- config_name: SUN397
  data_files:
  - split: train
    path: "SUN397/original-*.parquet"
- config_name: ShareGPTVideo_caption_t2v
  data_files:
  - split: train
    path: "ShareGPTVideo_caption_t2v/original-*.parquet"
- config_name: ShareGPTVideo_caption_v2t
  data_files:
  - split: train
    path: "ShareGPTVideo_caption_v2t/original-*.parquet"
- config_name: ShareGPTVideo_qa_v2t
  data_files:
  - split: train
    path: "ShareGPTVideo_qa_v2t/original-*.parquet"
- config_name: VOC2007
  data_files:
  - split: train
    path: "VOC2007/original-*.parquet"
- config_name: ViDoRe
  data_files:
  - split: train
    path: "ViDoRe/original-*.parquet"
- config_name: VisDial
  data_files:
  - split: train
    path: "VisDial/original-*.parquet"
- config_name: VisRAG
  data_files:
  - split: train
    path: "VisRAG/original-*.parquet"
- config_name: Visual7W
  data_files:
  - split: train
    path: "Visual7W/original-*.parquet"
- config_name: VisualNews_i2t
  data_files:
  - split: train
    path: "VisualNews_i2t/original-*.parquet"
- config_name: VisualNews_t2i
  data_files:
  - split: train
    path: "VisualNews_t2i/original-*.parquet"
- config_name: WebQA
  data_files:
  - split: train
    path: "WebQA/original-*.parquet"
---
# miniReranker multimodal training pairs

This dataset supports supervised point-wise multimodal reranking and embedder-guided
visual token pruning for **miniReranker: Efficient Multimodal Reranking through Visual
Cache Reuse and Interaction Sparsity**.

Hub repository: [Mickey25/miniReranker_736k](https://huggingface.co/datasets/Mickey25/miniReranker_736k).
Code: [EIT-NLP/miniRanker](https://github.com/EIT-NLP/miniRanker).
The release contains **736,327 query-document pairs** across 25 subsets and 66 Parquet shards.

## Composition

There are 25 subsets and 66 Parquet shards, with one `train` split. Counts refer to
point-wise pairs, not unique queries, images, videos, or documents. No separate
validation/test split is supplied. Benchmark evaluation uses external data.

| Candidate type | Label | Pairs |
|---|---|---:|
| positive | yes | 147,899 |
| hard_negative | no | 294,375 |
| random_negative | no | 294,053 |
| **Total** | | **736,327** |

Each positive was paired with up to four negatives: two hard and two random negatives.
According to the paper, hard negatives were mined with Qwen3-VL-Embedding-2B from
Top-20 image/document retrieval candidates and Top-5 video retrieval candidates.
Actual retained counts are listed below.

| Subset/configuration | Retrieval direction | Pairs | Shards |
|---|---|---:|---:|
| A-OKVQA | Image + text → Text | 8,525 | 1 |
| CIRR | Image + text → Image | 13,055 | 1 |
| ChartQA | Image + text → Text | 14,145 | 1 |
| DocVQA | Image + text → Text | 19,730 | 1 |
| HatefulMemes | Image → Text | 1,700 | 1 |
| ImageNet_1K | Image → Text | 50,000 | 3 |
| InfographicsVQA | Image + text → Text | 11,970 | 1 |
| MSCOCO | Image → Image | 50,000 | 3 |
| MSCOCO_i2t | Image → Text | 56,640 | 3 |
| MSCOCO_t2i | Text → Image | 50,000 | 3 |
| N24News | Image → Text | 24,490 | 2 |
| NIGHTS | Image + text → Image | 7,970 | 1 |
| OK-VQA | Image + text → Text | 4,500 | 1 |
| SUN397 | Image → Text | 9,925 | 1 |
| ShareGPTVideo_caption_t2v | Text → Video | 29,971 | 8 |
| ShareGPTVideo_caption_v2t | Video → Text | 29,998 | 8 |
| ShareGPTVideo_qa_v2t | Video + text → Text | 24,243 | 6 |
| VOC2007 | Image → Text | 3,920 | 1 |
| ViDoRe | Text → Document image | 59,095 | 3 |
| VisDial | Text → Image + text | 61,640 | 4 |
| VisRAG | Text → Document image | 61,375 | 4 |
| Visual7W | Image + text → Text | 34,905 | 2 |
| VisualNews_i2t | Image → Text | 50,000 | 3 |
| VisualNews_t2i | Text → Image + text | 49,950 | 3 |
| WebQA | Text → Image + text | 8,580 | 1 |

## Storage and fields

Each subset contains `original-*.parquet` with the same Arrow schema.

| Fields | Type | Meaning |
|---|---|---|
| `subset` | string | Source subset identifier, matching its configuration name. |
| `source_row_idx` | int64 | Row index in the source subset; not a globally unique pair ID. |
| `candidate_type` | string | `positive`, `hard_negative`, or `random_negative`. |
| `label` | string | Point-wise relevance target: `yes` or `no`. |
| `instruction` | string | Task-specific reranking instruction. |
| `query_text`, `document_text` | string | Text on either side; may be empty. |
| `query_image_bytes`, `document_image_bytes` | binary, nullable | Encoded image bytes; no external image download is needed for these fields. |
| `doc_modality` | string | Document-side modality metadata; examples include `text`, `image`, `image+text`, `video`. |
| `query_video_frame_paths`, `document_video_frame_paths` | list[string] | Ordered paths to external video frames; empty for non-video sides. |
| `query_visual_modalities`, `document_visual_modalities` | list[string] | Visual input types, e.g. `image` or `video`, in media order on that side. |
| `query_visual_token_counts`, `document_visual_token_counts` | list[int32] | Number of visual tokens per media input before pruning. |
| `query_visual_ranked_tokens`, `document_visual_ranked_tokens` | list[list[int32]] | Descending importance ranking as zero-based `[media_index, token_index]` pairs. |

Rankings are computed separately for the query and document. For example,
`[[0, 12], [0, 3]]` means token 12 of media input 0 is more important than token 3.
The `media_index` refers to a visual input on that side, not a query/document row or
necessarily an individual video frame. Nonvisual sides have empty ranking lists.
The training code selects high-ranked tokens and restores their original sequence
order, keeping at least one token for each nonempty media input.

The inspected preprocessing used Qwen3-VL-Embedding-2B attention, summed over all
layers from the final embedding token to visual tokens (with head averaging), seed 42,
maximum input length 10,240, image budgets 4,096–1,843,200 pixels, video frame budget
786,432 pixels, total video budget 7,864,320 pixels, at most 64 frames, and FPS metadata
2.0. Video rows were retained with probability 0.2 during this preprocessing stage.
**Do not apply this 0.2 sampling again when training on this release.** Training's
default video sampling ratio is 1.0. Training/evaluation FPS metadata defaults to 1;
retain the stored frame ordering and matching pixel budgets so token counts agree.
If geometry differs, recompute rankings rather than applying stale token indices.

## Load a subset or the complete collection

Load the published dataset:

```python
from datasets import load_dataset
all_pairs = load_dataset("Mickey25/miniReranker_736k", split="train")
video_pairs = load_dataset("Mickey25/miniReranker_736k",
                           "ShareGPTVideo_caption_t2v", split="train")
```

Local loading and image decoding:

```python
from datasets import load_dataset
from PIL import Image
from io import BytesIO
pairs = load_dataset("parquet", data_files={"train": "data/miniReranker_736k/VisRAG/original-*.parquet"},
                     split="train")
row = pairs[0]
image = Image.open(BytesIO(row["document_image_bytes"])).convert("RGB")
```

## Download video frames separately

The Parquet files contain **paths only**, not video files or frame bytes.
Download [ShareGPTVideo train_300k](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction/tree/main/train_300k),
which the [upstream dataset card](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction)
describes as the video-frame subset used for SFT/DPO. The three video configurations
are caption text-to-video, caption video-to-text, and video question/answer-to-text.

```bash
hf download ShareGPTVideo/train_video_and_instruction --repo-type dataset \
  --include 'train_300k/*' --local-dir ./downloads/ShareGPTVideo
```

Extract the downloaded archives while preserving their relative directory hierarchy.
Set the root to the directory **containing** `train_300k`, not `train_300k` itself:

```text
data/ShareGPTVideo/                 <- video_root
└── train_300k/
    └── <video_id>/
        ├── c01_0001.jpeg
        └── c01_0002.jpeg
```

For a stored path `train_300k/<video_id>/c01_0001.jpeg`, the loader resolves
`video_root / stored_path`. Configure `--video_root ./data/ShareGPTVideo`,
`SHAREGPTVIDEO_ROOT`, or `VIDEO_ROOT`. Missing frames raise an error.

## Upload preparation and source terms

Upload only this README and the 25 subset folders' `original-*.parquet` files.
Exclude `training_split_cache/`, `inspection.ipynb`, and `manifest.worker*.json`:
the manifests contain machine-specific paths and are not needed to load the dataset.
Do not upload ShareGPTVideo frames as part of this repository.

The collection contains derived examples from the named source datasets. Consult
the original source terms before redistributing or using each subset; the code's
Apache-2.0 license does not replace their licenses. This card does not assign one
blanket license to the underlying images, documents or video frames. Dataset names
are provenance labels, and a pair count does not describe each original dataset's size.

Relevant upstream collections: [MMEB/VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec),
[ViDoRe](https://huggingface.co/vidore), [VisRAG](https://github.com/OpenBMB/VisRAG),
and [ShareGPTVideo](https://huggingface.co/datasets/ShareGPTVideo/train_video_and_instruction).
See the code repository's paper for construction and task details.
