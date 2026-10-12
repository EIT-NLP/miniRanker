from .data_utils import infer_video_root
import argparse
import glob
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from tqdm.auto import tqdm

from .rerank_utils import ensure_pil_or_none, normalize_text, normalize_video_paths
from .token_ranker import Qwen3VLVisualTokenRanker, VisualTokenRanking




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute embedder-guided visual token rankings.")
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--embedder_model_name_or_path", required=True)
    parser.add_argument("--subsets", default=None, help="Comma-separated subset names. Defaults to all.")
    parser.add_argument("--train_glob", default="original-*.parquet")
    parser.add_argument("--video_root", default=None)
    parser.add_argument("--video_sample_ratio", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rank_batch_size", type=int, default=2)
    parser.add_argument("--prune_method", default="attention", choices=["attention", "hidden_state_similarity"])
    parser.add_argument("--prune_layer", default="all")
    parser.add_argument("--max_length", type=int, default=10240)
    parser.add_argument("--min_pixels", type=int, default=4 * 32 * 32)
    parser.add_argument("--max_pixels", type=int, default=1800 * 32 * 32)
    parser.add_argument("--video_frame_max_pixels", type=int, default=768 * 32 * 32)
    parser.add_argument("--video_total_pixels", type=int, default=10 * 768 * 32 * 32)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--max_video_frames", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit_shards", type=int, default=None)
    parser.add_argument("--limit_rows_per_shard", type=int, default=None)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of manual shard-partition workers. Each worker handles shard_idx % num_workers == worker_rank.",
    )
    parser.add_argument(
        "--worker_rank",
        type=int,
        default=0,
        help="Rank of this manual shard-partition worker.",
    )
    return parser.parse_args()


def normalize_subsets(subsets: Optional[str]) -> Optional[List[str]]:
    if subsets is None:
        return None
    items = [item.strip() for item in subsets.split(",") if item.strip()]
    return items or None


def collect_parquet_files(input_root: str, subsets: Optional[str], pattern: str) -> List[str]:
    subset_names = normalize_subsets(subsets)
    if subset_names is None:
        subset_dirs = [path for path in glob.glob(os.path.join(input_root, "*")) if os.path.isdir(path)]
    else:
        subset_dirs = [os.path.join(input_root, subset_name) for subset_name in subset_names]

    files: List[str] = []
    for subset_dir in subset_dirs:
        files.extend(sorted(glob.glob(os.path.join(subset_dir, pattern))))
    return files





def row_has_video(row: Dict[str, Any]) -> bool:
    return bool(row.get("query_video_frame_paths")) or bool(row.get("document_video_frame_paths"))


def side_has_visual(row: Dict[str, Any], side: str) -> bool:
    return (
        row.get(f"{side}_image_bytes") is not None
        or bool(row.get(f"{side}_video_frame_paths"))
    )


def sha1_bytes(value: Optional[bytes]) -> Optional[str]:
    if value is None:
        return None
    return hashlib.sha1(bytes(value)).hexdigest()


def side_cache_key(
    row: Dict[str, Any],
    side: str,
    *,
    args: argparse.Namespace,
) -> str:
    image_bytes = row.get(f"{side}_image_bytes")
    video_paths = row.get(f"{side}_video_frame_paths") or []
    payload = {
        "side": side,
        "instruction": normalize_text(row.get("instruction", "")) if side == "query" else None,
        "text": normalize_text(row.get(f"{side}_text", "")),
        "image_sha1": sha1_bytes(image_bytes),
        "video_paths": [str(path) for path in video_paths],
        "embedder": args.embedder_model_name_or_path,
        "prune_method": args.prune_method,
        "prune_layer": args.prune_layer,
        "max_length": args.max_length,
        "min_pixels": args.min_pixels,
        "max_pixels": args.max_pixels,
        "video_frame_max_pixels": args.video_frame_max_pixels,
        "video_total_pixels": args.video_total_pixels,
        "fps": args.fps,
        "max_video_frames": args.max_video_frames,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()


def build_side_sample(
    row: Dict[str, Any],
    side: str,
    *,
    video_root: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    image = ensure_pil_or_none(row.get(f"{side}_image_bytes"))
    videos = normalize_video_paths(
        row.get(f"{side}_video_frame_paths"),
        video_root=video_root,
        max_frames=args.max_video_frames,
    )
    sample = {
        "text": normalize_text(row.get(f"{side}_text", "")),
        "image": image,
        "video": videos,
        "fps": args.fps,
        "max_frames": args.max_video_frames,
    }
    if side == "query":
        sample["instruction"] = normalize_text(row.get("instruction", ""))
    return sample


def process_pending(
    ranker: Qwen3VLVisualTokenRanker,
    cache: Dict[str, VisualTokenRanking],
    pending: Dict[str, Dict[str, Any]],
    batch_size: int,
    *,
    desc: str,
) -> None:
    keys = list(pending.keys())
    num_batches = (len(keys) + batch_size - 1) // batch_size
    batch_iter = range(0, len(keys), batch_size)
    progress = tqdm(
        batch_iter,
        total=num_batches,
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    for start in progress:
        batch_keys = keys[start:start + batch_size]
        batch_samples = [pending[key] for key in batch_keys]
        rankings = ranker.build_rankings(batch_samples)
        for key, ranking in zip(batch_keys, rankings):
            cache[key] = ranking
        progress.set_postfix(cache=len(cache))


def ranking_or_empty(cache: Dict[str, VisualTokenRanking], key: Optional[str]) -> VisualTokenRanking:
    if key is None:
        return VisualTokenRanking.empty()
    return cache[key]


def output_path_for(input_path: str, input_root: str, output_root: str) -> str:
    rel_path = os.path.relpath(input_path, input_root)
    return os.path.join(output_root, rel_path)


def main() -> None:
    args = parse_args()
    if not (0 < args.video_sample_ratio <= 1):
        raise ValueError(f"video_sample_ratio must be in (0, 1], got {args.video_sample_ratio}.")
    if args.rank_batch_size <= 0:
        raise ValueError("rank_batch_size must be positive.")
    if args.num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {args.num_workers}.")
    if not (0 <= args.worker_rank < args.num_workers):
        raise ValueError(
            f"worker_rank must be in [0, {args.num_workers - 1}], got {args.worker_rank}."
        )

    video_root = infer_video_root(args.input_root, args.video_root)
    all_parquet_files = collect_parquet_files(args.input_root, args.subsets, args.train_glob)
    if args.limit_shards is not None:
        all_parquet_files = all_parquet_files[:args.limit_shards]
    parquet_files = [
        path
        for global_shard_idx, path in enumerate(all_parquet_files)
        if global_shard_idx % args.num_workers == args.worker_rank
    ]
    if not parquet_files:
        raise ValueError(
            f"No parquet files assigned to worker {args.worker_rank}/{args.num_workers} "
            f"under {args.input_root}."
        )

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    ranker = Qwen3VLVisualTokenRanker(
        model_name_or_path=args.embedder_model_name_or_path,
        prune_method=args.prune_method,
        prune_layer=args.prune_layer,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        video_frame_max_pixels=args.video_frame_max_pixels,
        total_pixels=args.video_total_pixels,
        fps=args.fps,
        max_frames=args.max_video_frames,
        torch_dtype=torch.bfloat16,
    )

    rng = random.Random(args.seed)
    cache: Dict[str, VisualTokenRanking] = {}
    stats = {
        "input_rows": 0,
        "kept_rows": 0,
        "video_rows": 0,
        "kept_video_rows": 0,
        "cache_misses": 0,
        "shards": 0,
    }

    assigned_shards = [
        (global_shard_idx, path)
        for global_shard_idx, path in enumerate(all_parquet_files)
        if global_shard_idx % args.num_workers == args.worker_rank
    ]
    shard_iter = tqdm(
        assigned_shards,
        total=len(parquet_files),
        desc=f"shards worker {args.worker_rank}/{args.num_workers}",
        unit="shard",
        dynamic_ncols=True,
    )
    for global_shard_idx, input_path in shard_iter:
        rel_input_path = os.path.relpath(input_path, args.input_root)
        shard_iter.set_postfix(file=rel_input_path)
        out_path = output_path_for(input_path, args.input_root, args.output_root)
        if os.path.exists(out_path) and not args.overwrite:
            tqdm.write(f"[skip] {out_path} exists")
            continue

        table = pq.read_table(input_path)
        if args.limit_rows_per_shard is not None:
            table = table.slice(0, args.limit_rows_per_shard)
        rows = table.to_pylist()

        kept_indices: List[int] = []
        query_keys: List[Optional[str]] = []
        document_keys: List[Optional[str]] = []
        pending: Dict[str, Dict[str, Any]] = {}

        row_progress = tqdm(
            enumerate(rows),
            total=len(rows),
            desc=f"scan {rel_input_path}",
            unit="row",
            dynamic_ncols=True,
            leave=False,
        )
        for row_idx, row in row_progress:
            is_video = row_has_video(row)
            stats["input_rows"] += 1
            stats["video_rows"] += int(is_video)
            if is_video and rng.random() >= args.video_sample_ratio:
                if row_idx % 100 == 0:
                    row_progress.set_postfix(
                        kept=len(kept_indices),
                        pending=len(pending),
                        cache=len(cache),
                    )
                continue

            kept_indices.append(row_idx)
            stats["kept_rows"] += 1
            stats["kept_video_rows"] += int(is_video)

            for side, side_keys in (("query", query_keys), ("document", document_keys)):
                if not side_has_visual(row, side):
                    side_keys.append(None)
                    continue
                key = side_cache_key(row, side, args=args)
                side_keys.append(key)
                if key not in cache and key not in pending:
                    pending[key] = build_side_sample(row, side, video_root=video_root, args=args)
            if row_idx % 100 == 0:
                row_progress.set_postfix(
                    kept=len(kept_indices),
                    pending=len(pending),
                    cache=len(cache),
                )

        if pending:
            stats["cache_misses"] += len(pending)
            process_pending(
                ranker,
                cache,
                pending,
                args.rank_batch_size,
                desc=f"rank {rel_input_path}",
            )

        if len(kept_indices) == table.num_rows and kept_indices == list(range(table.num_rows)):
            kept_table = table
        else:
            kept_table = table.take(pa.array(kept_indices, type=pa.int64()))
        query_rankings = [ranking_or_empty(cache, key) for key in query_keys]
        document_rankings = [ranking_or_empty(cache, key) for key in document_keys]

        kept_table = kept_table.append_column(
            "query_visual_modalities",
            pa.array([item.modalities for item in query_rankings], type=pa.list_(pa.string())),
        )
        kept_table = kept_table.append_column(
            "document_visual_modalities",
            pa.array([item.modalities for item in document_rankings], type=pa.list_(pa.string())),
        )
        kept_table = kept_table.append_column(
            "query_visual_token_counts",
            pa.array([item.token_counts for item in query_rankings], type=pa.list_(pa.int32())),
        )
        kept_table = kept_table.append_column(
            "document_visual_token_counts",
            pa.array([item.token_counts for item in document_rankings], type=pa.list_(pa.int32())),
        )
        rank_type = pa.list_(pa.list_(pa.int32()))
        kept_table = kept_table.append_column(
            "query_visual_ranked_tokens",
            pa.array([item.ranked_tokens for item in query_rankings], type=rank_type),
        )
        kept_table = kept_table.append_column(
            "document_visual_ranked_tokens",
            pa.array([item.ranked_tokens for item in document_rankings], type=rank_type),
        )

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        pq.write_table(kept_table, out_path)
        stats["shards"] += 1
        tqdm.write(
            f"[global {global_shard_idx + 1}/{len(all_parquet_files)} | "
            f"worker {stats['shards']}/{len(parquet_files)}] wrote {out_path} "
            f"rows={kept_table.num_rows} cache_size={len(cache)}"
        )

    manifest = {
        "input_root": args.input_root,
        "output_root": args.output_root,
        "embedder_model_name_or_path": args.embedder_model_name_or_path,
        "prune_method": args.prune_method,
        "prune_layer": args.prune_layer,
        "video_sample_ratio": args.video_sample_ratio,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "worker_rank": args.worker_rank,
        "total_shards": len(all_parquet_files),
        "assigned_shards": [
            os.path.relpath(path, args.input_root)
            for _, path in assigned_shards
        ],
        "vision": {
            "max_length": args.max_length,
            "min_pixels": args.min_pixels,
            "max_pixels": args.max_pixels,
            "video_frame_max_pixels": args.video_frame_max_pixels,
            "video_total_pixels": args.video_total_pixels,
            "fps": args.fps,
            "max_video_frames": args.max_video_frames,
            "video_root": video_root,
        },
        "stats": stats,
    }
    manifest_name = (
        "manifest.json"
        if args.num_workers == 1
        else f"manifest.worker{args.worker_rank:03d}-of-{args.num_workers:03d}.json"
    )
    with open(output_root / manifest_name, "w") as f:
        json.dump(manifest, f, indent=2)
    print(json.dumps(manifest["stats"], indent=2))


if __name__ == "__main__":
    main()
