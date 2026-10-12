"""Use Qwen's MMEB-v2 datasets and metrics with miniReranker inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from .evaluation_utils import UPSTREAM_COMMIT

def _fingerprint(path, pattern="*"):
    path = Path(path)
    if not path.exists():
        return None
    files = [path] if path.is_file() else sorted(path.glob(pattern))
    payload = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files if p.is_file()]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description="miniReranker adapter; additional arguments are forwarded to Qwen MMEB-v2 evaluation.")
    parser.add_argument("--upstream_repo", default=os.environ.get("QWEN3_VL_EMBEDDING_REPO"))
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--embedder_model_name_or_path", default="Qwen/Qwen3-VL-Embedding-2B")
    parser.add_argument("--prune_k", type=float, default=None)
    parser.add_argument("--prune_method", choices=["attention", "hidden_state_similarity"], default="attention")
    parser.add_argument("--prune_layer", default="all")
    parser.add_argument("--rerank_format", default=None)
    parser.add_argument("--early_exit_layer", type=int, default=None)
    parser.add_argument("--no_interaction_layer", type=int, default=None)
    options, remaining = parser.parse_known_args()
    if not options.upstream_repo:
        parser.error("Set --upstream_repo or QWEN3_VL_EMBEDDING_REPO; see docs/evaluation.md.")
    root = Path(options.upstream_repo).expanduser().resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(f"Use Qwen3-VL-Embedding commit {UPSTREAM_COMMIT}.")
    entry = root / "src/evaluation/mmeb_v2/eval_reranker.py"
    if "MINIRERANKER_METADATA" not in entry.read_text():
        raise RuntimeError("Apply patches/qwen3-vl-evaluation.patch to the clean upstream clone first.")
    sys.path.insert(0, str(root))
    from src.evaluation.mmeb_v2 import eval_reranker as evaluator
    from transformers import AutoConfig, HfArgumentParser
    from .inference import MiniReranker

    if "--help" in remaining or "-h" in remaining:
        print("miniReranker options: --upstream_repo --shuffle_seed --prune_k --prune_method "
              "--prune_layer --embedder_model_name_or_path --rerank_format "
              "--early_exit_layer --no_interaction_layer")
    parsed = HfArgumentParser((evaluator.RerankArguments, evaluator.DataArguments,
                              evaluator.EvalArguments)).parse_args_into_dataclasses(args=remaining)
    model_args, data_args, eval_args = parsed
    config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    effective = {
        "prune_k": options.prune_k if options.prune_k is not None else getattr(config, "prune_k", None),
        "early_exit_layer": options.early_exit_layer if options.early_exit_layer is not None else getattr(config, "early_exit_layer", config.text_config.num_hidden_layers - 1),
        "no_interaction_layer": options.no_interaction_layer if options.no_interaction_layer is not None else getattr(config, "no_interaction_layer", getattr(config, "full_document_mask_layer", -1)),
        "rerank_format": options.rerank_format or getattr(config, "rerank_format", "image_first"),
    }
    argument_fingerprint = hashlib.sha256(json.dumps({
        "model": vars(model_args), "data": vars(data_args),
        "evaluation": eval_args.to_dict(), "adapter": vars(options),
    }, sort_keys=True, default=str).encode()).hexdigest()
    metadata = {
        "protocol": "minireranker-original-topk-v1", "upstream_commit": revision,
        "shuffle_seed": options.shuffle_seed, "score_type": "raw_yes_minus_no_logits",
        "effective_config": effective, "prune_method": options.prune_method,
        "prune_layer": options.prune_layer, "argument_fingerprint": argument_fingerprint,
        "model_config_sha256": hashlib.sha256(config.to_json_string(use_diff=False).encode()).hexdigest(),
        "local_checkpoint_fingerprint": _fingerprint(model_args.model_name_or_path),
        "embedding_predictions_fingerprint": _fingerprint(data_args.encode_output_path, "*_pred.jsonl"),
        "dataset_config_sha256": hashlib.sha256(Path(data_args.dataset_config).read_bytes()).hexdigest(),
    }
    evaluator.MINIRERANKER_METADATA = metadata
    evaluator.MINIRERANKER_SHUFFLE_SEED = options.shuffle_seed

    class Adapter(MiniReranker):
        def __init__(self, model_name_or_path, **kwargs):
            kwargs.pop("attn_implementation", None)  # The upstream launcher requests FlashAttention.
            if not evaluator.torch.cuda.is_available():
                kwargs["torch_dtype"] = evaluator.torch.float32
            super().__init__(
                model_name_or_path, **kwargs, **effective,
                embedder_model_name_or_path=options.embedder_model_name_or_path,
                prune_method=options.prune_method, prune_layer=options.prune_layer,
            )

    evaluator.Qwen3VLReranker = Adapter
    sys.argv = [str(entry), *remaining]
    evaluator.main()

if __name__ == "__main__":
    main()
