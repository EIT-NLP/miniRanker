"""Bounded GPU smoke check; does not run full training or benchmark evaluation."""
import argparse
import json
from pathlib import Path
import tempfile

import pyarrow.parquet as pq
import torch
from PIL import Image
from peft import LoraConfig, get_peft_model

from minireranker import MiniReranker
from minireranker.train import Qwen3VLRerankCollator, freeze_vision_and_projector, save_model_config


def first_row(root, subset):
    shard = sorted((root / subset).glob("original-*.parquet"))[0]
    return next(pq.ParquetFile(shard).iter_batches(batch_size=1)).to_pylist()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--embedder", required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--video_root", required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/validation"))
    parser.add_argument("--train_steps", type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke check requires one CUDA GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = MiniReranker(args.model, embedder_model_name_or_path=args.embedder,
                        prune_k=0.5, video_root=args.video_root)
    image = first_row(args.data_root, "VisRAG")
    video = first_row(args.data_root, "ShareGPTVideo_caption_t2v")
    scores = model.process_examples([image, video])
    assert len(scores) == 2 and all(torch.isfinite(torch.tensor(scores)))
    # Exercise online ranking on both image and video, without reading the full dataset.
    with tempfile.TemporaryDirectory(dir=args.output_dir) as folder:
        frames = []
        for i in range(2):
            target = Path(folder) / f"frame{i}.jpeg"
            Image.new("RGB", (64, 64), (30 + i, 90, 160)).save(target)
            frames.append(str(target.resolve()))
        online = model.process({"query": {"text": "A blue background"},
                                "documents": [{"image": frames[0]}, {"video": frames}]})
    assert len(online) == 2 and all(torch.isfinite(torch.tensor(online)))
    freeze_vision_and_projector(model.model, False)
    adapter = get_peft_model(model.model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    optimizer = torch.optim.AdamW((p for p in adapter.parameters() if p.requires_grad), lr=1e-4)
    collator = Qwen3VLRerankCollator(processor=model.processor, video_root=args.video_root,
                                    rerank_format=model.rerank_format, prune_k=0.5)
    adapter.config.use_cache = False
    adapter.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    adapter.train()
    losses = []
    for step in range(args.train_steps):
        batch = collator([image if step % 2 == 0 else video])
        batch = {key: value.to(model.device) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        loss = adapter(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in adapter.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
    target = args.output_dir / "lora"
    adapter.save_pretrained(target)
    save_model_config(adapter, str(target))
    model.processor.save_pretrained(target)
    summary = {"gpu": torch.cuda.get_device_name(), "dataset_scores": scores,
               "online_scores": online, "training_steps": len(losses), "losses": losses,
               "saved_adapter": True, "peak_gpu_memory_gib": torch.cuda.max_memory_allocated() / 2**30}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
