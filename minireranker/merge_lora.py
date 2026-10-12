import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor

from .custom_modeling_qwen3_vl import CrossAttnMasking_Qwen3_VLForConditionalGeneration


def apply_top_level_arch_config(
    config,
    k: int,
    k_selection: str | None,
    early_exit_layer: int,
    no_interaction_layer: int,
    rerank_format: str,
) -> None:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-path", type=str, required=True)
    parser.add_argument("--base-model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    adapter_config = AutoConfig.from_pretrained(args.lora_path)
    legacy_last_k = getattr(adapter_config, "last_k", -1)
    k = getattr(adapter_config, "k", legacy_last_k)
    k_selection = getattr(adapter_config, "k_selection", None)
    if k_selection is None and legacy_last_k >= 0:
        k_selection = "last"
    early_exit_layer = getattr(
        adapter_config,
        "early_exit_layer",
        adapter_config.text_config.num_hidden_layers - 1,
    )
    no_interaction_layer = getattr(
        adapter_config,
        "no_interaction_layer",
        getattr(adapter_config, "full_document_mask_layer", -1),
    )
    rerank_format = getattr(adapter_config, "rerank_format", "query_first")
    apply_top_level_arch_config(
        adapter_config,
        k=k,
        k_selection=k_selection,
        early_exit_layer=early_exit_layer,
        no_interaction_layer=no_interaction_layer,
        rerank_format=rerank_format,
    )

    base_model = CrossAttnMasking_Qwen3_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        config=adapter_config,
        k=k,
        k_selection=k_selection,
        early_exit_layer=early_exit_layer,
        no_interaction_layer=no_interaction_layer,
        rerank_format=rerank_format,
        dtype=torch.bfloat16,
        device_map="cpu",
        attn_implementation="eager",
    )

    model = PeftModel.from_pretrained(base_model, args.lora_path)
    merged_model = model.merge_and_unload()
    apply_top_level_arch_config(
        merged_model.config,
        k=k,
        k_selection=k_selection,
        early_exit_layer=early_exit_layer,
        no_interaction_layer=no_interaction_layer,
        rerank_format=rerank_format,
    )
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)

    processor = AutoProcessor.from_pretrained(args.base_model_path)
    processor.save_pretrained(args.output_dir)

    print(f"Merged model saved to: {args.output_dir}")
    print(
        "Preserved config: "
        f"k={k}, "
        f"k_selection={k_selection}, "
        f"early_exit_layer={early_exit_layer}, "
        f"no_interaction_layer={no_interaction_layer}, "
        f"rerank_format={rerank_format}"
    )


if __name__ == "__main__":
    main(parse_args())
