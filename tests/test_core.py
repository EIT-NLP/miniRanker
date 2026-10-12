import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from transformers import Qwen3VLConfig

from minireranker.batch_processing import apply_token_pruning_to_batch
from minireranker.custom_modeling_qwen3_vl import CrossAttnMasking_Qwen3_VLForConditionalGeneration
from minireranker.evaluation_utils import metadata_matches, rank_candidates, shuffled_candidates
from minireranker.inference import MiniReranker
from minireranker.rerank_utils import build_rerank_messages, normalize_video_paths
from minireranker.token_pruning_utils import resolve_keep_count, select_keep_indices_by_rank
from minireranker.token_ranker import Qwen3VLVisualTokenRanker
from minireranker._vendor.qwen3_vl_embedding import Qwen3VLForEmbedding


def test_tied_scores_use_shuffled_order_and_keep_candidate_identity():
    ids = list(range(20))
    shuffled = shuffled_candidates(ids, "query-7", 42)
    assert shuffled != ids
    assert sorted(shuffled) == ids
    assert shuffled_candidates(ids, "query-7", 42) == shuffled
    assert shuffled_candidates(ids, "query-7", 43) != shuffled
    ranked, scores = rank_candidates(shuffled, [1.0] * len(ids))
    assert ranked == shuffled and scores == [1.0] * len(ids)
    values = [float(i) for i in shuffled]
    ranked, _ = rank_candidates(shuffled, values)
    assert ranked == list(reversed(ids))
    assert ids == list(range(20))
    with pytest.raises(ValueError):
        rank_candidates(["id"], [])
    with pytest.raises(ValueError):
        rank_candidates(["id"], [float("nan")])


def test_result_cache_requires_exact_configuration(tmp_path):
    target = tmp_path / "scores.json"
    metadata = {"seed": 42, "prune_k": 0.5}
    target.write_text(json.dumps({"minireranker_metadata": metadata}))
    assert metadata_matches(target, metadata)
    assert not metadata_matches(target, {**metadata, "seed": 43})
    target.write_text("{broken")
    assert not metadata_matches(target, metadata)


def test_raw_scores_do_not_saturate_and_ignore_padding():
    reranker = MiniReranker.__new__(MiniReranker)
    hidden = torch.tensor([[[80.0], [90.0], [0.0]], [[-90.0], [-80.0], [0.0]]])
    reranker.model = SimpleNamespace(model=lambda **kwargs: SimpleNamespace(last_hidden_state=hidden))
    reranker.score_linear = torch.nn.Linear(1, 1, bias=False)
    reranker.score_linear.weight.data.fill_(1)
    assert reranker.compute_scores({"attention_mask": torch.tensor([[1, 1, 0], [1, 1, 0]])}) == [90.0, -80.0]
    assert torch.sigmoid(torch.tensor([80.0, 90.0])).tolist() == [1.0, 1.0]


def test_keep_budget_and_multiple_media():
    assert resolve_keep_count(7, 0.5) == 4
    assert resolve_keep_count(7, 1.0) == 7
    assert resolve_keep_count(7, 3) == 3
    assert select_keep_indices_by_rank([[0, 3], [0, 1], [1, 2]], [4, 3], 1.0) == [[1, 3], [2]]
    with pytest.raises(ValueError):
        resolve_keep_count(7, 0)


@pytest.mark.parametrize("query_visual,document_visual,first", [
    (True, False, "Query"), (False, True, "Document"), (True, True, "Document"),
])
def test_vision_first_matches_training_format(query_visual, document_visual, first):
    image = Image.new("RGB", (32, 32))
    messages = build_rerank_messages(
        "Find a match", query_text="query", document_text="document",
        query_image=image if query_visual else None,
        document_image=image if document_visual else None,
        rerank_format="image_first", label=None,
    )
    headers = [block["text"] for block in messages[1]["content"] if block["type"] == "text"]
    joined = " ".join(headers)
    assert joined.index(f"<{first}>:") < joined.index("<Document>:" if first == "Query" else "<Query>:")
    assert len(messages) == 2


def test_video_root_and_missing_frames(tmp_path):
    frames = []
    for i in range(4):
        name = f"frame{i}.jpeg"
        Image.new("RGB", (32, 32)).save(tmp_path / name)
        frames.append(name)
    assert normalize_video_paths(frames, str(tmp_path), 2) == [str(tmp_path / frames[0]), str(tmp_path / frames[-1])]
    with pytest.raises(FileNotFoundError):
        normalize_video_paths(["missing.jpeg"], str(tmp_path), 2)


def tiny_model():
    config = Qwen3VLConfig(
        text_config={"vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
                     "num_hidden_layers": 2, "num_attention_heads": 4,
                     "num_key_value_heads": 2, "head_dim": 16,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True}},
        vision_config={"depth": 2, "hidden_size": 32, "intermediate_size": 64,
                       "num_heads": 4, "out_hidden_size": 64, "patch_size": 16,
                       "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "deepstack_visual_indexes": []},
        image_token_id=120, video_token_id=121, vision_start_token_id=122, vision_end_token_id=123,
    )
    config._attn_implementation = {"text_config": "eager", "vision_config": "sdpa"}
    config.text_config._attn_implementation = "eager"
    config.vision_config._attn_implementation = "sdpa"
    config.early_exit_layer = 1
    config.no_interaction_layer = 0
    model = CrossAttnMasking_Qwen3_VLForConditionalGeneration(config)
    model.set_attn_implementation({"text_config": "eager", "vision_config": "sdpa"})
    return model


class TokenizerStub:
    pad_token_id = 0
    eos_token_id = 1
    def convert_tokens_to_ids(self, token):
        return {"<|image_pad|>": 120, "<|video_pad|>": 121}[token]


@pytest.mark.parametrize("modality,token,grid,patches,count", [
    ("image", 120, [1, 4, 4], 16, 4), ("video", 121, [2, 4, 4], 32, 8),
])
def test_pruned_image_and_video_forward_backward_and_reload(tmp_path, modality, token, grid, patches, count):
    torch.manual_seed(42)
    model = tiny_model()
    processor = SimpleNamespace(tokenizer=TokenizerStub(), image_processor=SimpleNamespace(merge_size=2),
                                image_token_id=120, video_token_id=121)
    # Two token runs represent separate temporal positions for a video.
    visual = [token] * count if modality == "image" else [token] * 4 + [14] + [token] * 4
    ids = torch.tensor([[11, 122, *visual, 123, 12, 13]])
    full = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
            f"{modality}_grid_thw": torch.tensor([grid]),
            "pixel_values" if modality == "image" else "pixel_values_videos": torch.randn(patches, 1536)}
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 5
    example = {"document_image_bytes": b"present" if modality == "image" else None,
               "document_video_frame_paths": ["frame"] if modality == "video" else [],
               "document_visual_modalities": [modality], "document_visual_token_counts": [count],
               "document_visual_ranked_tokens": [[0, i] for i in reversed(range(count))]}
    prepared = apply_token_pruning_to_batch(full, labels, [(0, len(visual) + 3)], [example],
               prune_k=0.5, rerank_format="image_first", processor=processor, k=-1, k_selection=None)
    for key, value in full.items():
        prepared.setdefault(key, value)
    assert int((prepared["input_ids"] == token).sum()) == count // 2
    out = model(**prepared, use_cache=False)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert model.lm_head.weight.grad is not None
    model.eval()
    with torch.no_grad():
        expected = model(**prepared, use_cache=False).logits
    model.save_pretrained(tmp_path)
    reloaded = CrossAttnMasking_Qwen3_VLForConditionalGeneration.from_pretrained(tmp_path)
    reloaded.eval()
    with torch.no_grad():
        actual = reloaded(**prepared, use_cache=False).logits
    torch.testing.assert_close(expected, actual)
    assert reloaded.config.early_exit_layer == 1 and reloaded.config.no_interaction_layer == 0


def test_empty_documents_do_not_invoke_model():
    reranker = MiniReranker.__new__(MiniReranker)
    reranker.default_instruction = "instruction"
    assert reranker.process({"query": {"text": "anything"}, "documents": []}) == []


def test_streamed_attention_ranking_matches_original_all_layer_formula():
    native = Qwen3VLForEmbedding(tiny_model().config).eval()
    native.set_attn_implementation("eager")
    inputs = {"input_ids": torch.tensor([[11, 122, 120, 120, 120, 120, 123, 12, 13]]),
              "attention_mask": torch.ones((1, 9), dtype=torch.long),
              "image_grid_thw": torch.tensor([[1, 4, 4]]), "pixel_values": torch.randn(16, 1536)}
    ranker = Qwen3VLVisualTokenRanker.__new__(Qwen3VLVisualTokenRanker)
    ranker.prune_method, ranker.prune_layer = "attention", "all"
    ranker.normalize_embedding = True
    ranker.video_frame_max_pixels = None
    ranker.embedder = SimpleNamespace(model=native, format_model_input=lambda **kwargs: kwargs,
                                     _preprocess_inputs=lambda conversations: inputs)
    prepared = ranker._prepare_sequence(inputs)
    positions = prepared["media_spans"][0][0][1]
    with torch.no_grad():
        original = native.language_model(
            input_ids=None, inputs_embeds=prepared["inputs_embeds"],
            attention_mask=prepared["attention_mask"], position_ids=prepared["position_ids"],
            visual_pos_masks=prepared["visual_pos_masks"],
            deepstack_visual_embeds=prepared["deepstack_visual_embeds"],
            use_cache=False, output_attentions=True, return_dict=True,
        )
    golden = sum(attention[0, :, -1, positions].mean(0) for attention in original.attentions)
    ranking = ranker.build_rankings([{"image": "present"}])[0]
    assert ranking.token_counts == [4] and ranking.modalities == ["image"]
    assert ranking.ranked_tokens == [[0, i] for i in torch.argsort(golden, descending=True, stable=True).tolist()]
