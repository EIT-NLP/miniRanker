"""Candidate ordering independent of the embedding model's tie order."""
import hashlib
import json
import math
import random

UPSTREAM_COMMIT = "683585a8d1a4cdedd40cd797af9410ed87ede728"

def shuffled_candidates(candidate_ids, query_key, seed=42):
    candidates = list(candidate_ids)
    digest = hashlib.sha256(f"{seed}|{query_key}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(candidates)
    return candidates

def rank_candidates(candidate_ids, scores):
    if len(candidate_ids) != len(scores):
        raise ValueError("Candidate IDs and scores must have the same length.")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Reranker returned a non-finite score.")
    order = sorted(range(len(scores)), key=lambda i: -float(scores[i]))
    return [candidate_ids[i] for i in order], [float(scores[i]) for i in order]

def metadata_matches(path, expected):
    try:
        with open(path, encoding="utf-8") as stream:
            saved = json.load(stream)
        return saved.get("minireranker_metadata") == expected
    except (OSError, ValueError):
        return False
