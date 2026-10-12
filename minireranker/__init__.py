"""Transformers-based multimodal reranking."""
__version__ = "0.1.0"

def __getattr__(name):
    if name == "MiniReranker":
        from .inference import MiniReranker
        return MiniReranker
    raise AttributeError(name)
