"""Model architectures for glucose estimation."""

__all__ = ["HybridCNNGRU", "ModelConfig"]

def __getattr__(name):
    """Lazy import to avoid torch dependency when not needed."""
    if name in ["HybridCNNGRU", "ModelConfig"]:
        from src.models.hybrid_model import HybridCNNGRU, ModelConfig
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")