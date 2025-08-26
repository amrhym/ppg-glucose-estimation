"""PPG Glucose Estimation Package."""

__version__ = "1.0.0"
__author__ = "PPG Glucose Team"

# Lazy imports to avoid torch dependency when not needed
__all__ = ["PreprocessingPipeline", "SignalQualityValidator"]

def __getattr__(name):
    if name == "HybridCNNGRU":
        from src.models.hybrid_model import HybridCNNGRU
        return HybridCNNGRU
    elif name == "PreprocessingPipeline":
        from src.preprocessing.pipeline import PreprocessingPipeline
        return PreprocessingPipeline
    elif name == "SignalQualityValidator":
        from src.quality.validator import SignalQualityValidator
        return SignalQualityValidator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")