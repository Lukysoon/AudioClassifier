"""
Audio Classifier - Audio classification and visualization using audio embeddings.

This package provides tools for:
- Extracting audio embeddings (ContentVec, HuBERT, etc.)
- Dimensionality reduction with UMAP
- Interactive 3D visualization with Plotly
"""

from .cache import EmbeddingCache
from .config import (
    AudioConfig,
    ModelConfig,
    UMAPConfig,
    VisualizationConfig,
    PipelineConfig,
)
from .pipeline import AudioClassifierPipeline

__version__ = "0.1.0"
__all__ = [
    "AudioClassifierPipeline",
    "EmbeddingCache",
    "AudioConfig",
    "ModelConfig",
    "UMAPConfig",
    "VisualizationConfig",
    "PipelineConfig",
]
