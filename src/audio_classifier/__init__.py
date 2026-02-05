"""
Audio Classifier - Audio classification and visualization using HuBERT embeddings.

This package provides tools for:
- Extracting audio embeddings using HuBERT
- Dimensionality reduction with UMAP
- Interactive 3D visualization with Plotly
"""

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
    "AudioConfig",
    "ModelConfig",
    "UMAPConfig",
    "VisualizationConfig",
    "PipelineConfig",
]
