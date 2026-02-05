"""
Configuration dataclasses for the Audio Classifier pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional


@dataclass
class AudioConfig:
    """Audio preprocessing parameters."""

    target_sample_rate: int = 16000
    """Target sample rate for audio (HuBERT expects 16kHz)."""

    normalize: bool = True
    """Whether to normalize audio to [-1, 1] range."""

    max_duration_seconds: Optional[float] = 240
    """Maximum audio duration in seconds. None for no limit."""

    extensions: List[str] = field(default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg"])
    """Supported audio file extensions."""

    chunking_enabled: bool = False
    """Enable fixed-length chunking of audio files."""

    chunk_duration_seconds: float = 5.0
    """Duration of each chunk in seconds (when chunking is enabled)."""

    chunk_handling: Literal["pad", "discard", "keep"] = "discard"
    """How to handle the final chunk if shorter than chunk_duration_seconds:
       - 'pad': Zero-pad to full chunk length
       - 'discard': Drop the final short chunk
       - 'keep': Keep as-is (shorter chunk)
    """

    min_chunk_duration_seconds: float = 0.5
    """Minimum duration for 'keep' mode - chunks shorter than this are discarded."""

    silence_removal_enabled: bool = True
    """Remove silent segments from audio."""

    silence_threshold_db: float = 40.0
    """Threshold in dB below reference to consider as silence."""


@dataclass
class ModelConfig:
    """HuBERT model parameters."""

    model_name: str = "facebook/hubert-base-ls960"
    """HuggingFace model identifier."""

    pooling: Literal["mean", "max"] = "mean"
    """Pooling strategy for temporal aggregation."""

    layer: int = -1
    """Which transformer layer to use (-1 = last, 6 = middle)."""

    device: Optional[str] = None
    """Device to use (None = auto-detect cuda/cpu)."""


@dataclass
class UMAPConfig:
    """UMAP dimensionality reduction parameters."""

    n_components: int = 3
    """Number of output dimensions."""

    n_neighbors: int = 15
    """Number of neighbors for local structure. Smaller = more local clusters."""

    min_dist: float = 0.1
    """Minimum distance in low-dim space. Smaller = tighter clusters."""

    metric: str = "cosine"
    """Distance metric. 'cosine' is good for embeddings."""

    random_state: int = 42
    """Random seed for reproducibility."""


@dataclass
class VisualizationConfig:
    """Visualization output parameters."""

    output_dir: Path = field(default_factory=lambda: Path("./output"))
    """Directory for output HTML files."""

    marker_size: int = 5
    """Size of markers in 3D plot."""

    marker_opacity: float = 0.8
    """Opacity of markers (0-1)."""

    color_scale: str = "Viridis"
    """Color scale for heatmaps."""

    auto_open: bool = True
    """Automatically open visualizations in browser."""


@dataclass
class PipelineConfig:
    """Aggregated configuration for the full pipeline."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    umap: UMAPConfig = field(default_factory=UMAPConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()

        if "audio" in data:
            for key, value in data["audio"].items():
                if hasattr(config.audio, key):
                    setattr(config.audio, key, value)

        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        if "umap" in data:
            for key, value in data["umap"].items():
                if hasattr(config.umap, key):
                    setattr(config.umap, key, value)

        if "visualization" in data:
            for key, value in data["visualization"].items():
                if hasattr(config.visualization, key):
                    if key == "output_dir":
                        value = Path(value)
                    setattr(config.visualization, key, value)

        return config

    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        import yaml

        data = {
            "audio": {
                "target_sample_rate": self.audio.target_sample_rate,
                "normalize": self.audio.normalize,
                "max_duration_seconds": self.audio.max_duration_seconds,
                "extensions": self.audio.extensions,
                "chunking_enabled": self.audio.chunking_enabled,
                "chunk_duration_seconds": self.audio.chunk_duration_seconds,
                "chunk_handling": self.audio.chunk_handling,
                "min_chunk_duration_seconds": self.audio.min_chunk_duration_seconds,
                "silence_removal_enabled": self.audio.silence_removal_enabled,
                "silence_threshold_db": self.audio.silence_threshold_db,
            },
            "model": {
                "model_name": self.model.model_name,
                "pooling": self.model.pooling,
                "layer": self.model.layer,
                "device": self.model.device,
            },
            "umap": {
                "n_components": self.umap.n_components,
                "n_neighbors": self.umap.n_neighbors,
                "min_dist": self.umap.min_dist,
                "metric": self.umap.metric,
                "random_state": self.umap.random_state,
            },
            "visualization": {
                "output_dir": str(self.visualization.output_dir),
                "marker_size": self.visualization.marker_size,
                "marker_opacity": self.visualization.marker_opacity,
                "color_scale": self.visualization.color_scale,
                "auto_open": self.visualization.auto_open,
            },
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)
