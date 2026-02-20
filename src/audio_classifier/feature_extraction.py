"""
ContentVec feature extraction module.
"""

import numpy as np
import torch
from pathlib import Path
from transformers import HubertModel, Wav2Vec2FeatureExtractor

from .config import ModelConfig
from .pooling import get_pooling_strategy

# Local cache directory for models (relative to project root)
MODELS_DIR = Path(__file__).parent.parent.parent / "models"


class ContentVecExtractor:
    """Wrapper for ContentVec feature extraction."""

    def __init__(self, config: ModelConfig | None = None):
        """
        Initialize the ContentVec extractor.

        Args:
            config: Model configuration. Uses defaults if None.
        """
        self.config = config or ModelConfig()
        self.device = self._get_device()

        # Ensure models directory exists
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        cache_dir = str(MODELS_DIR)

        print(f"Loading model: {self.config.model_name}")
        print(f"Model cache: {cache_dir}")
        print(f"Using device: {self.device}")

        # Load feature extractor (handles input normalization)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir
        )

        # Load ContentVec model (HuBERT architecture)
        self.model = HubertModel.from_pretrained(
            self.config.model_name,
            cache_dir=cache_dir
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        # Get pooling strategy
        self.pooling = get_pooling_strategy(self.config.pooling)

        print("Model loaded successfully")

    def _get_device(self) -> torch.device:
        """Determine the device to use for inference."""
        if self.config.device:
            return torch.device(self.config.device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def extract(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract 768-dimensional embedding from preprocessed waveform.

        Args:
            waveform: 1D numpy array of audio samples (16kHz, normalized).

        Returns:
            768-dimensional embedding vector as numpy array.
        """
        # Prepare input using feature extractor
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        # Move to device
        input_values = inputs.input_values.to(self.device)

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                input_values,
                output_hidden_states=True
            )

            # Select layer
            if self.config.layer == -1:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[self.config.layer]

        # Apply pooling to get single vector
        embedding = self.pooling(hidden_states)

        return embedding.cpu().numpy().squeeze()

    def extract_sequence(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract sequence of embeddings without pooling.

        Useful for analysis of temporal patterns.

        Args:
            waveform: 1D numpy array of audio samples.

        Returns:
            Sequence of embeddings with shape [seq_len, 768].
        """
        inputs = self.feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )

        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_values,
                output_hidden_states=True
            )

            if self.config.layer == -1:
                hidden_states = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states[self.config.layer]

        return hidden_states.cpu().numpy().squeeze()
