"""
Dimensionality reduction module using UMAP.
"""

import numpy as np
import umap

from .config import UMAPConfig


class DimensionalityReducer:
    """Reduces high-dimensional embeddings to 3D using UMAP."""

    def __init__(self, config: UMAPConfig | None = None):
        """
        Initialize the dimensionality reducer.

        Args:
            config: UMAP configuration. Uses defaults if None.
        """
        self.config = config or UMAPConfig()
        self.reducer: umap.UMAP | None = None

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP on embeddings and transform to lower dimensions.

        Args:
            embeddings: Array of shape [n_samples, 768].

        Returns:
            Reduced coordinates of shape [n_samples, n_components].
        """
        print(f"Reducing dimensions: {embeddings.shape} -> (n, {self.config.n_components})")

        self.reducer = umap.UMAP(
            n_components=self.config.n_components,
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            metric=self.config.metric,
            repulsion_strength=self.config.repulsion_strength,
            random_state=self.config.random_state,
            verbose=False
        )

        coords = self.reducer.fit_transform(embeddings)
        print("Dimension reduction complete")

        return coords

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform new embeddings using already fitted UMAP.

        Args:
            embeddings: Array of shape [n_samples, 768].

        Returns:
            Reduced coordinates of shape [n_samples, n_components].

        Raises:
            RuntimeError: If reducer has not been fitted yet.
        """
        if self.reducer is None:
            raise RuntimeError("Reducer not fitted. Call fit_transform first.")

        return self.reducer.transform(embeddings)

    @property
    def is_fitted(self) -> bool:
        """Check if the reducer has been fitted."""
        return self.reducer is not None
