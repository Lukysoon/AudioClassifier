"""
Analysis module for distance calculations and clustering metrics.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ClusteringMetrics:
    """Container for clustering quality metrics."""

    intra_class_distances: Dict[str, float]
    """Average distance from points to their category centroid."""

    inter_class_distances: List[Tuple[str, str, float]]
    """Pairwise distances between category centroids."""

    mean_intra: float
    """Mean of all intra-class distances."""

    mean_inter: float
    """Mean of all inter-class distances."""

    @property
    def separation_ratio(self) -> float:
        """
        Ratio of inter-class to intra-class distances.

        Higher is better - means categories are well separated.
        """
        if self.mean_intra == 0:
            return float("inf")
        return self.mean_inter / self.mean_intra


def calculate_centroid_distances(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate distance matrix between category centroids.

    Args:
        df: DataFrame with columns: label, x, y, z

    Returns:
        Square DataFrame with distances between categories.
    """
    labels = sorted(df["label"].unique())

    # Calculate centroids
    centroids = {}
    for label in labels:
        mask = df["label"] == label
        centroids[label] = df.loc[mask, ["x", "y", "z"]].mean().values

    # Calculate pairwise distances
    n = len(labels)
    dist_matrix = np.zeros((n, n))

    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            dist_matrix[i, j] = np.linalg.norm(centroids[l1] - centroids[l2])

    return pd.DataFrame(dist_matrix, index=labels, columns=labels)


def analyze_clustering_quality(
    coords: np.ndarray,
    labels: List[str]
) -> ClusteringMetrics:
    """
    Analyze clustering quality with intra/inter-class distances.

    Args:
        coords: Array of shape [n_samples, 3] with 3D coordinates.
        labels: List of category labels for each sample.

    Returns:
        ClusteringMetrics with detailed distance analysis.
    """
    unique_labels = sorted(set(labels))
    labels_array = np.array(labels)

    # Calculate intra-class distances (points to their centroid)
    intra_distances = {}
    centroids = {}

    for label in unique_labels:
        mask = labels_array == label
        points = coords[mask]
        centroid = points.mean(axis=0)
        centroids[label] = centroid

        # Average distance from points to centroid
        distances = [np.linalg.norm(p - centroid) for p in points]
        intra_distances[label] = np.mean(distances) if distances else 0.0

    # Calculate inter-class distances (between centroids)
    inter_distances = []
    for i, l1 in enumerate(unique_labels):
        for l2 in unique_labels[i + 1:]:
            dist = np.linalg.norm(centroids[l1] - centroids[l2])
            inter_distances.append((l1, l2, dist))

    # Calculate means
    mean_intra = np.mean(list(intra_distances.values())) if intra_distances else 0.0
    mean_inter = np.mean([d[2] for d in inter_distances]) if inter_distances else 0.0

    return ClusteringMetrics(
        intra_class_distances=intra_distances,
        inter_class_distances=inter_distances,
        mean_intra=mean_intra,
        mean_inter=mean_inter
    )


def print_clustering_report(metrics: ClusteringMetrics) -> None:
    """
    Print a formatted clustering quality report.

    Args:
        metrics: ClusteringMetrics from analyze_clustering_quality.
    """
    print("\n" + "=" * 50)
    print("CLUSTERING QUALITY REPORT")
    print("=" * 50)

    print("\nIntra-class distances (lower is better):")
    for label, dist in sorted(metrics.intra_class_distances.items()):
        print(f"  {label}: {dist:.3f}")
    print(f"  Mean: {metrics.mean_intra:.3f}")

    print("\nInter-class distances (higher is better):")
    for l1, l2, dist in sorted(metrics.inter_class_distances, key=lambda x: -x[2]):
        print(f"  {l1} <-> {l2}: {dist:.3f}")
    print(f"  Mean: {metrics.mean_inter:.3f}")

    print(f"\nSeparation ratio: {metrics.separation_ratio:.2f}")
    print("  (>1 = good separation, <1 = categories overlap)")
    print("=" * 50)
