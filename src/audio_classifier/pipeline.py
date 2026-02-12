"""
Main pipeline orchestration for the Audio Classifier.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Union
from tqdm import tqdm

from .config import PipelineConfig
from .preprocessing import AudioPreprocessor
from .feature_extraction import HuBERTExtractor
from .dimensionality import DimensionalityReducer
from .visualization import Visualizer
from .analysis import (
    calculate_centroid_distances,
    analyze_clustering_quality,
    print_clustering_report
)


@dataclass
class AudioSample:
    """Represents a single audio sample with its embeddings."""

    file_path: str
    """Path to the original audio file."""

    label: str
    """Category label (typically folder name)."""

    embedding: Optional[np.ndarray] = None
    """768-dimensional HuBERT embedding."""

    coords_3d: Optional[np.ndarray] = None
    """3D coordinates after UMAP reduction."""

    chunk_index: Optional[int] = None
    """Chunk index within the source file (None if not chunked)."""

    start_time_seconds: Optional[float] = None
    """Start time of this chunk in the original audio."""

    end_time_seconds: Optional[float] = None
    """End time of this chunk in the original audio."""

    is_padded: bool = False
    """Whether this chunk was zero-padded."""

    @property
    def display_name(self) -> str:
        """Generate display name for visualization."""
        filename = Path(self.file_path).stem
        if self.chunk_index is not None:
            return f"{filename}_chunk_{self.chunk_index:03d}"
        return filename

    @property
    def is_chunked(self) -> bool:
        """Check if this sample is a chunk of a larger file."""
        return self.chunk_index is not None


class AudioClassifierPipeline:
    """
    Main orchestration class for the audio classification pipeline.

    This class coordinates:
    1. Loading audio files from directory structure
    2. Extracting HuBERT embeddings
    3. Reducing dimensions with UMAP
    4. Visualizing results in 3D

    Example usage:
        >>> pipeline = AudioClassifierPipeline()
        >>> df = pipeline.run("./data")
        >>> pipeline.visualize_and_save()
    """

    def __init__(self, config: PipelineConfig | None = None):
        """
        Initialize the pipeline with configuration.

        Args:
            config: Pipeline configuration. Uses defaults if None.
        """
        self.config = config or PipelineConfig()

        # Initialize components (lazy loading for extractor)
        self.preprocessor = AudioPreprocessor(self.config.audio)
        self._extractor: Optional[HuBERTExtractor] = None
        self.reducer = DimensionalityReducer(self.config.umap)
        self.visualizer = Visualizer(self.config.visualization)

        # Data storage
        self.samples: List[AudioSample] = []

    @property
    def extractor(self) -> HuBERTExtractor:
        """Lazy load the HuBERT extractor (downloads model on first use)."""
        if self._extractor is None:
            self._extractor = HuBERTExtractor(self.config.model)
        return self._extractor

    def load_dataset(self, data_dir: Union[str, Path]) -> None:
        """
        Load dataset from directory structure.

        Expected structure:
            data_dir/
            ├── category_a/
            │   ├── audio1.wav
            │   └── audio2.mp3
            └── category_b/
                └── audio3.flac

        The folder name becomes the label (color in visualization).

        Args:
            data_dir: Root directory containing category folders.
        """
        data_path = Path(data_dir)

        if not data_path.exists():
            raise FileNotFoundError(f"Data directory not found: {data_path}")

        # Clear any existing samples
        self.samples = []

        # Find all audio files
        for audio_file in data_path.rglob("*"):
            if audio_file.is_file() and self.preprocessor.is_valid_audio(audio_file):
                # Use parent folder name as label
                label = audio_file.parent.name

                self.samples.append(AudioSample(
                    file_path=str(audio_file),
                    label=label
                ))

        if not self.samples:
            raise ValueError(
                f"No audio files found in {data_path}. "
                f"Supported extensions: {self.config.audio.extensions}"
            )

        # Print summary
        labels = set(s.label for s in self.samples)
        print(f"\nLoaded {len(self.samples)} audio files")
        print(f"Categories ({len(labels)}): {sorted(labels)}")

        for label in sorted(labels):
            count = sum(1 for s in self.samples if s.label == label)
            print(f"  - {label}: {count} files")

    def extract_embeddings(self, show_progress: bool = True) -> None:
        """
        Extract HuBERT embeddings for all loaded samples.

        When chunking is enabled, each audio file is split into chunks
        and each chunk gets its own embedding.

        Args:
            show_progress: Whether to show progress bar.
        """
        if not self.samples:
            raise RuntimeError("No samples loaded. Call load_dataset first.")

        print("\nExtracting HuBERT embeddings...")

        if self.config.audio.chunking_enabled:
            self._extract_embeddings_chunked(show_progress)
        else:
            self._extract_embeddings_simple(show_progress)

    def _extract_embeddings_simple(self, show_progress: bool) -> None:
        """Original non-chunked embedding extraction."""
        iterator = self.samples
        if show_progress:
            iterator = tqdm(self.samples, desc="Processing audio")

        failed = []
        for sample in iterator:
            try:
                waveform = self.preprocessor.load(sample.file_path)
                sample.embedding = self.extractor.extract(waveform)
            except Exception as e:
                failed.append((sample.file_path, str(e)))

        # Remove failed samples
        self.samples = [s for s in self.samples if s.embedding is not None]

        if failed:
            print(f"\nWarning: {len(failed)} files failed to process:")
            for path, error in failed[:5]:
                print(f"  - {Path(path).name}: {error}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        print(f"Successfully extracted {len(self.samples)} embeddings")

    def _extract_embeddings_chunked(self, show_progress: bool) -> None:
        """Chunked embedding extraction - processes each file and creates multiple samples."""
        # Get unique file paths with their labels
        file_labels = {s.file_path: s.label for s in self.samples}
        file_paths = list(file_labels.keys())

        iterator = file_paths
        if show_progress:
            iterator = tqdm(file_paths, desc="Processing files")

        failed_files = []
        new_samples = []

        for file_path in iterator:
            try:
                label = file_labels[file_path]
                chunks = self.preprocessor.load_chunked(file_path)

                for chunk in chunks:
                    embedding = self.extractor.extract(chunk.waveform)
                    new_samples.append(AudioSample(
                        file_path=file_path,
                        label=label,
                        embedding=embedding,
                        chunk_index=chunk.chunk_index,
                        start_time_seconds=chunk.start_time_seconds,
                        end_time_seconds=chunk.end_time_seconds,
                        is_padded=chunk.is_padded
                    ))

            except Exception as e:
                failed_files.append((file_path, str(e)))

        self.samples = new_samples

        if failed_files:
            print(f"\nWarning: {len(failed_files)} files failed to process:")
            for path, error in failed_files[:5]:
                print(f"  - {Path(path).name}: {error}")
            if len(failed_files) > 5:
                print(f"  ... and {len(failed_files) - 5} more")

        print(f"Successfully extracted {len(self.samples)} chunk embeddings from {len(file_paths) - len(failed_files)} files")

    def reduce_dimensions(self) -> None:
        """Reduce embeddings from 768D to 3D using UMAP."""
        if not self.samples:
            raise RuntimeError("No samples with embeddings. Call extract_embeddings first.")

        if not all(s.embedding is not None for s in self.samples):
            raise RuntimeError("Some samples missing embeddings.")

        print("\nReducing dimensions with UMAP...")

        # Stack all embeddings
        embeddings = np.array([s.embedding for s in self.samples])

        # Fit and transform
        coords_3d = self.reducer.fit_transform(embeddings)

        # Assign coordinates to samples
        for sample, coords in zip(self.samples, coords_3d):
            sample.coords_3d = coords

    def get_dataframe(self) -> pd.DataFrame:
        """
        Export data as pandas DataFrame.

        Returns:
            DataFrame with columns: file, label, x, y, z, display_name,
            and optionally chunk metadata (chunk_index, start_time, end_time)
        """
        data = []
        for s in self.samples:
            if s.coords_3d is not None:
                row = {
                    "file": s.file_path,
                    "label": s.label,
                    "x": s.coords_3d[0],
                    "y": s.coords_3d[1],
                    "z": s.coords_3d[2],
                    "display_name": s.display_name,
                }

                if s.is_chunked:
                    row.update({
                        "chunk_index": s.chunk_index,
                        "start_time": s.start_time_seconds,
                        "end_time": s.end_time_seconds,
                        "is_padded": s.is_padded,
                    })

                data.append(row)

        return pd.DataFrame(data)

    def run(self, data_dir: Union[str, Path]) -> pd.DataFrame:
        """
        Run the complete pipeline end-to-end.

        Args:
            data_dir: Root directory containing category folders.

        Returns:
            DataFrame with file paths, labels, and 3D coordinates.
        """
        self.load_dataset(data_dir)
        self.extract_embeddings()
        self.reduce_dimensions()
        return self.get_dataframe()

    def visualize_and_save(self, prefix: str = "audio_classifier") -> None:
        """
        Generate and save all visualizations.

        Creates:
        - {prefix}_3d.html: Interactive 3D scatter plot
        - {prefix}_distances.html: Category distance heatmap

        Also prints a clustering quality report.

        Args:
            prefix: Filename prefix for output files.
        """
        df = self.get_dataframe()

        if df.empty:
            raise RuntimeError("No data to visualize. Run the pipeline first.")

        print("\nGenerating visualizations...")

        # 3D scatter plot
        fig_3d = self.visualizer.plot_3d_scatter(df, title="Audio Embeddings - 3D Visualization")
        self.visualizer.save_and_open(fig_3d, f"{prefix}_3d.html")

        # Distance heatmap
        dist_matrix = calculate_centroid_distances(df)
        fig_dist = self.visualizer.plot_distance_heatmap(dist_matrix, title="Category Distances")
        self.visualizer.save_and_open(fig_dist, f"{prefix}_distances.html")

        # Print clustering analysis
        coords = df[["x", "y", "z"]].values
        labels = df["label"].tolist()
        metrics = analyze_clustering_quality(coords, labels)
        print_clustering_report(metrics)

    def save_embeddings(self, output_path: Union[str, Path]) -> None:
        """
        Save embeddings to a numpy file for later use.

        Args:
            output_path: Path to save the .npz file.
        """
        valid_samples = [s for s in self.samples if s.embedding is not None]

        embeddings = np.array([s.embedding for s in valid_samples])
        labels = [s.label for s in valid_samples]
        files = [s.file_path for s in valid_samples]
        display_names = [s.display_name for s in valid_samples]

        # Chunk metadata (None values for non-chunked samples)
        chunk_indices = [s.chunk_index for s in valid_samples]
        start_times = [s.start_time_seconds for s in valid_samples]
        end_times = [s.end_time_seconds for s in valid_samples]

        np.savez(
            output_path,
            embeddings=embeddings,
            labels=labels,
            files=files,
            display_names=display_names,
            chunk_indices=chunk_indices,
            start_times=start_times,
            end_times=end_times
        )
        print(f"Saved embeddings to {output_path}")
