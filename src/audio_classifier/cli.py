"""
Command-line interface for the Audio Classifier.
"""

import argparse
import sys
from pathlib import Path

from .config import PipelineConfig
from .pipeline import AudioClassifierPipeline
from .preprocessing import preprocess_dataset


def main() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Audio Classifier - Visualize audio embeddings in 3D",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - analyze audio files organized in folders
  audio-classifier ./data

  # Split long recordings into 5-second chunks
  audio-classifier ./data --chunk 5.0

  # Chunking with padding for final short chunks
  audio-classifier ./data --chunk 10.0 --chunk-handling pad

  # Keep short final chunks if at least 2 seconds
  audio-classifier ./data --chunk 5.0 --chunk-handling keep --min-chunk 2.0

  # Specify output directory
  audio-classifier ./data --output ./results

  # Use max pooling instead of mean
  audio-classifier ./data --pooling max

  # Adjust UMAP parameters for tighter clusters
  audio-classifier ./data --n-neighbors 10 --min-dist 0.05

  # Use a config file
  audio-classifier ./data --config config.yaml

Data structure:
  Your data directory should be organized as:
    data/
    ├── category_a/       <- folder name = label = color
    │   ├── audio1.wav
    │   └── audio2.mp3
    └── category_b/
        └── audio3.flac
        """
    )

    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing audio files organized as category/audio_files"
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("./output"),
        help="Output directory for visualizations (default: ./output)"
    )

    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Path to YAML configuration file"
    )

    parser.add_argument(
        "--pooling",
        choices=["mean", "max"],
        default="mean",
        help="Pooling strategy for HuBERT embeddings (default: mean)"
    )

    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter - smaller = tighter clusters (default: 15)"
    )

    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter - smaller = denser clusters (default: 0.1)"
    )

    parser.add_argument(
        "--max-duration",
        type=float,
        default=30.0,
        help="Maximum audio duration in seconds (default: 30)"
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="audio_classifier",
        help="Output filename prefix (default: audio_classifier)"
    )

    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Don't automatically open visualizations in browser"
    )

    parser.add_argument(
        "--save-embeddings",
        type=Path,
        help="Save embeddings to .npz file for later use"
    )

    # Chunking arguments
    parser.add_argument(
        "--chunk",
        type=float,
        metavar="SECONDS",
        help="Split audio into fixed-length chunks (e.g., --chunk 5.0)"
    )

    parser.add_argument(
        "--chunk-handling",
        choices=["pad", "discard", "keep"],
        default="discard",
        help="How to handle final short chunks: pad, discard (default), keep"
    )

    parser.add_argument(
        "--min-chunk",
        type=float,
        default=0.5,
        help="Minimum chunk duration in seconds for 'keep' mode (default: 0.5)"
    )

    # Silence removal arguments
    parser.add_argument(
        "--no-silence-removal",
        action="store_true",
        help="Disable silence removal (enabled by default)"
    )

    parser.add_argument(
        "--silence-threshold",
        type=float,
        default=40.0,
        help="Silence threshold in dB (default: 40)"
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    # Load or create configuration
    if args.config:
        if not args.config.exists():
            print(f"Error: Config file not found: {args.config}", file=sys.stderr)
            return 1
        config = PipelineConfig.from_yaml(str(args.config))
    else:
        config = PipelineConfig()

    # Override with CLI arguments
    config.model.pooling = args.pooling
    config.umap.n_neighbors = args.n_neighbors
    config.umap.min_dist = args.min_dist
    config.audio.max_duration_seconds = args.max_duration
    config.visualization.output_dir = args.output
    config.visualization.auto_open = not args.no_open

    # Chunking configuration
    if args.chunk:
        config.audio.chunking_enabled = True
        config.audio.chunk_duration_seconds = args.chunk
        config.audio.chunk_handling = args.chunk_handling
        config.audio.min_chunk_duration_seconds = args.min_chunk

    # Silence removal configuration
    config.audio.silence_removal_enabled = not args.no_silence_removal
    config.audio.silence_threshold_db = args.silence_threshold

    # Run pipeline
    try:
        print("=" * 60)
        print("AUDIO CLASSIFIER")
        print("=" * 60)
        print(f"\nInput: {args.data_dir}")
        print(f"Output: {args.output}")
        print(f"Pooling: {config.model.pooling}")
        print(f"UMAP: n_neighbors={config.umap.n_neighbors}, min_dist={config.umap.min_dist}")

        if config.audio.chunking_enabled:
            print(f"Chunking: {config.audio.chunk_duration_seconds}s chunks ({config.audio.chunk_handling} mode)")

        if config.audio.silence_removal_enabled:
            print(f"Silence removal: enabled (threshold: {config.audio.silence_threshold_db}dB)")

        pipeline = AudioClassifierPipeline(config)
        df = pipeline.run(args.data_dir)

        print(f"\nProcessed {len(df)} audio files")
        print(f"Categories: {sorted(df['label'].unique().tolist())}")

        # Save embeddings if requested
        if args.save_embeddings:
            pipeline.save_embeddings(args.save_embeddings)

        # Generate visualizations
        pipeline.visualize_and_save(prefix=args.prefix)

        print(f"\nDone! Output saved to: {config.visualization.output_dir}/")
        return 0

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
