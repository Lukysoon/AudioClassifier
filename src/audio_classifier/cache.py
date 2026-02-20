"""
Embedding cache for resumable audio processing.

Stores extracted embeddings to disk so that already-processed files
can be skipped on subsequent runs.
"""

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import PipelineConfig


CACHE_FILENAME = "embedding_cache.pkl"


def compute_config_hash(config: PipelineConfig) -> str:
    """Compute hash of config fields that affect embedding output.

    Changes to these fields invalidate the entire cache.
    """
    relevant = {
        "audio.target_sample_rate": config.audio.target_sample_rate,
        "audio.normalize": config.audio.normalize,
        "audio.max_duration_seconds": config.audio.max_duration_seconds,
        "audio.chunking_enabled": config.audio.chunking_enabled,
        "audio.chunk_duration_seconds": config.audio.chunk_duration_seconds,
        "audio.chunk_handling": config.audio.chunk_handling,
        "audio.min_chunk_duration_seconds": config.audio.min_chunk_duration_seconds,
        "audio.silence_removal_enabled": config.audio.silence_removal_enabled,
        "audio.silence_threshold_db": config.audio.silence_threshold_db,
        "model.model_name": config.model.model_name,
        "model.pooling": config.model.pooling,
        "model.layer": config.model.layer,
    }
    raw = json.dumps(relevant, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class EmbeddingCache:
    """File-based cache of audio embeddings.

    Cache format (pickle):
        {
            "config_hash": str,
            "entries": {
                resolved_file_path: {
                    "file_mtime": float,
                    "file_size": int,
                    "samples": [
                        {
                            "embedding": np.ndarray,
                            "chunk_index": int | None,
                            "start_time_seconds": float | None,
                            "end_time_seconds": float | None,
                            "is_padded": bool,
                        },
                        ...
                    ]
                }
            }
        }
    """

    def __init__(self, cache_dir: Path, config_hash: str):
        self.cache_path = cache_dir / CACHE_FILENAME
        self.config_hash = config_hash
        self.entries: Dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Load cache from disk. Discard if config hash mismatches."""
        if not self.cache_path.exists():
            return

        try:
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)

            if data.get("config_hash") != self.config_hash:
                print("Cache invalidated: config changed. Re-processing all files.")
                return

            self.entries = data.get("entries", {})
            print(f"Loaded embedding cache: {len(self.entries)} files cached")
        except Exception as e:
            print(f"Warning: Could not load cache ({e}). Starting fresh.")
            self.entries = {}

    def _save(self) -> None:
        """Save cache to disk."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "config_hash": self.config_hash,
            "entries": self.entries,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def lookup(self, file_path: str) -> Optional[List[dict]]:
        """Look up cached samples for a file.

        Returns list of sample dicts if cache hit and file unchanged,
        None if cache miss or file was modified.
        """
        key = str(Path(file_path).resolve())
        entry = self.entries.get(key)
        if entry is None:
            return None

        try:
            stat = os.stat(file_path)
            if stat.st_mtime != entry["file_mtime"] or stat.st_size != entry["file_size"]:
                return None
        except OSError:
            return None

        return entry["samples"]

    def store(self, file_path: str, samples: List[dict]) -> None:
        """Store embedding results for a file and save to disk immediately."""
        key = str(Path(file_path).resolve())
        try:
            stat = os.stat(file_path)
        except OSError:
            return

        self.entries[key] = {
            "file_mtime": stat.st_mtime,
            "file_size": stat.st_size,
            "samples": samples,
        }

        try:
            self._save()
        except Exception as e:
            print(f"Warning: Could not save cache ({e})")

    def load_all_entries(self) -> Dict[str, List[dict]]:
        """Return all cached entries without filesystem checks.

        Returns:
            Dict mapping resolved file paths to their list of sample dicts.
        """
        return {path: entry["samples"] for path, entry in self.entries.items()}

    def clear(self) -> None:
        """Delete the cache file from disk."""
        if self.cache_path.exists():
            self.cache_path.unlink()
            print(f"Cache cleared: {self.cache_path}")
        self.entries = {}
