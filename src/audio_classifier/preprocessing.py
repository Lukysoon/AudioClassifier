"""
Audio preprocessing module - loading, resampling, and normalization.
"""

import librosa
import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torchaudio
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

from .config import AudioConfig


@dataclass
class AudioChunk:
    """Represents a single chunk of audio."""

    waveform: np.ndarray
    """Preprocessed waveform for this chunk."""

    chunk_index: int
    """Zero-based index of this chunk within the source file."""

    start_time_seconds: float
    """Start time of this chunk in the original audio."""

    end_time_seconds: float
    """End time of this chunk in the original audio."""

    is_padded: bool = False
    """Whether this chunk was zero-padded (for final short chunks)."""


class AudioPreprocessor:
    """Handles audio loading, resampling, and normalization."""

    def __init__(self, config: AudioConfig | None = None):
        """
        Initialize the preprocessor.

        Args:
            config: Audio configuration. Uses defaults if None.
        """
        self.config = config or AudioConfig()

    def load(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess a single audio file.

        Performs the following transformations:
        1. Load raw audio from file
        2. Convert stereo to mono (average channels)
        3. Resample to target sample rate (16kHz for HuBERT)
        4. Truncate to max duration if configured
        5. Normalize to [-1, 1] range if configured

        Args:
            audio_path: Path to the audio file.

        Returns:
            Preprocessed waveform as 1D numpy array (float32).

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If audio loading fails.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono (average channels)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.config.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.target_sample_rate
            )
            waveform = resampler(waveform)

        # Noise reduction (before silence removal for better silence detection)
        waveform_np = waveform.squeeze().numpy()
        waveform_np = self._reduce_noise(waveform_np)

        # Remove silence
        waveform_np = self._remove_silence(waveform_np)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)

        # Truncate to max duration
        if self.config.max_duration_seconds is not None:
            max_samples = int(self.config.max_duration_seconds * self.config.target_sample_rate)
            waveform = waveform[:, :max_samples]

        # Normalize to [-1, 1]
        if self.config.normalize:
            max_val = waveform.abs().max()
            if max_val > 0:
                waveform = waveform / (max_val + 1e-8)

        return waveform.squeeze().numpy().astype(np.float32)

    def load_chunked(self, audio_path: Union[str, Path]) -> List[AudioChunk]:
        """
        Load and preprocess an audio file, splitting into fixed-length chunks.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of AudioChunk objects. Returns single chunk if chunking disabled.

        Raises:
            FileNotFoundError: If audio file doesn't exist.
            RuntimeError: If audio loading fails.
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load audio
        waveform, sample_rate = torchaudio.load(str(audio_path))

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if necessary
        if sample_rate != self.config.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.target_sample_rate
            )
            waveform = resampler(waveform)

        # Noise reduction (before silence removal for better silence detection)
        waveform_np = waveform.squeeze().numpy()
        waveform_np = self._reduce_noise(waveform_np)

        # Remove silence before any chunking
        waveform_np = self._remove_silence(waveform_np)
        waveform = torch.from_numpy(waveform_np).unsqueeze(0)

        # If chunking not enabled, return single chunk (backward compatible)
        if not self.config.chunking_enabled:
            # Apply max_duration truncation
            if self.config.max_duration_seconds is not None:
                max_samples = int(self.config.max_duration_seconds * self.config.target_sample_rate)
                waveform = waveform[:, :max_samples]

            # Normalize
            processed = self._normalize(waveform.squeeze().numpy())

            total_duration = len(processed) / self.config.target_sample_rate
            return [AudioChunk(
                waveform=processed,
                chunk_index=0,
                start_time_seconds=0.0,
                end_time_seconds=total_duration,
                is_padded=False
            )]

        # Chunking is enabled - split into fixed-length chunks
        return self._split_into_chunks(waveform.squeeze())

    def _split_into_chunks(self, waveform: torch.Tensor) -> List[AudioChunk]:
        """
        Split waveform into fixed-length chunks.

        Args:
            waveform: 1D tensor of audio samples.

        Returns:
            List of AudioChunk objects.
        """
        chunk_samples = int(self.config.chunk_duration_seconds * self.config.target_sample_rate)
        min_samples = int(self.config.min_chunk_duration_seconds * self.config.target_sample_rate)
        total_samples = waveform.shape[0]

        chunks = []
        chunk_index = 0

        for start_sample in range(0, total_samples, chunk_samples):
            end_sample = min(start_sample + chunk_samples, total_samples)
            chunk_waveform = waveform[start_sample:end_sample]
            chunk_length = chunk_waveform.shape[0]

            is_final_chunk = (end_sample >= total_samples)
            is_short = (chunk_length < chunk_samples)
            is_padded = False

            if is_short and is_final_chunk:
                if self.config.chunk_handling == "discard":
                    continue
                elif self.config.chunk_handling == "keep":
                    if chunk_length < min_samples:
                        continue
                elif self.config.chunk_handling == "pad":
                    padding = torch.zeros(chunk_samples - chunk_length)
                    chunk_waveform = torch.cat([chunk_waveform, padding])
                    is_padded = True

            # Normalize the chunk
            processed = self._normalize(chunk_waveform.numpy())

            start_time = start_sample / self.config.target_sample_rate
            end_time = end_sample / self.config.target_sample_rate

            chunks.append(AudioChunk(
                waveform=processed,
                chunk_index=chunk_index,
                start_time_seconds=start_time,
                end_time_seconds=end_time,
                is_padded=is_padded
            ))
            chunk_index += 1

        return chunks

    def _reduce_noise(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply spectral gating noise reduction.

        Args:
            waveform: Audio samples as numpy array.

        Returns:
            Waveform with reduced noise.
        """
        if not self.config.noise_reduction_enabled:
            return waveform

        import noisereduce as nr

        return nr.reduce_noise(
            y=waveform,
            sr=self.config.target_sample_rate,
            stationary=self.config.noise_reduction_stationary
        )

    def _normalize(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize waveform to [-1, 1] range if configured.

        Args:
            waveform: Audio samples as numpy array.

        Returns:
            Normalized waveform as float32.
        """
        if self.config.normalize:
            max_val = np.abs(waveform).max()
            if max_val > 0:
                waveform = waveform / (max_val + 1e-8)
        return waveform.astype(np.float32)

    def _remove_silence(self, waveform: np.ndarray) -> np.ndarray:
        """
        Remove all silent segments from waveform.

        Uses librosa.effects.split() to detect non-silent intervals
        and concatenates them.

        Args:
            waveform: Audio samples as numpy array.

        Returns:
            Waveform with silent segments removed.
        """
        if not self.config.silence_removal_enabled:
            return waveform

        # Detect non-silent intervals
        intervals = librosa.effects.split(
            waveform,
            top_db=self.config.silence_threshold_db,
            frame_length=2048,
            hop_length=512
        )

        if len(intervals) == 0:
            return waveform  # No non-silent parts found, keep original

        # Concatenate non-silent segments
        non_silent_parts = [waveform[start:end] for start, end in intervals]
        return np.concatenate(non_silent_parts)

    def is_valid_audio(self, path: Path) -> bool:
        """
        Check if file has a valid audio extension.

        Args:
            path: Path to check.

        Returns:
            True if file extension is in supported extensions list.
        """
        return path.suffix.lower() in self.config.extensions

    def get_duration(self, audio_path: Union[str, Path]) -> float:
        """
        Get duration of audio file in seconds without loading full waveform.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        info = torchaudio.info(str(audio_path))
        return info.num_frames / info.sample_rate

    def split_and_save_chunks(
        self,
        audio_path: Union[str, Path],
        output_dir: Path,
        category: str
    ) -> List[Path]:
        """
        Split audio file into chunks and save as .wav files.

        Args:
            audio_path: Path to the source audio file.
            output_dir: Base output directory.
            category: Category name (subdirectory).

        Returns:
            List of paths to saved chunk files.
        """
        audio_path = Path(audio_path)
        chunks = self.load_chunked(audio_path)

        # Create output category directory
        category_dir = output_dir / category
        category_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        base_name = audio_path.stem

        for chunk in chunks:
            # Generate chunk filename
            if len(chunks) == 1:
                chunk_filename = f"{base_name}.wav"
            else:
                chunk_filename = f"{base_name}_chunk_{chunk.chunk_index:03d}.wav"

            chunk_path = category_dir / chunk_filename

            # Convert to int16 for wav file
            waveform_int16 = (chunk.waveform * 32767).astype(np.int16)

            # Save as wav
            wavfile.write(
                str(chunk_path),
                self.config.target_sample_rate,
                waveform_int16
            )

            saved_paths.append(chunk_path)

        return saved_paths


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    chunk_duration: float = 10.0,
    chunk_handling: str = "discard",
    min_chunk_duration: float = 0.5,
    silence_removal: bool = True,
    silence_threshold_db: float = 40.0,
    delete_originals: bool = False,
    show_progress: bool = True
) -> int:
    """
    Preprocess entire dataset: split audio files into chunks and save as .wav.

    Args:
        input_dir: Directory containing category folders with audio files.
        output_dir: Output directory for processed chunks.
        chunk_duration: Duration of each chunk in seconds.
        chunk_handling: How to handle final short chunks ('pad', 'discard', 'keep').
        min_chunk_duration: Minimum chunk duration for 'keep' mode.
        silence_removal: Whether to remove silence before chunking.
        silence_threshold_db: Silence threshold in dB.
        delete_originals: Whether to delete original files after processing.
        show_progress: Whether to show progress output.

    Returns:
        Total number of chunks created.
    """
    from tqdm import tqdm

    # Configure preprocessor
    config = AudioConfig(
        chunking_enabled=True,
        chunk_duration_seconds=chunk_duration,
        chunk_handling=chunk_handling,
        min_chunk_duration_seconds=min_chunk_duration,
        silence_removal_enabled=silence_removal,
        silence_threshold_db=silence_threshold_db
    )
    preprocessor = AudioPreprocessor(config)

    # Find all audio files
    audio_files = []
    for audio_file in input_dir.rglob("*"):
        if audio_file.is_file() and preprocessor.is_valid_audio(audio_file):
            category = audio_file.parent.name
            audio_files.append((audio_file, category))

    if not audio_files:
        raise ValueError(f"No audio files found in {input_dir}")

    if show_progress:
        print(f"Found {len(audio_files)} audio files")

    # Process files
    total_chunks = 0
    failed_files = []
    files_to_delete = []

    iterator = tqdm(audio_files, desc="Processing") if show_progress else audio_files

    for audio_path, category in iterator:
        try:
            saved_paths = preprocessor.split_and_save_chunks(
                audio_path,
                output_dir,
                category
            )
            total_chunks += len(saved_paths)

            if delete_originals:
                files_to_delete.append(audio_path)

        except Exception as e:
            failed_files.append((audio_path, str(e)))

    # Delete originals if requested
    if delete_originals and files_to_delete:
        if show_progress:
            print(f"\nDeleting {len(files_to_delete)} original files...")
        for file_path in files_to_delete:
            file_path.unlink()

    # Report failures
    if failed_files and show_progress:
        print(f"\nWarning: {len(failed_files)} files failed to process:")
        for path, error in failed_files[:5]:
            print(f"  - {path.name}: {error}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")

    if show_progress:
        print(f"\nCreated {total_chunks} chunks in {output_dir}")

    return total_chunks
