#!/usr/bin/env python3
"""
Stažení Multilingual LibriSpeech (MLS) - 10h limited supervision sety
pro všechny dostupné jazyky. Audio se uloží jako MP3.

Struktura výstupu:
    data/
    ├── german/
    │   ├── 0001.mp3
    │   ├── 0002.mp3
    │   ├── ...
    │   └── transcripts.csv
    ├── english/
    │   ├── 0001.mp3
    │   ├── ...
    │   └── transcripts.csv
    └── ...

transcripts.csv obsahuje: filename, transcript, speaker_id, duration_s

Instalace závislostí:
    pip3 install datasets soundfile pydub

Na macOS je potřeba mít nainstalovaný ffmpeg:
    brew install ffmpeg
"""

import os
import csv
import io
import shutil
import numpy as np
from datasets import load_dataset
from pydub import AudioSegment

# --- Konfigurace ---
OUTPUT_DIR = "data"
SAMPLE_RATE = 16000  # MLS má 16kHz
LANGUAGES = [
    "german",
    "dutch",
    "french",
    "spanish",
    "italian",
    "portuguese",
    "polish",
]
SPLIT = "train.9h"  # ~10h limited supervision set
# -------------------


def numpy_to_mp3(audio_array: np.ndarray, sample_rate: int, output_path: str):
    """Převede numpy audio pole na MP3 soubor."""
    # Převod float32 -> int16
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        audio_int16 = (audio_array * 32767).astype(np.int16)
    else:
        audio_int16 = audio_array.astype(np.int16)

    # Vytvoření AudioSegment z raw dat
    audio_segment = AudioSegment(
        data=audio_int16.tobytes(),
        sample_width=2,  # int16 = 2 bytes
        frame_rate=sample_rate,
        channels=1,
    )

    # Export jako MP3
    audio_segment.export(output_path, format="mp3", bitrate="128k")


def download_language(language: str):
    """Stáhne a uloží 10h set pro jeden jazyk."""
    lang_dir = os.path.join(OUTPUT_DIR, language)
    os.makedirs(lang_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Stahování: {language} (split: {SPLIT})")
    print(f"  Cíl: {lang_dir}/")
    print(f"{'='*60}")

    # Stažení datasetu z Hugging Face
    cache_dir = os.path.join(OUTPUT_DIR, ".hf_cache")
    try:
        dataset = load_dataset(
            "facebook/multilingual_librispeech",
            language,
            split=SPLIT,
            trust_remote_code=True,
            cache_dir=cache_dir,
        )
    except Exception as e:
        print(f"  ❌ Chyba při stahování {language}: {e}")
        return

    total = len(dataset)
    print(f"  Počet nahrávek: {total}")

    # CSV soubor s transkripty
    csv_path = os.path.join(lang_dir, "transcripts.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["filename", "transcript", "speaker_id", "duration_s"])

        for i, sample in enumerate(dataset):
            # Název souboru
            filename = f"{i:05d}.mp3"
            output_path = os.path.join(lang_dir, filename)

            # Audio data
            audio = sample["audio"]
            audio_array = np.array(audio["array"], dtype=np.float32)
            sample_rate = audio["sampling_rate"]
            duration_s = round(len(audio_array) / sample_rate, 2)

            # Transkript
            transcript = sample.get("transcript") or sample.get("text", "")

            # Speaker ID
            speaker_id = sample.get("speaker_id", "")

            # Uložení MP3
            numpy_to_mp3(audio_array, sample_rate, output_path)

            # Zápis do CSV
            writer.writerow([filename, transcript, speaker_id, duration_s])

            # Progress
            if (i + 1) % 100 == 0 or (i + 1) == total:
                print(f"  [{i+1}/{total}] uloženo ({(i+1)/total*100:.0f}%)")

    print(f"  ✅ {language}: {total} nahrávek uloženo do {lang_dir}/")
    print(f"     Transkripty: {csv_path}")

    # Smazání HF cache
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"  🗑️  Cache smazána ({cache_dir})")


def main():
    print("=" * 60)
    print("  MLS 10h Limited Supervision - Download Script")
    print(f"  Jazyky: {', '.join(LANGUAGES)}")
    print(f"  Formát: MP3 (128kbps)")
    print(f"  Výstup: {OUTPUT_DIR}/<jazyk>/")
    print("=" * 60)

    for language in LANGUAGES:
        download_language(language)

    print(f"\n{'='*60}")
    print("  🎉 Hotovo! Všechny jazyky staženy.")
    print(f"  Výstupní adresář: {OUTPUT_DIR}/")
    print("=" * 60)

    # Souhrn
    print("\nSouhrn:")
    for language in LANGUAGES:
        lang_dir = os.path.join(OUTPUT_DIR, language)
        if os.path.exists(lang_dir):
            mp3_count = len([f for f in os.listdir(lang_dir) if f.endswith(".mp3")])
            print(f"  {language:>12}: {mp3_count} nahrávek")


if __name__ == "__main__":
    main()