#!/usr/bin/env python3
"""
Script pro parsování audia z Parquet souboru.
Parquet soubor obsahuje sloupce: audio_id, audio (a případně další).

Použití:
    python3 parse_audio_parquet.py --input data.parquet --output_dir ./audio_files
    python3 parse_audio_parquet.py --input data.parquet --output_dir ./audio_files --format wav
    python3 parse_audio_parquet.py --input data.parquet --info
"""

import argparse
import os
import sys
import struct
import wave
import io
from pathlib import Path

import pandas as pd
import numpy as np


def load_parquet(path: str) -> pd.DataFrame:
    """Načte parquet soubor a vrátí DataFrame."""
    print(f"[INFO] Načítám parquet soubor: {path}")
    df = pd.read_parquet(path)
    print(f"[INFO] Načteno {len(df)} řádků, sloupce: {list(df.columns)}")
    return df


def print_info(df: pd.DataFrame):
    """Vypíše informace o datasetu."""
    print("\n=== INFORMACE O DATASETU ===")
    print(f"Počet záznamů : {len(df)}")
    print(f"Sloupce       : {list(df.columns)}")
    print(f"\nTypy sloupců:")
    for col in df.columns:
        print(f"  {col}: {df[col].dtype}")
    
    if "audio_id" in df.columns:
        print(f"\nUkázkové audio_id: {df['audio_id'].head(5).tolist()}")
    
    if "audio" in df.columns:
        sample = df["audio"].iloc[0]
        print(f"\nTyp audio dat (1. záznam): {type(sample)}")
        if isinstance(sample, dict):
            print(f"  Klíče audio dict: {list(sample.keys())}")
        elif isinstance(sample, (bytes, bytearray)):
            print(f"  Délka bytes: {len(sample)}")
        elif isinstance(sample, np.ndarray):
            print(f"  NumPy array shape: {sample.shape}, dtype: {sample.dtype}")


def extract_audio_bytes(audio_data) -> bytes:
    """
    Extrahuje surová audio data z různých formátů.
    Podporuje: bytes, dict (HuggingFace format), numpy array.
    """
    if isinstance(audio_data, (bytes, bytearray)):
        return bytes(audio_data)
    
    # HuggingFace datasets formát: {'bytes': b'...', 'path': '...'}
    if isinstance(audio_data, dict):
        if "bytes" in audio_data and audio_data["bytes"] is not None:
            return bytes(audio_data["bytes"])
        # Někdy je audio uloženo jako array + sampling_rate
        if "array" in audio_data:
            return numpy_to_wav_bytes(
                audio_data["array"],
                sample_rate=audio_data.get("sampling_rate", 16000)
            )
    
    # NumPy array (raw PCM float)
    if isinstance(audio_data, np.ndarray):
        return numpy_to_wav_bytes(audio_data)
    
    raise ValueError(f"Nepodporovaný formát audio dat: {type(audio_data)}")


def numpy_to_wav_bytes(array: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Převede numpy array (float32) na WAV bytes."""
    # Normalizace na int16
    if array.dtype in (np.float32, np.float64):
        array = np.clip(array, -1.0, 1.0)
        array = (array * 32767).astype(np.int16)
    elif array.dtype != np.int16:
        array = array.astype(np.int16)
    
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(array.tobytes())
    return buf.getvalue()


def get_file_extension(audio_data) -> str:
    """Odhadne příponu souboru podle dat."""
    if isinstance(audio_data, dict):
        if "path" in audio_data and audio_data["path"]:
            ext = Path(audio_data["path"]).suffix
            if ext:
                return ext.lstrip(".")
        if "array" in audio_data:
            return "wav"
    
    raw = None
    if isinstance(audio_data, (bytes, bytearray)):
        raw = bytes(audio_data[:12])
    elif isinstance(audio_data, dict) and "bytes" in audio_data:
        raw = bytes(audio_data["bytes"][:12])
    
    if raw:
        if raw[:4] == b"RIFF":
            return "wav"
        if raw[:3] == b"ID3" or raw[:2] == b"\xff\xfb":
            return "mp3"
        if raw[:4] == b"fLaC":
            return "flac"
        if raw[:4] == b"OggS":
            return "ogg"
    
    return "wav"  # fallback


def save_audio(audio_bytes: bytes, output_path: str):
    """Uloží audio bytes do souboru."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(audio_bytes)


def parse_and_export(df: pd.DataFrame, output_dir: str, force_format: str = None, limit: int = None):
    """Hlavní funkce pro export audio souborů z DataFrame."""
    os.makedirs(output_dir, exist_ok=True)
    
    if "audio" not in df.columns:
        print("[ERROR] Sloupec 'audio' nebyl nalezen v parquet souboru!")
        sys.exit(1)
    
    audio_id_col = "audio_id" if "audio_id" in df.columns else None
    
    rows = df.iterrows()
    if limit:
        import itertools
        rows = itertools.islice(rows, limit)
        print(f"[INFO] Exportuji prvních {limit} záznamů...")
    else:
        print(f"[INFO] Exportuji {len(df)} záznamů...")
    
    success = 0
    errors = 0
    
    for idx, row in rows:
        audio_data = row["audio"]
        audio_id = row[audio_id_col] if audio_id_col else idx
        
        try:
            audio_bytes = extract_audio_bytes(audio_data)
            
            if force_format:
                ext = force_format
            else:
                ext = get_file_extension(audio_data)
            
            filename = f"{audio_id}.{ext}"
            output_path = os.path.join(output_dir, filename)
            save_audio(audio_bytes, output_path)
            
            success += 1
            if success % 100 == 0:
                print(f"  Exportováno: {success} souborů...")
        
        except Exception as e:
            print(f"[WARNING] Chyba u záznamu {audio_id}: {e}")
            errors += 1
    
    print(f"\n=== VÝSLEDEK ===")
    print(f"Úspěšně exportováno : {success} souborů")
    print(f"Chyby               : {errors}")
    print(f"Výstupní adresář    : {os.path.abspath(output_dir)}")


def main():
    parser = argparse.ArgumentParser(
        description="Parsování audia z Parquet souboru",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", "-i", required=True, help="Cesta k parquet souboru")
    parser.add_argument("--output_dir", "-o", default="./audio_output", help="Výstupní adresář pro audio soubory")
    parser.add_argument("--format", "-f", default=None, choices=["wav", "mp3", "flac", "ogg"],
                        help="Vynutit formát výstupních souborů (výchozí: autodetekce)")
    parser.add_argument("--info", action="store_true", help="Pouze zobrazit informace o datasetu")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Exportovat pouze prvních N záznamů")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"[ERROR] Soubor nenalezen: {args.input}")
        sys.exit(1)
    
    df = load_parquet(args.input)
    
    if args.info:
        print_info(df)
        return
    
    parse_and_export(df, args.output_dir, force_format=args.format, limit=args.limit)


if __name__ == "__main__":
    main()
