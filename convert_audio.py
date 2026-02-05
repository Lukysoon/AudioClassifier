#!/usr/bin/env python3
"""
Konvertor FLAC/MP3 → WAV pro AudioClassifier.

Použití:
    python convert_audio.py --input ./mls_flac --output ./data --pocet 20
"""

import argparse
import os
from pathlib import Path

import torchaudio


def convert_audio(input_dir: str, output_dir: str, pocet: int = 20):
    """Konvertuje audio soubory na WAV."""

    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        print(f"Chyba: Složka {input_dir} neexistuje")
        return

    print("=" * 60)
    print(f"Konverze audio souborů")
    print(f"Vstup:  {input_path.absolute()}")
    print(f"Výstup: {output_path.absolute()}")
    print(f"Max souborů na jazyk: {pocet}")
    print("=" * 60)

    celkem = 0

    # Projdi všechny podsložky (jazyky)
    for lang_dir in sorted(input_path.iterdir()):
        if not lang_dir.is_dir():
            continue

        jazyk = lang_dir.name
        out_lang_dir = output_path / jazyk
        out_lang_dir.mkdir(parents=True, exist_ok=True)

        # Najdi audio soubory
        audio_files = []
        for ext in ["*.flac", "*.mp3", "*.wav", "*.ogg"]:
            audio_files.extend(lang_dir.rglob(ext))

        if not audio_files:
            print(f"\n{jazyk}: žádné audio soubory")
            continue

        # Zkontroluj existující
        existujici = len(list(out_lang_dir.glob("*.wav")))
        if existujici >= pocet:
            print(f"\n{jazyk}: [skip] již má {existujici} souborů")
            celkem += existujici
            continue

        print(f"\n{jazyk}: konvertuji...")

        for i, audio_file in enumerate(sorted(audio_files)[:pocet]):
            out_file = out_lang_dir / f"{jazyk}_{i:04d}.wav"

            try:
                waveform, sample_rate = torchaudio.load(audio_file)
                torchaudio.save(out_file, waveform, sample_rate)
                print(f"  [{i+1}/{min(pocet, len(audio_files))}] {out_file.name}")
            except Exception as e:
                print(f"  [chyba] {audio_file.name}: {e}")

        celkem += min(pocet, len(audio_files))

    print("\n" + "=" * 60)
    print(f"Hotovo: {celkem} WAV souborů")
    print(f"\nSpusť: python run.py {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Konverze audio souborů na WAV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Příklad:
  python convert_audio.py --input ./mls_flac --output ./data --pocet 20

Očekávaná struktura vstupu:
  mls_flac/
  ├── german/
  │   └── *.flac
  ├── french/
  │   └── *.flac
  └── ...
        """
    )
    parser.add_argument(
        "--input", "-i",
        default="./mls_flac",
        help="Vstupní složka s audio soubory (default: ./mls_flac)"
    )
    parser.add_argument(
        "--output", "-o",
        default="./data",
        help="Výstupní složka pro WAV soubory (default: ./data)"
    )
    parser.add_argument(
        "--pocet", "-n",
        type=int,
        default=20,
        help="Max počet souborů na jazyk (default: 20)"
    )

    args = parser.parse_args()
    convert_audio(args.input, args.output, args.pocet)


if __name__ == "__main__":
    main()
