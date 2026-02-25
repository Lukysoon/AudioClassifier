import os
import io
import soundfile as sf
from datasets import load_dataset, Audio

LANGUAGES = ["en", "de", "fr", "es", "pl", "it", "ro", "hu", "cs", "nl", "hr", "sk", "fi", "sl"]
MAX_SECONDS = 36_000  # 10 hours
OUTPUT_DIR = "/workspace/Voxpopuli"


def download_language(lang):
    lang_dir = os.path.join(OUTPUT_DIR, lang)
    os.makedirs(lang_dir, exist_ok=True)

    existing = len([f for f in os.listdir(lang_dir) if f.endswith(".wav")])
    if existing > 0:
        print(f"[{lang}] Skipping - already has {existing} files")
        return

    print(f"[{lang}] Starting download...")
    ds = load_dataset("facebook/voxpopuli", lang, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    total_seconds = 0.0
    count = 0

    for example in ds:
        audio_bytes = example["audio"]["bytes"]
        data, sr = sf.read(io.BytesIO(audio_bytes))
        duration = len(data) / sr
        total_seconds += duration

        out_path = os.path.join(lang_dir, f"{count:06d}.wav")
        sf.write(out_path, data, sr)
        count += 1

        if count % 500 == 0:
            hours = total_seconds / 3600
            print(f"  [{lang}] {count} files, {hours:.2f}h")

        if total_seconds >= MAX_SECONDS:
            break

    hours = total_seconds / 3600
    print(f"[{lang}] Done: {count} files, {hours:.2f}h")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for lang in LANGUAGES:
        download_language(lang)
    print("All done!")
