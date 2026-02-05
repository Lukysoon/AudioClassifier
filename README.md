# Audio Classifier

Vizualizace audio souborů v 3D prostoru pomocí HuBERT embeddingů a UMAP.

![Pipeline](https://img.shields.io/badge/Pipeline-Audio→HuBERT→UMAP→3D-blue)

## Co to dělá

1. Načte audio soubory z adresářů (každá složka = jedna kategorie)
2. Extrahuje features pomocí HuBERT modelu
3. Redukuje 768 dimenzí na 3D pomocí UMAP
4. Zobrazí interaktivní 3D vizualizaci v prohlížeči

## Rychlý start

```bash
git clone https://github.com/Lukysoon/AudioClassifier.git
cd AudioClassifier
./setup.sh
source venv/bin/activate
python download_mls.py
python run.py ./data
```

`setup.sh` automaticky nainstaluje ffmpeg, vytvoří virtuální prostředí a nainstaluje Python závislosti.

## Struktura dat

```
data/
├── kategorie_A/       ← název složky = barva v grafu
│   ├── audio1.wav
│   └── audio2.mp3
├── kategorie_B/
│   └── audio3.wav
└── kategorie_C/
    └── audio4.flac
```

## Stažení dat

### Automatické stažení (doporučeno)

Script `download_mls.py` stáhne MLS 10h limited supervision sety z Hugging Face pro 7 jazyků (german, dutch, french, spanish, italian, portuguese, polish) a uloží je jako MP3. Cache se po zpracování každého jazyka automaticky maže.

```bash
python download_mls.py
```

Výstupní struktura:
```
data/
├── german/
│   ├── 00000.mp3
│   ├── 00001.mp3
│   ├── ...
│   └── transcripts.csv
├── english/
│   └── ...
└── ...
```

### Ruční stažení

Dataset MLS (Multilingual LibriSpeech) je také dostupný na:
- **OpenSLR**: https://www.openslr.org/94/

Stáhněte si audio soubory pro požadované jazyky (např. `mls_german.tar.gz`, `mls_french.tar.gz`) a rozbalte do složky `mls_flac/`.

⚠️ Dataset je velký (desítky GB na jazyk). Pro testování stačí stáhnout menší subset.

## Použití

### Příprava dat

**Konverze FLAC/MP3 → WAV:**

```bash
python convert_audio.py --input ./mls_flac --output ./data --pocet 20
```

Očekávaná struktura vstupu:
```
mls_flac/
├── german/
│   └── *.flac
├── french/
│   └── *.flac
└── ...
```

### Spuštění

```bash
python run.py ./data
```

### Parametry

**Vzorový příkaz:**
```bash
source venv/bin/activate && python run.py ./data --chunk 2.0 --chunk-handling discard --pooling max --n-neighbors 30
```

#### Audio parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--max-duration` | `240` | Maximální délka audia v sekundách |
| `--chunk` | vypnuto | Rozdělí audio na chunky zadané délky (s) |
| `--chunk-handling` | `discard` | Zpracování posledního krátkého chunku: `pad` / `discard` / `keep` |
| `--min-chunk` | `0.5` | Minimální délka chunku pro `keep` režim (s) |
| `--no-silence-removal` | `False` | Vypnout odstranění ticha |
| `--silence-threshold` | `40` | Práh ticha v dB (vyšší = méně citlivé) |

#### Model parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--pooling` | `mean` | Pooling strategie: `mean` (průměr) / `max` (maximum) |

#### UMAP parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--n-neighbors` | `15` | Počet sousedů - nižší = těsnější clustery |
| `--min-dist` | `0.1` | Minimální vzdálenost - nižší = hustší clustery |

#### Výstupní parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--output` / `-o` | `./output` | Výstupní složka pro HTML soubory |
| `--no-open` | `False` | Neotevírat vizualizace v prohlížeči |
| `--save-embeddings` | vypnuto | Uložit embeddingy do .npz souboru |
| `--prefix` | `audio_classifier` | Prefix pro názvy výstupních souborů |
| `--config` / `-c` | - | Načíst konfiguraci z YAML souboru |

#### Příklady

```bash
# Základní použití (ticho se automaticky odstraňuje)
python run.py ./data

# Dlouhé nahrávky s 5s chunky
python run.py ./data --chunk 5.0 --chunk-handling keep --min-chunk 1.0

# Těsné clustery pro krátké zvuky
python run.py ./data --n-neighbors 5 --min-dist 0.01 --pooling max

# Vypnout odstranění ticha
python run.py ./data --no-silence-removal

# Citlivější detekce ticha (odstraní i tišší zvuky)
python run.py ./data --silence-threshold 30

# Produkční běh bez prohlížeče
python run.py ./data --no-open --save-embeddings embeddings.npz -o ./results
```

Všechny parametry:
```bash
python run.py --help
```

## Výstup

- `output/audio_classifier_3d.html` - Interaktivní 3D scatter plot
- `output/audio_classifier_distances.html` - Heatmapa vzdáleností mezi kategoriemi

## Požadavky

- Python 3.10+
- ~360 MB pro HuBERT model (stáhne se automaticky při prvním spuštění)

## Podporované formáty

`.wav`, `.mp3`, `.flac`, `.ogg`
