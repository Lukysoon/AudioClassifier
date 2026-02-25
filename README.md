# Audio Classifier

Vizualizace audio souborů v 3D prostoru pomocí ContentVec embeddingů a UMAP.

![Pipeline](https://img.shields.io/badge/Pipeline-Audio→ContentVec→UMAP→3D-blue)

## Co to dělá

1. Načte audio soubory z adresářů (každá složka = jedna kategorie)
2. Extrahuje features pomocí ContentVec modelu
3. Redukuje 768 dimenzí na 3D pomocí UMAP
4. Zobrazí interaktivní 3D vizualizaci v prohlížeči

## Rychlý start

```bash
git clone https://github.com/Lukysoon/AudioClassifier.git
cd AudioClassifier
./setup.sh
source venv/bin/activate
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

## Spuštění

```bash
python run.py ./data --chunk 10.0 --clear-cache
```

### Parametry

**Vzorový příkaz:**
```bash
source venv/bin/activate && python run.py ./data --chunk 2.0 --chunk-handling discard --pooling max --n-neighbors 30
```

#### Audio parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--max-duration` | `30` | Maximální délka audia v sekundách |
| `--chunk SECONDS` | `10` | Přepíše výchozí délku chunků (chunking je zapnutý defaultně) |
| `--chunk-handling` | `discard` | Zpracování posledního krátkého chunku: `pad` / `discard` / `keep` |
| `--min-chunk` | `0.5` | Minimální délka chunku pro `keep` režim (s) |
| `--no-silence-removal` | - | Vypnout odstranění ticha (zapnuto defaultně) |
| `--silence-threshold` | `40` | Práh ticha v dB (vyšší = méně citlivé) |
| `--no-noise-reduction` | - | Vypnout spektrální šumovou redukci (zapnuta defaultně) |
| `--noise-non-stationary` | - | Adaptivní (non-stationary) šumová redukce místo stationary |

#### Model parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--model` | `contentvec` | Model: `contentvec` / `hubert-base` / `hubert-large` |
| `--pooling` | `mean` | Pooling strategie: `mean` (průměr) / `max` (maximum) |
| `--layer` | `-1` | Transformer vrstva (-1 = poslední, 6 = střední) |

#### UMAP parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--n-neighbors` | `15` | Počet sousedů - nižší = těsnější clustery |
| `--min-dist` | `0.1` | Minimální vzdálenost - nižší = hustší clustery |
| `--repulsion` | `1.0` | Síla odpuzování mezi clustery - vyšší = větší separace |

#### Cache parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--no-cache` | - | Vypnout embedding cache (přepočítat vše od začátku) |
| `--clear-cache` | - | Smazat existující cache před spuštěním |
| `--load-cache FILE` | - | Načíst embeddingy z .pkl souboru (přeskočí model) |

#### Výstupní parametry

| Parametr | Výchozí | Popis |
|----------|---------|-------|
| `--output` / `-o` | `./output` | Výstupní složka pro HTML soubory |
| `--no-open` | - | Neotevírat vizualizace v prohlížeči |
| `--save-embeddings FILE` | - | Uložit embeddingy do .npz souboru |
| `--prefix` | `audio_classifier` | Prefix pro názvy výstupních souborů |
| `--config` / `-c` | - | Načíst konfiguraci z YAML souboru |
| `--preprocessing-only DIR` | - | Jen preprocessing (bez embeddingů), výstup uložit do DIR |
| `--umap-only` | - | Přeskočit audio processing, načíst z cache a přegenerovat UMAP + vizualizaci |

#### Příklady

```bash
# Základní použití (chunking 10s, noise reduction, silence removal — vše defaultně ON)
python run.py ./data

# Změnit délku chunků na 5s
python run.py ./data --chunk 5.0 --chunk-handling keep --min-chunk 1.0

# Těsné clustery pro krátké zvuky
python run.py ./data --n-neighbors 5 --min-dist 0.01 --pooling max

# Použít HuBERT model se střední vrstvou
python run.py ./data --model hubert-large --layer 6

# Vypnout noise reduction / silence removal
python run.py ./data --no-noise-reduction
python run.py ./data --no-silence-removal

# Adaptivní šumová redukce (pro proměnlivý šum)
python run.py ./data --noise-non-stationary

# Citlivější detekce ticha (odstraní i tišší zvuky)
python run.py ./data --silence-threshold 30

# Smazat cache a přepočítat
python run.py ./data --clear-cache

# Přegenerovat vizualizaci s jinými UMAP parametry (bez přepočtu embeddingů)
python run.py ./data --umap-only --n-neighbors 30 --min-dist 0.3

# Jen preprocessing (uložit chunky bez embeddingů)
python run.py ./data --chunk 5.0 --preprocessing-only ./preprocessed

# Produkční běh bez prohlížeče
python run.py ./data --no-open --save-embeddings embeddings.npz -o ./results
```

Všechny parametry:
```bash
python run.py --help
```

## Jupyter Notebook

Alternativní interaktivní workflow v `notebook.ipynb`:

```bash
jupyter notebook notebook.ipynb
```

Notebook umožňuje krokové spouštění pipeline a rychlé experimentování s parametry. Podporuje režim **pouze UMAP + vizualizace z cache** — přeskočí extrakci embeddingů a použije uložený cache, takže stačí měnit UMAP parametry a okamžitě vidět výsledek.

## Výstup

- `output/audio_classifier_3d.html` - Interaktivní 3D scatter plot
- `output/audio_classifier_distances.html` - Heatmapa vzdáleností mezi kategoriemi

## Požadavky

- Python 3.10+
- ~360 MB pro ContentVec model (stáhne se automaticky při prvním spuštění)

## Podporované formáty

`.wav`, `.mp3`, `.flac`, `.ogg`

## Export z parquet

```bash
python parse_audio_parquet.py --input data.parquet --info

python parse_audio_parquet.py --input data.parquet --output_dir ./audio_files
```