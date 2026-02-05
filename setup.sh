#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="venv"

# --- ffmpeg ---
if command -v ffmpeg &>/dev/null; then
    echo "✅ ffmpeg už je nainstalovaný ($(ffmpeg -version 2>&1 | head -1))"
else
    echo "📦 Instaluji ffmpeg..."
    if [[ "$(uname)" == "Darwin" ]]; then
        if command -v brew &>/dev/null; then
            brew install ffmpeg
        else
            echo "❌ Homebrew není nainstalovaný. Nainstaluj ho z https://brew.sh a spusť znovu."
            exit 1
        fi
    elif [[ "$(uname)" == "Linux" ]]; then
        if command -v apt-get &>/dev/null; then
            sudo apt-get update && sudo apt-get install -y ffmpeg
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y ffmpeg
        else
            echo "❌ Nepodporovaný package manager. Nainstaluj ffmpeg ručně."
            exit 1
        fi
    else
        echo "❌ Nepodporovaný OS. Nainstaluj ffmpeg ručně."
        exit 1
    fi
    echo "✅ ffmpeg nainstalovaný"
fi

# --- venv ---
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtuální prostředí už existuje ($VENV_DIR/)"
else
    echo "📦 Vytvářím virtuální prostředí..."
    python3 -m venv "$VENV_DIR"
    echo "✅ Virtuální prostředí vytvořeno ($VENV_DIR/)"
fi

# --- requirements ---
echo "📦 Instaluji Python závislosti..."
source "$VENV_DIR/bin/activate"
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Závislosti nainstalované"

echo ""
echo "=========================================="
echo "  Setup hotový!"
echo "  Aktivuj venv příkazem:"
echo "    source $VENV_DIR/bin/activate"
echo "=========================================="
