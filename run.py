#!/usr/bin/env python3
"""
Audio Classifier - Simple entry point.

Usage:
    python run.py ./data
    python run.py ./data --output ./results
    python run.py ./data --pooling max --n-neighbors 10
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from audio_classifier.cli import main

if __name__ == "__main__":
    sys.exit(main())
