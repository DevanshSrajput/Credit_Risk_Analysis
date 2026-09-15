"""
Generic reusable helper functions.
"""

import os
from pathlib import Path


def create_project_directories():
    """Create all necessary project directories."""
    from src.config import (
        FIGURES_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
    )
    for directory in [MODELS_DIR, REPORTS_DIR, FIGURES_DIR, PROCESSED_DATA_DIR]:
        os.makedirs(directory, exist_ok=True)


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def check_file_exists(file_path):
    """Check if a file exists and return boolean."""
    return Path(file_path).exists()
