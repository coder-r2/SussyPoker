"""Project configuration and well-known paths."""

from functools import cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@cache
def load_config() -> dict[str, Any]:
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


_paths = load_config()["paths"]
RAW_DIR = ROOT / _paths["raw"]
INTERIM_DIR = ROOT / _paths["interim"]
EXTERNAL_DIR = ROOT / _paths["external"]
ARTIFACTS_DIR = ROOT / _paths["artifacts"]
MODELS_DIR = ROOT / _paths["models"]


def has_raw_data() -> bool:
    return (RAW_DIR / "hands.parquet").exists()
