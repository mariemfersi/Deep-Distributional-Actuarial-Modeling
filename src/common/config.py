"""Chargement centralisé de la configuration du projet."""

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(config_path: str = "config/config.yaml") -> dict:
    full_path = PROJECT_ROOT / config_path
    with open(full_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_project_root() -> Path:
    return PROJECT_ROOT