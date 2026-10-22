"""Configuration loading and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML config, resolving project paths relative to its location."""
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    required = {"data", "results", "audio", "mfcc", "dtw", "gaussian"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration sections: {sorted(missing)}")
    if config["audio"]["sample_rate"] <= 0 or config["gaussian"]["regularization"] <= 0:
        raise ValueError("sample_rate and regularization must be positive")
    vowels = config["data"].get("vowels", [])
    if len(vowels) != 5 or len(set(vowels)) != 5:
        raise ValueError("Exactly five distinct vowels must be configured")
    train = set(config["data"].get("training_speakers", []))
    open_speakers = set(config["data"].get("open_test_speakers", []))
    if train & open_speakers:
        raise ValueError("Open-test speakers must be disjoint from training speakers")
    for section, key in (("data", "directory"), ("results", "dir")):
        config[section][key] = str((path.parent / config[section][key]).resolve())
    return config
