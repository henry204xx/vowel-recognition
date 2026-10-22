"""Dataset manifest construction and validation."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import soundfile as sf
import pandas as pd


def expected_path(data_dir: Path, speaker: str, vowel: str, split: str) -> Path:
    return data_dir / speaker / vowel / f"{split}.wav"


def build_manifest(config: dict[str, Any], speakers: list[str] | None = None,
                   splits: tuple[str, ...] = ("train", "test")) -> pd.DataFrame:
    data_dir = Path(config["data"]["directory"])
    speakers = speakers or list(config["data"]["training_speakers"])
    rows = [{"file_path": str(expected_path(data_dir, speaker, vowel, split)),
             "speaker_id": speaker, "true_vowel": vowel, "split": split}
            for speaker in speakers for vowel in config["data"]["vowels"] for split in splits]
    return pd.DataFrame(rows)


def validate_dataset(config: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Return manifest and human-readable validation errors."""
    manifest = build_manifest(config)
    errors: list[str] = []
    for row in manifest.itertuples():
        file_path = Path(row.file_path)
        if not file_path.is_file():
            errors.append(f"Missing recording: {file_path}")
            continue
        try:
            info = sf.info(file_path)
            if info.frames <= 0 or info.samplerate <= 0:
                errors.append(f"Invalid audio metadata: {file_path}")
        except RuntimeError as exc:
            errors.append(f"Unreadable audio {file_path}: {exc}")
    return manifest, errors


def dataset_summary(manifest: pd.DataFrame, errors: list[str]) -> str:
    return (f"Expected recordings: {len(manifest)} | speakers: {manifest.speaker_id.nunique()} "
            f"| vowels: {manifest.true_vowel.nunique()} | validation issues: {len(errors)}")
