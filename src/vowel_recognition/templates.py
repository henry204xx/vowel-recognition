"""Generalized DTW template construction."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from .dtw import dtw


def generalized_template(sequences: list[np.ndarray], normalize: str = "length") -> np.ndarray:
    """Align every sequence to the first reference then average frames per reference index."""
    if not sequences: raise ValueError("At least one training sequence is required")
    reference = np.asarray(sequences[0], float)
    aligned = [[reference[index]] for index in range(len(reference))]
    for sequence in sequences[1:]:
        _, path = dtw(reference, sequence, normalize, return_path=True)
        buckets: list[list[np.ndarray]] = [[] for _ in reference]
        for ref_idx, seq_idx in path: buckets[ref_idx].append(np.asarray(sequence)[seq_idx])
        for index, values in enumerate(buckets):
            if values: aligned[index].append(np.mean(values, axis=0))
    return np.vstack([np.mean(values, axis=0) for values in aligned])


def build_templates(features_by_vowel: dict[str, list[np.ndarray]], normalize: str = "length") -> dict[str, np.ndarray]:
    return {vowel: generalized_template(sequences, normalize) for vowel, sequences in features_by_vowel.items()}


def save_templates(templates: dict[str, np.ndarray], directory: str | Path) -> None:
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    for vowel, template in templates.items(): np.save(directory / f"template_{vowel}.npy", template)
    (directory / "templates.json").write_text(json.dumps({k: list(v.shape) for k, v in templates.items()}, indent=2))
