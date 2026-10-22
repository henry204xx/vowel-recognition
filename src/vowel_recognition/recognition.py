"""Recognition functions, independent of experiment orchestration."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from .dtw import dtw
from .gaussian_model import GaussianModel, negative_log_likelihood
from .preprocessing import preprocess_audio
from .features import extract_features

def predict_dtw(features: np.ndarray, templates: dict[str, np.ndarray], normalize: str = "length") -> tuple[str, dict[str, float]]:
    scores = {label: dtw(features, template, normalize) for label, template in templates.items()}
    return min(scores, key=scores.get), scores

def predict_gaussian(features: np.ndarray, models: dict[str, GaussianModel]) -> tuple[str, dict[str, float]]:
    # Frames are conditionally independent; temporal dependencies are intentionally not modeled.
    scores = {label: float(negative_log_likelihood(features, model).mean()) for label, model in models.items()}
    return min(scores, key=scores.get), scores

def predict_audio(file_path: str | Path, system: str, artifacts: dict[str, Any], config: dict, **metadata: Any) -> dict[str, Any]:
    result = {"file_path": str(file_path), "system": system, **metadata}
    try:
        audio, rate = preprocess_audio(file_path, config); features = extract_features(audio, rate, config)
        if system == "dtw": prediction, scores = predict_dtw(features, artifacts, config["dtw"]["normalize"])
        elif system == "gaussian": prediction, scores = predict_gaussian(features, artifacts)
        else: raise ValueError(f"Unknown system: {system}")
        result.update(predicted_vowel=prediction, scores=scores, error=None)
    except Exception as exc:
        result.update(predicted_vowel=None, scores={}, error=str(exc))
    return result

def predict_batch(manifest: pd.DataFrame, system: str, artifacts: dict[str, Any], config: dict) -> list[dict[str, Any]]:
    return [predict_audio(row.file_path, system, artifacts, config, true_vowel=row.true_vowel, speaker_id=row.speaker_id)
            for row in manifest.itertuples()]
