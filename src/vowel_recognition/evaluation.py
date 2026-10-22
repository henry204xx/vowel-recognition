"""Metrics for successful recognition predictions."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

def evaluate(predictions: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
    frame = pd.DataFrame(predictions)
    failed = int(frame.get("error", pd.Series(dtype=object)).notna().sum()) if len(frame) else 0
    valid = frame[frame.get("error", pd.Series(index=frame.index, dtype=object)).isna()].copy() if len(frame) else frame
    if valid.empty:
        return {"available": False, "accuracy": None, "evaluated": 0, "failed": failed, "skipped": 0,
                "correct": 0, "incorrect": 0, "per_vowel": {}, "per_speaker": {}, "confusion_matrix": []}
    valid["correct"] = valid.true_vowel == valid.predicted_vowel
    matrix = pd.crosstab(pd.Categorical(valid.true_vowel, categories=labels),
                         pd.Categorical(valid.predicted_vowel, categories=labels), dropna=False)
    per_vowel = (valid.groupby("true_vowel").correct.mean() * 100).to_dict()
    per_speaker = (valid.groupby("speaker_id").correct.mean() * 100).to_dict()
    return {"available": True, "accuracy": float(valid.correct.mean() * 100), "evaluated": len(valid), "failed": failed,
            "skipped": 0, "correct": int(valid.correct.sum()), "incorrect": int((~valid.correct).sum()),
            "per_vowel": per_vowel, "per_speaker": per_speaker, "confusion_matrix": matrix.values.tolist(), "labels": labels}
