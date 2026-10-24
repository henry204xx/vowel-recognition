"""Metrics for successful recognition predictions."""
from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

SYSTEM_LABELS = {"dtw": "DTW", "gaussian": "Gaussian"}
SCENARIO_LABELS = {
    "closed_recording": "Closed Recording",
    "closed_speakers": "Closed Speakers",
    "open_speakers": "Open Speakers",
}

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

def _column_label(system: str, scenario: str) -> str:
    return f"{SYSTEM_LABELS.get(system, system.title())} {SCENARIO_LABELS.get(scenario, scenario.replace('_', ' ').title())} (%)"

def _score_table(metrics: dict, index: list[str], systems: tuple[str, ...], scenarios: list[str],
                 group: str, index_name: str) -> pd.DataFrame:
    frame = pd.DataFrame({index_name: list(index)})
    for system in systems:
        for scenario in scenarios:
            metric = metrics.get(system, {}).get(scenario, {})
            scores = metric.get(group, {}) if metric.get("available") else {}
            frame[_column_label(system, scenario)] = [scores.get(key) for key in index]
    return frame

def build_vowel_table(metrics: dict, vowels: list[str], systems: tuple[str, ...] = ("dtw", "gaussian"),
                      scenarios: list[str] | None = None) -> pd.DataFrame:
    """Accuracy per vowel for each system/data combination, without averages."""
    scenarios = list(SCENARIO_LABELS) if scenarios is None else list(scenarios)
    return _score_table(metrics, list(vowels), systems, scenarios, "per_vowel", "Vowel")

def build_speaker_table(metrics: dict, systems: tuple[str, ...] = ("dtw", "gaussian"),
                        scenarios: list[str] | None = None) -> pd.DataFrame:
    """Accuracy per speaker for each system/data combination, without averages."""
    scenarios = list(SCENARIO_LABELS) if scenarios is None else list(scenarios)
    speakers = sorted({speaker for system in systems for scenario in scenarios
                       for speaker in metrics.get(system, {}).get(scenario, {}).get("per_speaker", {})})
    return _score_table(metrics, speakers, systems, scenarios, "per_speaker", "Speaker")

def add_averages(table: pd.DataFrame, index_name: str) -> pd.DataFrame:
    """Display copy with a row average per item, a column average per data, and an overall average."""
    display_table = table.copy()
    score_columns = [column for column in display_table.columns if column != index_name]
    display_table["Average (%)"] = display_table[score_columns].mean(axis=1, skipna=True)
    if len(display_table):
        row = {index_name: "Average", **{column: display_table[column].mean(skipna=True) for column in score_columns}}
        row["Average (%)"] = display_table["Average (%)"].mean(skipna=True)
        display_table = pd.concat([display_table, pd.DataFrame([row])], ignore_index=True)
    return display_table
