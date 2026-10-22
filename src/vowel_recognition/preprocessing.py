"""Audio loading and deterministic preprocessing."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def to_mono(samples: np.ndarray) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if samples.ndim != 1:
        raise ValueError("Audio must be one- or two-dimensional")
    return samples


def preprocess_audio(file_path: str | Path, config: dict) -> tuple[np.ndarray, int]:
    samples, sample_rate = sf.read(file_path, always_2d=False)
    samples = to_mono(samples)
    if not len(samples) or not np.isfinite(samples).all():
        raise ValueError(f"Empty or non-finite audio: {file_path}")
    target = int(config["audio"]["sample_rate"])
    if sample_rate != target:
        divisor = np.gcd(sample_rate, target)
        samples = resample_poly(samples, target // divisor, sample_rate // divisor)
    if config["audio"].get("trim_silence", False):
        idx = np.flatnonzero(np.abs(samples) >= config["audio"].get("silence_threshold", 0.02))
        if len(idx):
            samples = samples[idx[0]:idx[-1] + 1]
    if not len(samples):
        raise ValueError(f"Audio contains only silence: {file_path}")
    if config["audio"].get("normalize", False):
        peak = np.max(np.abs(samples))
        if peak:
            samples = samples / peak
    return samples.astype(np.float64), target
