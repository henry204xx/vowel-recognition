"""MFCC feature extraction."""
from __future__ import annotations
import numpy as np
from python_speech_features import mfcc, delta


def extract_features(audio: np.ndarray, sample_rate: int, config: dict) -> np.ndarray:
    """Return 13 MFCC + delta + delta-delta features (39 dimensions)."""
    options = config["mfcc"]
    base = mfcc(audio, samplerate=sample_rate, numcep=options.get("numcep", 13),
                winlen=options.get("winlen", .025), winstep=options.get("winstep", .01))
    if len(base) < 3:
        base = np.pad(base, ((0, max(0, 3 - len(base))), (0, 0)), mode="edge")
    output = np.hstack((base, delta(base, 2), delta(delta(base, 2), 2)))
    if output.ndim != 2 or output.shape[1] != 39 or not np.isfinite(output).all():
        raise ValueError("MFCC extraction did not produce finite (frames, 39) output")
    return output
