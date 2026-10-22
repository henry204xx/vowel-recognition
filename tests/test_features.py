import numpy as np
from vowel_recognition.features import extract_features
def test_mfcc_dimensions_and_finite():
    cfg = {"mfcc": {"numcep": 13, "winlen": .025, "winstep": .01}}
    result = extract_features(np.random.default_rng(1).normal(size=1600), 16000, cfg)
    assert result.shape[1] == 39 and np.isfinite(result).all()
