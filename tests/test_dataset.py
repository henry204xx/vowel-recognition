from pathlib import Path
import pytest
from vowel_recognition.config import load_config
from vowel_recognition.data_loader import validate_dataset
def test_missing_dataset_files(tmp_path):
    cfg = {"data":{"directory":str(tmp_path),"vowels":["a","i","u","e","o"],"training_speakers":["S01"],"open_test_speakers":[]},"results":{"dir":str(tmp_path)},"audio":{"sample_rate":16000},"mfcc":{},"dtw":{},"gaussian":{"regularization":.1}}
    _, errors = validate_dataset(cfg); assert len(errors) == 10
def test_open_speaker_leakage(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("data: {directory: data, vowels: [a, i, u, e, o], training_speakers: [S01], open_test_speakers: [S01]}\nresults: {dir: r}\naudio: {sample_rate: 16000}\nmfcc: {}\ndtw: {}\ngaussian: {regularization: 0.1}")
    with pytest.raises(ValueError): load_config(path)
