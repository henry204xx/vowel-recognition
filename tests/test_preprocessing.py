import numpy as np
import soundfile as sf
from vowel_recognition.preprocessing import to_mono, preprocess_audio
def test_stereo_to_mono(): assert np.allclose(to_mono(np.array([[1., 3.], [2., 4.]])), [2., 3.])
def test_resampling(tmp_path):
    path = tmp_path / "sample.wav"; sf.write(path, np.ones(800), 8000)
    output, rate = preprocess_audio(path, {"audio":{"sample_rate":16000,"normalize":False,"trim_silence":False}})
    assert rate == 16000 and len(output) == 1600
