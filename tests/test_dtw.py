import numpy as np
from vowel_recognition.dtw import dtw
def test_dtw_identity_and_path():
    x = np.array([[0.], [1.]])
    cost, path = dtw(x, x, return_path=True)
    assert cost == 0 and path == [(0, 0), (1, 1)]
