import numpy as np
from vowel_recognition.gaussian_model import estimate_gaussian, negative_log_likelihood
from vowel_recognition.recognition import predict_gaussian
def test_gaussian_dimensions_regularization_and_prediction():
    a = np.zeros((10, 39)); b = np.ones((10, 39)) * 5
    ma, mb = estimate_gaussian(a, .1), estimate_gaussian(b, .1)
    assert ma.mean.shape == (39,) and np.all(np.diag(ma.covariance) >= .1)
    assert negative_log_likelihood(a, ma).shape == (10,)
    assert predict_gaussian(a, {"a": ma, "i": mb})[0] == "a"
