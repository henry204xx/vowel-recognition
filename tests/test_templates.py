import numpy as np
from vowel_recognition.templates import generalized_template
def test_generalized_template_shape():
    result = generalized_template([np.ones((3, 39)), np.ones((4, 39)) * 3])
    assert result.shape == (3, 39) and np.all(result >= 1)
