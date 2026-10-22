from vowel_recognition.evaluation import evaluate
def test_accuracy_and_confusion():
    records = [{"true_vowel":"a","predicted_vowel":"a","speaker_id":"S01","error":None}, {"true_vowel":"i","predicted_vowel":"a","speaker_id":"S01","error":None}]
    metric = evaluate(records, ["a", "i"])
    assert metric["accuracy"] == 50 and metric["confusion_matrix"] == [[1, 0], [1, 0]]
def test_unavailable(): assert not evaluate([], ["a"])["available"]
