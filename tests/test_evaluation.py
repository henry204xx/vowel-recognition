import pandas as pd
from vowel_recognition.evaluation import add_averages, build_speaker_table, build_vowel_table, evaluate
def test_accuracy_and_confusion():
    records = [{"true_vowel":"a","predicted_vowel":"a","speaker_id":"S01","error":None}, {"true_vowel":"i","predicted_vowel":"a","speaker_id":"S01","error":None}]
    metric = evaluate(records, ["a", "i"])
    assert metric["accuracy"] == 50 and metric["confusion_matrix"] == [[1, 0], [1, 0]]
def test_unavailable(): assert not evaluate([], ["a"])["available"]

def _metric(per_vowel, per_speaker):
    return {"available": True, "accuracy": 100.0, "per_vowel": per_vowel, "per_speaker": per_speaker}
def _metrics():
    return {
        "dtw": {
            "closed_recording": _metric({"a": 100.0, "i": 50.0}, {"S01": 75.0}),
            "closed_speakers": _metric({"a": 0.0, "i": 100.0}, {"S02": 50.0}),
            "open_speakers": _metric({"a": 50.0, "i": 100.0}, {"S03": 75.0}),
        },
        "gaussian": {
            "closed_recording": _metric({"a": 50.0, "i": 100.0}, {"S01": 75.0}),
            "closed_speakers": _metric({"a": 100.0, "i": 0.0}, {"S02": 50.0}),
            "open_speakers": _metric({"a": 100.0, "i": 50.0}, {"S03": 75.0}),
        },
    }
def test_vowel_table_compares_every_system_and_data_without_averages():
    table = build_vowel_table(_metrics(), ["a", "i"])
    assert list(table["Vowel"]) == ["a", "i"]
    assert list(table.columns) == [
        "Vowel", "DTW Closed Recording (%)", "DTW Closed Speakers (%)", "DTW Open Speakers (%)",
        "Gaussian Closed Recording (%)", "Gaussian Closed Speakers (%)", "Gaussian Open Speakers (%)",
    ]
    assert "Average (%)" not in table.columns
    assert table.loc[0, "DTW Closed Recording (%)"] == 100.0 and table.loc[1, "Gaussian Closed Speakers (%)"] == 0.0
def test_speaker_table_uses_requested_scenarios():
    table = build_speaker_table(_metrics(), scenarios=["closed_recording", "closed_speakers"])
    assert list(table["Speaker"]) == ["S01", "S02"]
    assert list(table.columns) == [
        "Speaker", "DTW Closed Recording (%)", "DTW Closed Speakers (%)",
        "Gaussian Closed Recording (%)", "Gaussian Closed Speakers (%)",
    ]
    assert table.loc[1, "DTW Closed Speakers (%)"] == 50.0
def test_speaker_table_keeps_columns_when_scenario_has_no_data():
    table = build_speaker_table(_metrics(), scenarios=["open_speakers"])
    table_empty = build_speaker_table({"dtw": {"open_speakers": {"available": False, "per_speaker": {}}},
                                       "gaussian": {"open_speakers": {"available": False, "per_speaker": {}}}},
                                      scenarios=["open_speakers"])
    assert list(table["Speaker"]) == ["S03"] and list(table_empty.columns) == list(table.columns)
def test_add_averages_only_changes_the_display_copy():
    table = pd.DataFrame({"Vowel": ["a", "i"], "DTW Closed Recording (%)": [100.0, 50.0]})
    averaged = add_averages(table, "Vowel")
    assert "Average (%)" not in table.columns
    assert list(averaged["Average (%)"]) == [100.0, 50.0, 75.0]
    assert averaged.loc[2, "Vowel"] == "Average" and averaged.loc[2, "DTW Closed Recording (%)"] == 75.0
def test_add_averages_skips_missing_scores():
    table = pd.DataFrame({"Vowel": ["a", "i"], "DTW Open Speakers (%)": [None, 80.0]})
    averaged = add_averages(table, "Vowel")
    assert pd.isna(averaged.loc[0, "Average (%)"]) and averaged.loc[1, "Average (%)"] == 80.0
