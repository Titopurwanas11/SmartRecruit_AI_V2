import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.pipeline.hr_pipeline import run_hr_pipeline


def _make_mock_model(seed=None):
    model = MagicMock()
    rng = np.random.default_rng(seed)
    model.encode.side_effect = lambda texts: rng.random((len(texts), 384)).astype("float32")
    return model


def test_returns_list_sorted_by_score():
    # Use controlled embeddings: JD matches Alice best, Bob worst.
    # We do this by making the mock return fixed vectors so cosine scores are deterministic.
    call_count = [0]
    vectors = {
        0: np.ones(384, dtype="float32"),          # JD embedding
        1: np.ones(384, dtype="float32") * 0.9,    # Alice — very similar to JD
        2: np.zeros(384, dtype="float32"),          # Bob — orthogonal to JD
        3: np.ones(384, dtype="float32") * 0.5,    # Charlie — moderate
    }

    def controlled_encode(texts):
        idx = call_count[0]
        call_count[0] += 1
        return np.array([vectors[idx]])

    model = MagicMock()
    model.encode.side_effect = controlled_encode

    resumes = [
        ("Alice", "Python developer with machine learning and SQL experience"),
        ("Bob", "Frontend developer HTML CSS JavaScript React"),
        ("Charlie", "Data scientist pandas numpy scikit-learn deep learning"),
    ]
    job_desc = "We are hiring a senior data scientist with Python and machine learning skills."

    with patch("src.pipeline.hr_pipeline.load_sbert_model", return_value=model):
        results = run_hr_pipeline(job_desc, resumes)

    assert len(results) == 3
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted descending by score"
    # Alice should be #1, Bob #3 (zero vector → score=0 or near-zero)
    assert results[0]["candidate"] == "Alice"
    assert results[-1]["candidate"] == "Bob"


def test_result_dict_keys():
    resumes = [("Alice", "Python developer")]
    job_desc = "Python engineer needed"

    with patch("src.pipeline.hr_pipeline.load_sbert_model", return_value=_make_mock_model()):
        results = run_hr_pipeline(job_desc, resumes)

    assert set(results[0].keys()) == {"rank", "candidate", "score"}


def test_score_is_float_two_decimals():
    resumes = [("Alice", "Python developer")]
    job_desc = "Python engineer needed"

    with patch("src.pipeline.hr_pipeline.load_sbert_model", return_value=_make_mock_model()):
        results = run_hr_pipeline(job_desc, resumes)

    score = results[0]["score"]
    assert isinstance(score, float)
    assert score == round(score, 2)


def test_empty_resumes_returns_empty_list():
    with patch("src.pipeline.hr_pipeline.load_sbert_model") as mock_load:
        results = run_hr_pipeline("any job description", [])
    assert results == []
    mock_load.assert_not_called()


def test_rank_starts_at_one():
    resumes = [("Alice", "text"), ("Bob", "text")]
    with patch("src.pipeline.hr_pipeline.load_sbert_model", return_value=_make_mock_model()):
        results = run_hr_pipeline("job", resumes)
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
