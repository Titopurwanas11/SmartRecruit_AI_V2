import numpy as np
import pytest
from src.matching.similarity import compute_similarity, get_top_matches

def test_compute_similarity():
    # Arrange
    resume_embedding = np.array([1, 0, 1], dtype=np.float32)
    job_embeddings = np.array([
        [1, 0, 1],   # Exact match (1.0)
        [0, 1, 0],   # Orthogonal (0.0)
        [-1, 0, -1]  # Opposite (-1.0)
    ], dtype=np.float32)
    
    # Act
    scores = compute_similarity(resume_embedding, job_embeddings)
    
    # Assert
    assert isinstance(scores, np.ndarray)
    assert scores.shape == (3,)
    assert pytest.approx(scores[0]) == 1.0
    assert pytest.approx(scores[1]) == 0.0
    assert pytest.approx(scores[2]) == -1.0

def test_get_top_matches():
    # Arrange
    scores = np.array([0.5, 0.9, 0.1, 0.8], dtype=np.float32)
    metadata = [
        {"title": "Job 0", "company": "Co A", "required_skills": ["S1"]},
        {"title": "Job 1", "company": "Co B", "required_skills": ["S2"]},
        {"title": "Job 2", "company": "Co C", "required_skills": ["S3"]},
        {"title": "Job 3", "company": "Co D", "required_skills": ["S4"]}
    ]
    
    # Act
    top_matches = get_top_matches(scores, metadata, top_n=2)
    
    # Assert
    assert len(top_matches) == 2
    
    # First match should be Job 1 (score 0.9)
    assert top_matches[0]["title"] == "Job 1"
    assert top_matches[0]["score"] == 0.9
    
    # Second match should be Job 3 (score 0.8)
    assert top_matches[1]["title"] == "Job 3"
    assert top_matches[1]["score"] == 0.8
    
    # Verify keys
    for match in top_matches:
        assert set(match.keys()) == {"title", "company", "score", "required_skills"}

def test_get_top_matches_rounding():
    # Arrange
    scores = np.array([0.88888], dtype=np.float32)
    metadata = [{"title": "Job", "company": "Co", "required_skills": ["S"]}]
    
    # Act
    top_matches = get_top_matches(scores, metadata, top_n=1)
    
    # Assert
    assert top_matches[0]["score"] == 0.89
