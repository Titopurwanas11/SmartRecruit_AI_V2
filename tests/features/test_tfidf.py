import os
import pytest
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from src.features.tfidf import load_vectorizer, vectorize

def test_load_vectorizer_not_found():
    """Test that a specific FileNotFoundError is raised when path is missing."""
    missing_path = "non_existent_vectorizer.joblib"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_vectorizer(missing_path)
    
    assert "[SmartRecruitAI] File not found" in str(exc_info.value)
    assert missing_path in str(exc_info.value)
    assert "Check config.py paths" in str(exc_info.value)

def test_load_vectorizer_success(tmp_path):
    """Test successful loading of a vectorizer."""
    # Create a dummy vectorizer and save it
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["test document one", "test document two"])
    
    model_path = os.path.join(tmp_path, "test_vec.joblib")
    joblib.dump(vectorizer, model_path)
    
    # Load it back
    loaded = load_vectorizer(model_path)
    assert isinstance(loaded, TfidfVectorizer)
    assert hasattr(loaded, "vocabulary_")

def test_vectorize():
    """Test that text is correctly vectorized."""
    # Setup a dummy vectorizer
    vectorizer = TfidfVectorizer()
    vectorizer.fit(["python developer", "data scientist", "machine learning engineer"])
    
    text = "python engineer"
    result = vectorize(text, vectorizer)
    
    # Should return a sparse matrix of shape (1, n_features)
    assert result.shape[0] == 1
    assert result.shape[1] == len(vectorizer.vocabulary_)
