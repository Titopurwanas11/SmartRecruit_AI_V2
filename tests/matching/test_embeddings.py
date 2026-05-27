import pytest
import numpy as np
import os
import pickle
from sentence_transformers import SentenceTransformer
from src.matching.embeddings import load_sbert_model, embed_text, load_job_embeddings
import config

def test_load_sbert_model():
    """Test that the SBERT model loads correctly and is of the right type."""
    model = load_sbert_model()
    assert isinstance(model, SentenceTransformer)


def test_load_sbert_model_uses_cpu_device(monkeypatch):
    calls = []

    class DummyModel:
        pass

    def fake_sentence_transformer(model_name, device=None):
        calls.append((model_name, device))
        return DummyModel()

    monkeypatch.setattr("src.matching.embeddings.SentenceTransformer", fake_sentence_transformer)

    model = load_sbert_model()

    assert isinstance(model, DummyModel)
    assert calls == [(config.SBERT_MODEL_NAME, "cpu")]


def test_embed_text():
    """Test that embed_text returns a numpy array of the correct shape and type."""
    model = load_sbert_model()
    text = "this is a test resume string"
    embedding = embed_text(text, model)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)
    assert embedding.dtype == np.float32

def test_embed_text_empty():
    """Test that embed_text handles empty strings gracefully."""
    model = load_sbert_model()
    text = ""
    embedding = embed_text(text, model)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)

def test_load_job_embeddings_missing_files(monkeypatch):
    """Test that load_job_embeddings raises FileNotFoundError with the exact message when files are missing."""
    # Mock paths to non-existent files
    monkeypatch.setattr(config, "JOB_EMBEDDINGS_PATH", "non_existent_embeddings.npy")
    monkeypatch.setattr(config, "JOB_METADATA_PATH", "non_existent_metadata.pkl")
    
    expected_msg = "Job embeddings not found. Run scripts/generate_embeddings.py on Colab first."
    
    with pytest.raises(FileNotFoundError) as excinfo:
        load_job_embeddings()
    
    assert str(excinfo.value) == expected_msg

def test_load_job_embeddings_success(tmp_path, monkeypatch):
    """Test that load_job_embeddings correctly loads dummy data."""
    # Create dummy data
    dummy_embeddings = np.random.rand(5, 384).astype(np.float32)
    dummy_metadata = [
        {"title": "Job 1", "company": "Co 1", "required_skills": ["Python"]},
        {"title": "Job 2", "company": "Co 2", "required_skills": ["SQL"]}
    ]
    
    # Save to temp files
    embeddings_file = tmp_path / "job_embeddings.npy"
    metadata_file = tmp_path / "job_metadata.pkl"
    
    np.save(embeddings_file, dummy_embeddings)
    with open(metadata_file, "wb") as f:
        pickle.dump(dummy_metadata, f)
        
    # Mock config paths
    monkeypatch.setattr(config, "JOB_EMBEDDINGS_PATH", str(embeddings_file))
    monkeypatch.setattr(config, "JOB_METADATA_PATH", str(metadata_file))
    
    # Load and verify
    loaded_embeddings, loaded_metadata = load_job_embeddings()
    
    assert np.array_equal(loaded_embeddings, dummy_embeddings)
    assert loaded_metadata == dummy_metadata
