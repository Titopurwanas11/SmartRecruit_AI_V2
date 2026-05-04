import logging
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)

def load_sbert_model() -> SentenceTransformer:
    """
    Input:  None — uses SBERT_MODEL_NAME from config.py ("all-MiniLM-L6-v2")
    Output: loaded SentenceTransformer model
    Note:   model downloads automatically on first run (~80MB), cached after that
    """
    model_name = config.SBERT_MODEL_NAME
    logger.info(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)
    return model

def embed_text(text: str, model: SentenceTransformer) -> np.ndarray:
    """
    Input:  single cleaned string, loaded SentenceTransformer
    Output: embedding vector shape (384,) — dtype float32
    Note:   use model.encode([text])[0] to get shape (384,) not (1, 384)
    """
    # model.encode returns a numpy array. 
    # Passing a list [text] returns a list of embeddings.
    # We take the first one [0].
    embedding = model.encode([text])[0]
    return embedding.astype(np.float32)

def load_job_embeddings() -> tuple[np.ndarray, list[dict]]:
    """
    Input:  None — reads JOB_EMBEDDINGS_PATH and JOB_METADATA_PATH from config.py
    Output: (numpy array shape (N, 384), list of N dicts)
    Error:  raise FileNotFoundError if either file is missing, with message:
            "Job embeddings not found. Run scripts/generate_embeddings.py on Colab first."
    Metadata dict shape: {"title": str, "company": str, "required_skills": list[str]}
    """
    import pickle
    
    if not os.path.exists(config.JOB_EMBEDDINGS_PATH) or not os.path.exists(config.JOB_METADATA_PATH):
        raise FileNotFoundError("Job embeddings not found. Run scripts/generate_embeddings.py on Colab first.")
        
    logger.info(f"Loading job embeddings from {config.JOB_EMBEDDINGS_PATH}")
    job_embeddings = np.load(config.JOB_EMBEDDINGS_PATH)
    
    logger.info(f"Loading job metadata from {config.JOB_METADATA_PATH}")
    with open(config.JOB_METADATA_PATH, 'rb') as f:
        job_metadata = pickle.load(f)
        
    return job_embeddings, job_metadata
