import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import config

def compute_similarity(
    resume_embedding: np.ndarray,
    job_embeddings: np.ndarray
) -> np.ndarray:
    """
    Input:  resume vector shape (384,), job matrix shape (N, 384)
    Output: cosine similarity scores shape (N,) — values between -1.0 and 1.0
    Method: sklearn.metrics.pairwise.cosine_similarity
            reshape resume_embedding to (1, 384) before calling
    """
    # Reshape resume_embedding to (1, 384) as required by sklearn
    resume_reshaped = resume_embedding.reshape(1, -1)
    
    # Compute cosine similarity
    similarity_matrix = cosine_similarity(resume_reshaped, job_embeddings)
    
    # Flatten result to (N,)
    return similarity_matrix.flatten()

def get_top_matches(
    scores: np.ndarray,
    metadata: list[dict],
    top_n: int = 5,
    min_threshold: float = None
) -> list[dict]:
    """
    Input:  scores array shape (N,), metadata list of N dicts, number of results
    Output: list of top_n dicts sorted by score descending, filtered by min_threshold
    Each output dict has keys: title, company, score (float, 2 decimal places), required_skills
    Note:   use numpy argsort for efficiency
    """
    if min_threshold is None:
        min_threshold = config.MIN_SIMILARITY_THRESHOLD

    top_indices = np.argsort(scores)[-top_n:][::-1]

    results = []
    for idx in top_indices:
        score = round(float(scores[idx]), 2)
        if score < min_threshold:
            continue
        meta = metadata[idx]
        results.append({
            "title": meta.get("title", "N/A"),
            "company": meta.get("company", "N/A"),
            "score": score,
            "required_skills": meta.get("required_skills", [])
        })

    return results
