import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocessing.clean_text import clean_text
from src.matching.embeddings import load_sbert_model

logger = logging.getLogger(__name__)


def run_hr_pipeline(
    raw_job_description: str,
    resumes: list[tuple[str, str, str]]
) -> list[dict]:
    """
    Input:
        raw_job_description: uncleaned job description string from HR input
        resumes: list of (candidate_filename: str, owner_name: str, raw_resume_text: str)
    Output:
        list of dicts sorted by score descending:
        [{"rank": int, "candidate": str, "owner_name": str, "score": float}, ...]
        Empty list if resumes is empty.
    """
    if not resumes:
        return []

    model = load_sbert_model()

    cleaned_jd = clean_text(raw_job_description)
    jd_embedding = model.encode([cleaned_jd])[0].reshape(1, -1)  # (1, 384)

    results = []
    for filename, owner_name, raw_text in resumes:
        cleaned = clean_text(raw_text)
        resume_embedding = model.encode([cleaned])[0].reshape(1, -1)  # (1, 384)
        score = float(cosine_similarity(jd_embedding, resume_embedding)[0][0])
        results.append({"candidate": filename, "owner_name": owner_name, "score": round(score, 2)})

    results.sort(key=lambda x: x["score"], reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1

    logger.info(f"HR pipeline scored {len(results)} candidates")
    return results
