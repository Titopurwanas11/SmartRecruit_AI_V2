import logging
import os
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

def load_vectorizer(path: str) -> TfidfVectorizer:
    """
    Input:  absolute path to .joblib file
    Output: loaded TfidfVectorizer object
    Error:  raise FileNotFoundError if path does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[SmartRecruitAI] File not found: {path}\n"
            f"Check config.py paths and ensure required files are generated."
        )
    logger.info(f"Loading from {path}")
    return joblib.load(path)

def vectorize(text: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    """
    Input:  single cleaned string + loaded vectorizer
    Output: sparse matrix shape (1, n_features)
    Note:   use vectorizer.transform([text]), not fit_transform
    """
    return vectorizer.transform([text])
