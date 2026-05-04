import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Data paths ────────────────────────────────────────────────────────────────
RAW_DATA_DIR        = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR  = os.path.join(BASE_DIR, "data", "processed")
RESUME_CSV          = os.path.join(RAW_DATA_DIR, "resumes.csv")
JOB_DESC_CSV        = os.path.join(RAW_DATA_DIR, "job_descriptions.csv")
JOB_SKILL_CSV       = os.path.join(RAW_DATA_DIR, "job_skills.csv")

# ── Processed artifacts ───────────────────────────────────────────────────────
JOB_EMBEDDINGS_PATH = os.path.join(PROCESSED_DATA_DIR, "job_embeddings.npy")
JOB_METADATA_PATH   = os.path.join(PROCESSED_DATA_DIR, "job_metadata.pkl")
SKILL_LIST_PATH     = os.path.join(BASE_DIR, "models", "artifacts", "skill_list.pkl")

# ── Model paths ───────────────────────────────────────────────────────────────
VECTORIZER_PATH     = os.path.join(BASE_DIR, "models", "classification", "tfidf_vectorizer.joblib")
CLASSIFIER_PATH     = os.path.join(BASE_DIR, "models", "classification", "classifier.joblib")

# ── Model settings ────────────────────────────────────────────────────────────
SBERT_MODEL_NAME        = "all-MiniLM-L6-v2"
TOP_N_MATCHES           = 5
MAX_JOB_SAMPLE          = 20000
RANDOM_SEED             = 42
MIN_SIMILARITY_THRESHOLD = 0.50

# ── LLM settings ─────────────────────────────────────────────────────────────
GEMINI_MODEL_NAME   = "gemini-2.5-flash"
