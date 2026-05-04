import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import logging
import time
import os

# Constants mirroring config.py for standalone Colab execution
MAX_JOB_SAMPLE = 20000
RANDOM_SEED = 42
SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
INPUT_FILE = "job_descriptions.csv"
OUTPUT_EMBEDDINGS = "job_embeddings.npy"
OUTPUT_METADATA = "job_metadata.pkl"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_embeddings():
    """
    Standalone script to pre-compute SBERT embeddings for job descriptions.
    Designed for Google Colab with GPU support.
    """
    start_time = time.time()
    
    # 1. Data Loading
    if not os.path.exists(INPUT_FILE):
        logger.error(f"Input file '{INPUT_FILE}' not found. Please ensure it's in the working directory.")
        return

    logger.info(f"Loading data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    logger.info(f"Loaded {len(df)} rows.")

    # 2. Random Sampling
    if len(df) > MAX_JOB_SAMPLE:
        logger.info(f"Sampling {MAX_JOB_SAMPLE} rows with seed {RANDOM_SEED}...")
        df = df.sample(n=MAX_JOB_SAMPLE, random_state=RANDOM_SEED).reset_index(drop=True)
    else:
        logger.info("Dataset size is smaller than MAX_JOB_SAMPLE, using all rows.")

    # 3. Preprocessing
    logger.info("Preprocessing text data...")
    # Expected columns: title, company, description, skills_desc
    required_cols = ['title', 'company', 'description', 'skills_desc']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Column '{col}' missing. Filling with empty strings.")
            df[col] = ""
    
    df[required_cols] = df[required_cols].fillna("")
    
    # Combine text for encoding
    texts_to_encode = (
        df['title'] + " " + 
        df['description'] + " " + 
        df['skills_desc']
    ).tolist()

    # 4. Load SBERT Model
    logger.info(f"Loading SBERT model '{SBERT_MODEL_NAME}'...")
    model = SentenceTransformer(SBERT_MODEL_NAME)
    
    # 5. Encoding
    logger.info(f"Starting encoding of {len(texts_to_encode)} job descriptions...")
    encoding_start = time.time()
    
    embeddings = model.encode(
        texts_to_encode,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    encoding_duration = time.time() - encoding_start
    logger.info(f"Encoding completed in {encoding_duration:.2f} seconds.")

    # 6. Save Artifacts
    logger.info(f"Saving embeddings to {OUTPUT_EMBEDDINGS}...")
    np.save(OUTPUT_EMBEDDINGS, embeddings)

    logger.info(f"Extracting and saving metadata to {OUTPUT_METADATA}...")
    metadata = []
    for _, row in df.iterrows():
        skills = [s.strip() for s in str(row['skills_desc']).split(',') if s.strip()]
        metadata.append({
            "title": row['title'],
            "company": row['company'],
            "required_skills": skills
        })

    with open(OUTPUT_METADATA, 'wb') as f:
        pickle.dump(metadata, f)

    total_duration = time.time() - start_time
    logger.info(f"Process completed successfully in {total_duration:.2f} seconds.")
    logger.info(f"Artifacts generated: {OUTPUT_EMBEDDINGS}, {OUTPUT_METADATA}")

if __name__ == "__main__":
    generate_embeddings()
