import os
import pickle
import logging
import pandas as pd

logger = logging.getLogger(__name__)

def build_skill_list(job_skill_csv_path: str, output_path: str) -> None:
    """
    Input:  path to job_skills.csv, output path for skill_list.pkl
    Output: None
    Side effect: builds deduplicated skill list from 'skills' column, saves as .pkl
    Run this ONCE manually before using load_skill_list()
    CSV schema: job_title,skills  (skills column is comma-separated string)
    """
    if not os.path.exists(job_skill_csv_path):
        raise FileNotFoundError(
            f"[SmartRecruitAI] File not found: {job_skill_csv_path}\n"
            f"Check config.py paths and ensure required files are generated."
        )
    
    logger.info(f"Loading skills from {job_skill_csv_path}")
    
    try:
        # Load CSV
        df = pd.read_csv(job_skill_csv_path)
        
        if 'skills' not in df.columns:
            logger.error(f"Column 'skills' not found in {job_skill_csv_path}")
            raise ValueError(f"CSV must contain a 'skills' column.")
            
        # Extract skills
        # Some rows might be NaN, so drop them or fill them
        all_skills_series = df['skills'].dropna().astype(str)
        
        # Split by comma and flatten
        skills_nested = all_skills_series.str.split(',')
        skills_flat = [skill.strip() for sublist in skills_nested for skill in sublist if skill.strip()]
        
        # Deduplicate
        unique_skills = list(set(skills_flat))
        
        logger.info(f"Extracted {len(unique_skills)} unique skills")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Save as .pkl
        with open(output_path, 'wb') as f:
            pickle.dump(unique_skills, f)
            
        logger.info(f"Saved skill list to {output_path}")
        
    except Exception as e:
        logger.error(f"Error building skill list: {str(e)}")
        raise

def load_skill_list(path: str) -> list[str]:
    """
    Input:  absolute path to skill_list.pkl
    Output: list of skill strings (e.g., ["Python", "SQL", "Docker", ...])
    Error:  raise FileNotFoundError if path does not exist
    Note:   skill_list.pkl is built from job_skills.csv — see build_skill_list() below
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[SmartRecruitAI] File not found: {path}\n"
            f"Check config.py paths and ensure required files are generated."
        )
    
    logger.info(f"Loading from {path}")
    
    try:
        with open(path, 'rb') as f:
            skill_list = pickle.load(f)
        return skill_list
    except Exception as e:
        logger.error(f"Error loading skill list: {str(e)}")
        raise

def extract_skills(text: str, skill_list: list[str]) -> list[str]:
    """
    Input:  cleaned resume text string + skill list
    Output: list of matched skill strings found in the text
    Method: EXACT string match, case-insensitive
            Implementation: [skill for skill in skill_list if skill.lower() in text.lower()]
    Note:   NO fuzzy matching, NO regex, NO external libraries beyond builtins
    """
    if not text:
        return []
    
    text_lower = text.lower()
    found_skills = [skill for skill in skill_list if skill.lower() in text_lower]
    return found_skills
