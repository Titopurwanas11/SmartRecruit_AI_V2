from src.preprocessing.clean_text import clean_text
from src.models.predict import load_models, predict_role
from src.nlp.skill_extraction import load_skill_list, extract_skills
from src.matching.embeddings import load_sbert_model, embed_text, load_job_embeddings
from src.matching.similarity import compute_similarity, get_top_matches
import config

def run_pipeline(raw_resume_text: str) -> dict:
    """
    Input:  raw resume string (uncleaned, directly from user input or PDF)
    Output: {
        "predicted_role":   str,          # e.g. "Data Science"
        "extracted_skills": list[str],    # e.g. ["Python", "SQL", "Pandas"]
        "top_matches":      list[dict],   # 5 dicts with keys: title, company, score, required_skills
        "skill_gap": {
            "matched_skills": list[str],  # skills in resume AND in top match
            "missing_skills": list[str]   # skills in top match NOT in resume
        }
    }
    """
    # 1. clean_text(raw_resume_text)
    cleaned_text = clean_text(raw_resume_text)
    
    # 2. load_models() → predict_role()
    v, clf = load_models()
    predicted_role = predict_role(cleaned_text, v, clf)
    
    # 3. load_skill_list() → extract_skills()
    skill_list = load_skill_list(config.SKILL_LIST_PATH)
    extracted_skills = extract_skills(cleaned_text, skill_list)
    
    # 4. load_sbert_model() → embed_text()
    sbert_model = load_sbert_model()
    resume_embedding = embed_text(cleaned_text, sbert_model)
    
    # 5. load_job_embeddings()
    job_embeddings, job_metadata = load_job_embeddings()
    
    # 6. compute_similarity() → get_top_matches()
    scores = compute_similarity(resume_embedding, job_embeddings)
    top_matches = get_top_matches(scores, job_metadata, top_n=config.TOP_N_MATCHES)
    
    # 7. compute skill_gap using top_matches[0]["required_skills"]
    matched_skills = []
    missing_skills = []
    
    if top_matches:
        top_job_skills = top_matches[0].get("required_skills", [])
        
        # We need a case-insensitive set of extracted skills for comparison
        resume_skills_lower = {s.lower() for s in extracted_skills}
        
        for skill in top_job_skills:
            if skill.lower() in resume_skills_lower:
                matched_skills.append(skill)
            else:
                missing_skills.append(skill)
                
    # 8. return assembled dict
    return {
        "predicted_role": predicted_role,
        "extracted_skills": extracted_skills,
        "top_matches": top_matches,
        "skill_gap": {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        }
    }
