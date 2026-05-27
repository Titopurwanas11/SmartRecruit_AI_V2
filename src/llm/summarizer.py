import logging
import os
import sys

import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert HR AI assistant.
Your task is to provide a very brief 1-sentence summary of why this candidate is a good or bad fit for the job.
Be extremely concise. Maximum 20 words.
Do not use introductory phrases like "The candidate is...". Start directly with the main point.
Focus on their most relevant experience or glaring skill gap.
"""

def _real_load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return os.getenv("GEMINI_API_KEY")
    except Exception:
        return None

def _init_module_defaults() -> None:
    _mod = sys.modules[__name__]
    if not hasattr(_mod, "genai"):
        try:
            import google.generativeai as _genai
            _mod.genai = _genai
        except Exception:
            _mod.genai = None

    if not hasattr(_mod, "_load_api_key"):
        _mod._load_api_key = _real_load_api_key

_init_module_defaults()

def generate_ai_summary(resume_text: str, jd_text: str) -> str:
    _mod = sys.modules[__name__]
    api_key = _mod._load_api_key()
    
    if _mod.genai is None or not api_key:
        logger.info("Gemini not available — using local fallback summary")
        return "Skor AI didasarkan pada perhitungan kemiripan teks."

    _mod.genai.configure(api_key=api_key)
    
    resume_excerpt = resume_text.strip()
    if len(resume_excerpt) > 1000:
        resume_excerpt = resume_excerpt[:1000] + "..."
        
    jd_excerpt = jd_text.strip()
    if len(jd_excerpt) > 1000:
        jd_excerpt = jd_excerpt[:1000] + "..."

    user_prompt = f"Job Description:\n{jd_excerpt}\n\nCandidate Resume:\n{resume_excerpt}\n\nProvide 1 short sentence summary."

    try:
        model = _mod.genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(user_prompt)
        return response.text.strip().replace("\n", " ")
    except Exception as e:
        logger.error(f"Gemini API call failed for summary: {e}")
        return "AI Summary unavailable due to API error."
