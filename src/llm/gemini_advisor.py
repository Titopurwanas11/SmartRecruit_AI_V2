import logging
import os
import sys

import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert career coach and resume consultant.
Your task is to give specific, actionable advice to help a job seeker improve their resume for a target role.

Rules:
- Give exactly 3 to 5 numbered suggestions.
- Each suggestion must be specific — mention exact skills, tools, certifications, or phrasing to add.
- Focus on the missing skills and how to acquire or demonstrate them.
- Keep the total response under 200 words.
- Use plain markdown (bold for skill names, no headers).
- Do not repeat information already obvious from the skill gap list."""


def _real_load_api_key() -> str | None:
    from dotenv import load_dotenv
    load_dotenv()
    return os.getenv("GEMINI_API_KEY")


def _init_module_defaults() -> None:
    """Attach default callables/objects to the module — only when absent.

    Called at the end of every module load/reload.  Because the check is
    ``not hasattr``, any mock previously placed by a test (which sets the
    attribute) is preserved across ``importlib.reload()``.
    """
    _mod = sys.modules[__name__]

    # Lazy-import google.generativeai and expose as module attribute "genai".
    if not hasattr(_mod, "genai"):
        import google.generativeai as _genai  # noqa: PLC0415
        _mod.genai = _genai

    # Expose the real key-loader as "_load_api_key" only if not already set.
    if not hasattr(_mod, "_load_api_key"):
        _mod._load_api_key = _real_load_api_key


_init_module_defaults()


def generate_resume_advice(
    raw_resume_text: str,
    top_match: dict,
    skill_gap: dict,
) -> str:
    """
    Input:
        raw_resume_text: original resume text (uncleaned)
        top_match: dict with keys: title, company, score, required_skills
        skill_gap: dict with keys: matched_skills (list), missing_skills (list)
    Output:
        Gemini advice string (markdown formatted, ~200 words)
    """
    # Look up through the live module dict so that unittest.mock patches
    # applied to "src.llm.gemini_advisor.<name>" are always respected,
    # even when this function is called after importlib.reload().
    _mod = sys.modules[__name__]
    api_key = _mod._load_api_key()
    if not api_key:
        raise RuntimeError(
            "Gemini API key not found. Set GEMINI_API_KEY in .env"
        )

    _active_genai = _mod.genai
    _active_genai.configure(api_key=api_key)

    matched = ", ".join(skill_gap.get("matched_skills", [])) or "None"
    missing = ", ".join(skill_gap.get("missing_skills", [])) or "None"
    required = ", ".join(top_match.get("required_skills", [])) or "None"

    user_prompt = f"""Target Role: {top_match['title']} at {top_match['company']}
Match Score: {top_match['score']}
Required Skills for this role: {required}

Skills already in my resume: {matched}
Skills I am missing: {missing}

My resume (excerpt):
\"\"\"
{raw_resume_text[:1500]}
\"\"\"

Please give me specific suggestions to improve my resume for this role."""

    try:
        model = _active_genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(user_prompt)
        logger.info("Gemini advice generated successfully")
        return response.text
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise RuntimeError(f"Gemini API error: {e}") from e
