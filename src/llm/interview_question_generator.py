import logging
import os
import sys

import config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert HR interviewer and recruitment coach.
Generate personalized interview questions for a candidate applying to a specific role.

Rules:
- Produce exactly 5 questions: 3 technical and 2 behavioral.
- For each question, include:
  - the question text
  - the purpose of the question
  - the expected insight the interviewer should look for
  - a scoring rubric (Poor, Good, Excellent)
- Keep the output plain markdown with this exact structure:

[Question]
**Purpose**: [brief purpose]
**Expected Insight**: [expected insight]
**Rubric**:
- *Poor*: [indicator of a poor answer]
- *Good*: [indicator of a good answer]
- *Excellent*: [indicator of an excellent answer]

- Do not add any extra sections, analysis, or narrative.
- Do not number the questions in the response.
- Use the candidate resume, matched skills, missing skills, and job description to personalize each question.
- Highlight real risks or weaknesses where the candidate has skill gaps.
-- Brevity rules:
 - Be concise. Each question text should be no longer than 20 words.
 - Keep `Purpose:` to 15 words or less and `Expected Insight:` to 20 words or less.
 - Keep the rubric items brief (10-15 words each).
 - Prefer short phrases and action-oriented language; avoid lengthy examples.
"""


def _real_load_api_key() -> str | None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
        return os.getenv("GEMINI_API_KEY")
    except Exception:
        # If dotenv is not installed or fails, return None so caller can fall back.
        return None


def _init_module_defaults() -> None:
    _mod = sys.modules[__name__]
    if not hasattr(_mod, "genai"):
        try:
            import google.generativeai as _genai  # noqa: PLC0415
            _mod.genai = _genai
        except Exception:
            # Allow the module to be imported and prompt-builder to be used
            # even when the `google` package is not installed in the env.
            _mod.genai = None

    if not hasattr(_mod, "_load_api_key"):
        _mod._load_api_key = _real_load_api_key


_init_module_defaults()


def build_interview_questions_prompt(
    target_role: str,
    job_description: str,
    matched_skills: list[str],
    missing_skills: list[str],
    resume_text: str,
) -> str:
    matched = ", ".join(matched_skills) if matched_skills else "None"
    missing = ", ".join(missing_skills) if missing_skills else "None"
    resume_excerpt = resume_text.strip()
    if len(resume_excerpt) > 1400:
        resume_excerpt = resume_excerpt[:1400].rsplit("\n", 1)[0] + "\n..."

    return (
        f"""Target Role: {target_role}

Job Description:
{job_description.strip()}

Candidate Resume Excerpt:
{resume_excerpt}

Skills present in the resume: {matched}
Skills missing for this role: {missing}

Generate 3 technical questions and 2 behavioral questions.
Each question must be tailored to the candidate's profile and the target role.
Use the format exactly as requested.
Please be concise: each question, purpose, and expected insight should be short (see system brevity rules).
"""
    )


def generate_interview_questions(
    target_role: str,
    job_description: str,
    matched_skills: list[str],
    missing_skills: list[str],
    resume_text: str,
) -> str:
    _mod = sys.modules[__name__]
    api_key = _mod._load_api_key()
    # If Gemini SDK is not available or API key missing, use a local concise fallback generator
    if _mod.genai is None or not api_key:
        logger.info("Gemini not available or API key missing — using local fallback generator")
        return _local_generate_questions(
            target_role, job_description, matched_skills, missing_skills, resume_text
        )

    _mod.genai.configure(api_key=api_key)
    user_prompt = build_interview_questions_prompt(
        target_role=target_role,
        job_description=job_description,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        resume_text=resume_text,
    )

    try:
        model = _mod.genai.GenerativeModel(
            model_name=config.GEMINI_MODEL_NAME,
            system_instruction=_SYSTEM_PROMPT,
        )
        response = model.generate_content(user_prompt)
        logger.info("Gemini interview questions generated successfully")
        return response.text
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        raise RuntimeError(f"Gemini API error: {e}") from e


def _local_generate_questions(
    target_role: str,
    job_description: str,
    matched_skills: list[str],
    missing_skills: list[str],
    resume_text: str,
) -> str:
    """Generate concise interview questions locally when Gemini is unavailable.

    This creates short, structured questions focused on practical evaluation.
    """
    mk = lambda arr: ", ".join(arr) if arr else "None"
    matched = mk(matched_skills)
    missing = mk(missing_skills)

    # Helper to ensure brevity
    def q(text):
        return text.strip()

    parts = []
    parts.append("### Technical Question 1")
    parts.append(f"Question: {q(f'Describe a recent production issue you fixed and your steps.')}")
    parts.append("**Purpose**: Assess debugging and verification approach.")
    parts.append("**Expected Insight**: Clear reproduce→isolate→fix→test steps.")
    parts.append("**Rubric**:")
    parts.append("- *Poor*: Random guessing, no verification.")
    parts.append("- *Good*: Systematic isolation and fix.")
    parts.append("- *Excellent*: Root cause analysis and added automated tests.")

    parts.append("\n### Technical Question 2")
    parts.append(
        f"Question: {q('Given slow ETL, what three prioritized fixes would you try?')}"
    )
    parts.append("**Purpose**: Evaluate performance triage and pragmatic fixes.")
    parts.append("**Expected Insight**: Measurement-first, bottleneck ID, practical mitigations.")
    parts.append("**Rubric**:")
    parts.append("- *Poor*: Suggests complete rewrite without metrics.")
    parts.append("- *Good*: Identifies bottlenecks and scales up resources.")
    parts.append("- *Excellent*: Analyzes query plans, optimizes indexes, and partitions data.")

    parts.append("\n### Technical Question 3")
    parts.append(
        f"Question: {q('How would you deduplicate high-volume streaming data with limited memory?')}"
    )
    parts.append("**Purpose**: Test production trade-offs and reliability.")
    parts.append("**Expected Insight**: Windowed/stateful dedupe, TTL or external store options.")
    parts.append("**Rubric**:")
    parts.append("- *Poor*: In-memory hash set without bounds.")
    parts.append("- *Good*: Uses an external key-value store (e.g. Redis).")
    parts.append("- *Excellent*: Bloom filters or time-windowed state stores.")

    parts.append("\n### Behavioral Question 1")
    parts.append(
        f"Question: {q('Tell me about a time you disagreed with a team decision and what you did.') }"
    )
    parts.append("**Purpose**: Assess communication and constructive conflict handling.")
    parts.append("**Expected Insight**: Evidence-based feedback, compromise, and lessons learned.")
    parts.append("**Rubric**:")
    parts.append("- *Poor*: Ignored decision or complained without solutions.")
    parts.append("- *Good*: Shared concerns, committed to the final decision.")
    parts.append("- *Excellent*: Proposed data-driven alternatives and fostered team alignment.")

    parts.append("\n### Behavioral Question 2")
    parts.append(
        f"Question: {q('Describe when you learned a new tool quickly and applied it to deliver results.') }"
    )
    parts.append("**Purpose**: Gauge learning agility and fast application.")
    parts.append("**Expected Insight**: Rapid learning steps, prototypes, measurable impact.")
    parts.append("**Rubric**:")
    parts.append("- *Poor*: Struggles to articulate learning method.")
    parts.append("- *Good*: Read docs, built a basic prototype.")
    parts.append("- *Excellent*: Built prototype, integrated it, and shared knowledge with the team.")

    return "\n".join(parts)
