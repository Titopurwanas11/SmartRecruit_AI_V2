import pytest
from unittest.mock import patch, MagicMock


TOP_MATCH = {
    "title": "Data Analyst",
    "company": "Acme Corp",
    "score": 0.82,
    "required_skills": ["SQL", "Python", "Tableau", "Excel"],
}
SKILL_GAP = {
    "matched_skills": ["Python", "SQL"],
    "missing_skills": ["Tableau", "Excel"],
}
RESUME_TEXT = "Experienced data professional with Python and SQL skills."


def test_generate_resume_advice_returns_string():
    mock_response = MagicMock()
    mock_response.text = "1. Add Tableau to your skills section.\n2. Get Excel certified."

    with patch("src.llm.gemini_advisor.genai") as mock_genai:
        mock_model = MagicMock()
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        with patch("src.llm.gemini_advisor._load_api_key", return_value="fake-key"):
            from src.llm.gemini_advisor import generate_resume_advice
            result = generate_resume_advice(RESUME_TEXT, TOP_MATCH, SKILL_GAP)

    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_resume_advice_missing_key_raises():
    with patch("src.llm.gemini_advisor._load_api_key", return_value=None):
        from src.llm import gemini_advisor
        import importlib
        importlib.reload(gemini_advisor)
        with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
            gemini_advisor.generate_resume_advice(RESUME_TEXT, TOP_MATCH, SKILL_GAP)


def test_generate_resume_advice_api_error_raises():
    with patch("src.llm.gemini_advisor.genai") as mock_genai:
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("quota exceeded")
        mock_genai.GenerativeModel.return_value = mock_model

        with patch("src.llm.gemini_advisor._load_api_key", return_value="fake-key"):
            from src.llm import gemini_advisor
            import importlib
            importlib.reload(gemini_advisor)
            with pytest.raises(RuntimeError, match="Gemini API error"):
                gemini_advisor.generate_resume_advice(RESUME_TEXT, TOP_MATCH, SKILL_GAP)
