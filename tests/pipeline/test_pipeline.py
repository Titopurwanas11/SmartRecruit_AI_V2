import pytest
from unittest.mock import patch, MagicMock
import numpy as np
from src.pipeline.pipeline import run_pipeline
import config

@patch("src.pipeline.pipeline.clean_text")
@patch("src.pipeline.pipeline.load_models")
@patch("src.pipeline.pipeline.predict_role")
@patch("src.pipeline.pipeline.load_skill_list")
@patch("src.pipeline.pipeline.extract_skills")
@patch("src.pipeline.pipeline.load_sbert_model")
@patch("src.pipeline.pipeline.embed_text")
@patch("src.pipeline.pipeline.load_job_embeddings")
@patch("src.pipeline.pipeline.compute_similarity")
@patch("src.pipeline.pipeline.get_top_matches")
def test_run_pipeline(
    mock_get_top_matches,
    mock_compute_similarity,
    mock_load_job_embeddings,
    mock_embed_text,
    mock_load_sbert_model,
    mock_extract_skills,
    mock_load_skill_list,
    mock_predict_role,
    mock_load_models,
    mock_clean_text
):
    # Setup mocks
    mock_clean_text.return_value = "cleaned resume"
    
    mock_v = MagicMock()
    mock_clf = MagicMock()
    mock_load_models.return_value = (mock_v, mock_clf)
    mock_predict_role.return_value = "Data Science"
    
    mock_skill_list = ["Python", "SQL", "Java"]
    mock_load_skill_list.return_value = mock_skill_list
    mock_extract_skills.return_value = ["Python", "SQL"]
    
    mock_sbert = MagicMock()
    mock_load_sbert_model.return_value = mock_sbert
    
    mock_resume_vec = np.zeros(384)
    mock_embed_text.return_value = mock_resume_vec
    
    mock_job_vecs = np.zeros((10, 384))
    mock_job_meta = [{"title": "Job 1", "required_skills": ["Python", "Machine Learning"]}] * 10
    mock_load_job_embeddings.return_value = (mock_job_vecs, mock_job_meta)
    
    mock_scores = np.zeros(10)
    mock_compute_similarity.return_value = mock_scores
    
    mock_top_matches = [
        {
            "title": "Data Scientist",
            "company": "Tech Co",
            "score": 0.95,
            "required_skills": ["Python", "SQL", "Machine Learning"]
        }
    ]
    mock_get_top_matches.return_value = mock_top_matches
    
    # Run pipeline
    result = run_pipeline("Raw resume text")
    
    # Assertions
    assert result["predicted_role"] == "Data Science"
    assert result["extracted_skills"] == ["Python", "SQL"]
    assert result["top_matches"] == mock_top_matches
    
    # Skill gap assertions
    # Resume skills: ["Python", "SQL"]
    # Top match skills: ["Python", "SQL", "Machine Learning"]
    # Matched: ["Python", "SQL"]
    # Missing: ["Machine Learning"]
    assert "skill_gap" in result
    assert set(result["skill_gap"]["matched_skills"]) == {"Python", "SQL"}
    assert set(result["skill_gap"]["missing_skills"]) == {"Machine Learning"}
    
    # Verify call order (approximate via mock calls)
    mock_clean_text.assert_called_once_with("Raw resume text")
    mock_load_models.assert_called_once()
    mock_predict_role.assert_called_once_with("cleaned resume", mock_v, mock_clf)
    mock_load_skill_list.assert_called_once()
    mock_extract_skills.assert_called_once_with("cleaned resume", mock_skill_list)
    mock_load_sbert_model.assert_called_once()
    mock_embed_text.assert_called_once_with("cleaned resume", mock_sbert)
    mock_load_job_embeddings.assert_called_once()
    mock_compute_similarity.assert_called_once_with(mock_resume_vec, mock_job_vecs)
    mock_get_top_matches.assert_called_once_with(mock_scores, mock_job_meta, top_n=config.TOP_N_MATCHES)

def test_run_pipeline_empty_matches():
    # Test case where top_matches might be empty (though unlikely in real usage)
    with patch("src.pipeline.pipeline.clean_text") as mock_clean, \
         patch("src.pipeline.pipeline.load_models") as mock_load_m, \
         patch("src.pipeline.pipeline.predict_role") as mock_predict, \
         patch("src.pipeline.pipeline.load_skill_list") as mock_load_s, \
         patch("src.pipeline.pipeline.extract_skills") as mock_extract, \
         patch("src.pipeline.pipeline.load_sbert_model") as mock_load_sb, \
         patch("src.pipeline.pipeline.embed_text") as mock_embed, \
         patch("src.pipeline.pipeline.load_job_embeddings") as mock_load_j, \
         patch("src.pipeline.pipeline.compute_similarity") as mock_sim, \
         patch("src.pipeline.pipeline.get_top_matches") as mock_top:
             
        mock_clean.return_value = "cleaned"
        mock_load_m.return_value = (None, None)
        mock_predict.return_value = "Role"
        mock_load_s.return_value = []
        mock_extract.return_value = ["Skill"]
        mock_load_sb.return_value = None
        mock_embed.return_value = None
        mock_load_j.return_value = (None, None)
        mock_sim.return_value = None
        mock_top.return_value = [] # Empty matches
        
        result = run_pipeline("text")
        
        assert result["top_matches"] == []
        assert result["skill_gap"]["matched_skills"] == []
        assert result["skill_gap"]["missing_skills"] == []
