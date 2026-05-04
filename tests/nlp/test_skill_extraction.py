import os
import pickle
import pandas as pd
import pytest
from src.nlp.skill_extraction import build_skill_list, load_skill_list, extract_skills

def test_build_skill_list(tmp_path):
    # Setup temporary files
    csv_path = tmp_path / "test_skills.csv"
    pkl_path = tmp_path / "skills.pkl"
    
    # Create sample data
    data = {
        'job_title': ['Software Engineer', 'Data Scientist', 'Frontend Developer'],
        'skills': ['Python, Java, Git', 'Python, R, SQL', 'JavaScript, HTML, CSS, Git']
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    
    # Call the function
    build_skill_list(str(csv_path), str(pkl_path))
    
    # Verify output file exists
    assert os.path.exists(pkl_path)
    
    # Verify content
    with open(pkl_path, 'rb') as f:
        skills = pickle.load(f)
    
    expected_skills = sorted(['Python', 'Java', 'Git', 'R', 'SQL', 'JavaScript', 'HTML', 'CSS'])
    assert sorted(skills) == expected_skills
    assert len(skills) == 8 # Deduplicated

def test_build_skill_list_file_not_found():
    with pytest.raises(FileNotFoundError):
        build_skill_list("non_existent.csv", "output.pkl")

def test_load_skill_list(tmp_path):
    pkl_path = tmp_path / "skills.pkl"
    sample_skills = ["Python", "SQL", "Docker"]
    with open(pkl_path, 'wb') as f:
        pickle.dump(sample_skills, f)
    
    loaded_skills = load_skill_list(str(pkl_path))
    assert loaded_skills == sample_skills

def test_load_skill_list_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_skill_list("non_existent.pkl")

def test_extract_skills():
    skill_list = ["Python", "SQL", "Docker", "Machine Learning"]
    text = "I am a developer with experience in Python and SQL. I also know about Docker."
    
    extracted = extract_skills(text, skill_list)
    assert "Python" in extracted
    assert "SQL" in extracted
    assert "Docker" in extracted
    assert "Machine Learning" not in extracted
    assert len(extracted) == 3

def test_extract_skills_empty():
    assert extract_skills("", ["Python"]) == []
    assert extract_skills("Some text", []) == []
