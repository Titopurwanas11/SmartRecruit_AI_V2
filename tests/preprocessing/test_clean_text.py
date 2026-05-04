import pytest
from src.preprocessing.clean_text import clean_text

def test_clean_text_basic():
    """Test basic cleaning: lowercase, special chars, and stopwords."""
    input_text = "Hello! I am a Senior Python Developer."
    # Expected: "hello senior python developer"
    # Stopwords like "i", "am", "a" should be removed
    result = clean_text(input_text)
    assert "hello" in result
    assert "senior" in result
    assert "python" in result
    assert "developer" in result
    assert "!" not in result
    assert "am" not in result

def test_clean_text_urls():
    """Test removal of URLs."""
    input_text = "Check my portfolio at https://github.com/user or http://example.com"
    result = clean_text(input_text)
    assert "github.com" not in result
    assert "http" not in result
    assert "portfolio" in result

def test_clean_text_numbers():
    """Test that numbers are preserved."""
    input_text = "I have 5 years of experience in Python 3.12"
    result = clean_text(input_text)
    assert "5" in result
    assert "3" in result
    assert "12" in result or "3.12" in result.replace(" ", "")

def test_clean_text_empty():
    """Test handling of empty or None inputs."""
    assert clean_text("") == ""
    assert clean_text(None) == ""

def test_clean_text_whitespace():
    """Test removal of extra whitespace."""
    input_text = "  Hello    World  "
    result = clean_text(input_text)
    assert result == "hello world"
