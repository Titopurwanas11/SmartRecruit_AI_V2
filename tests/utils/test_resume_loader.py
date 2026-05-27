import io
import zipfile
import pytest
from unittest.mock import MagicMock, patch
from src.utils.resume_loader import extract_text_from_pdf, extract_resumes_from_zip, extract_owner_name_from_pdf


def test_extract_text_from_pdf_returns_string():
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Python developer with 5 years experience"
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("src.utils.resume_loader.fitz.open", return_value=mock_doc):
        result = extract_text_from_pdf(b"fake_pdf_bytes", "candidate.pdf")

    assert isinstance(result, str)
    assert "Python" in result


def test_extract_text_from_pdf_empty_bytes_returns_empty_string():
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([]))

    with patch("src.utils.resume_loader.fitz.open", return_value=mock_doc):
        result = extract_text_from_pdf(b"", "empty.pdf")

    assert result == ""


def test_extract_resumes_from_zip_returns_list_of_tuples():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("alice.pdf", b"fake_pdf_1")
        zf.writestr("bob.pdf", b"fake_pdf_2")
        zf.writestr("readme.txt", "ignore me")
    buf.seek(0)

    mock_page = MagicMock()
    mock_page.get_text.return_value = "some resume text"
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("src.utils.resume_loader.fitz.open", return_value=mock_doc):
        results = extract_resumes_from_zip(buf.read())

    assert len(results) == 2
    names = [r[0] for r in results]
    assert "alice" in names
    assert "bob" in names
    texts = [r[1] for r in results]
    assert all(isinstance(t, str) for t in texts)


def test_extract_resumes_from_zip_skips_non_pdf():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("notes.docx", b"docx content")
    buf.seek(0)

    results = extract_resumes_from_zip(buf.read())
    assert results == []


def test_extract_owner_name_from_pdf_uses_largest_font_near_top():
    mock_line = MagicMock()
    mock_line.__iter__ = MagicMock(return_value=iter([]))

    mock_span_small = {"text": "Education", "size": 12}
    mock_span_large = {"text": "Jane Doe", "size": 18}

    mock_page = MagicMock()
    mock_page.get_text.side_effect = [
        {"blocks": [{"type": 0, "bbox": (0, 40, 100, 60), "lines": [{"spans": [mock_span_small]}]}, {"type": 0, "bbox": (0, 60, 100, 80), "lines": [{"spans": [mock_span_large]}]}]}
    ]
    mock_doc = MagicMock()
    mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))

    with patch("src.utils.resume_loader.fitz.open", return_value=mock_doc):
        result = extract_owner_name_from_pdf(b"fake_pdf_bytes", "candidate.pdf")

    assert result == "Jane Doe"
