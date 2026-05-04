import io
import zipfile
import logging
import os
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_bytes: bytes, filename: str = "file.pdf") -> str:
    """
    Input:  raw PDF bytes, optional filename for logging
    Output: concatenated text string from all pages, empty string if no text
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        logger.error(f"Failed to extract text from {filename}: {e}")
        return ""


def extract_resumes_from_zip(zip_bytes: bytes) -> list[tuple[str, str]]:
    """
    Input:  raw ZIP bytes containing PDF files
    Output: list of (candidate_name, resume_text) — one per PDF found in ZIP
            candidate_name is the filename without extension
            Non-PDF entries are silently skipped
    """
    results = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for name in zf.namelist():
                if not name.lower().endswith(".pdf"):
                    continue
                pdf_bytes = zf.read(name)
                text = extract_text_from_pdf(pdf_bytes, name)
                candidate_name = os.path.splitext(os.path.basename(name))[0]
                results.append((candidate_name, text))
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file: {e}")
    return results
