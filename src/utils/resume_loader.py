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


def extract_owner_name(text: str) -> str:
    """
    Simple heuristic to extract the owner's name from a resume text.
    Usually, the name is the first non-empty line of the document.
    """
    for line in text.strip().split('\n'):
        line = line.strip()
        # Consider a line as a name if it's not too short and not too long
        if 2 < len(line) < 50:
            return line.title()
    return "Unknown"


def extract_resumes_from_zip(zip_bytes: bytes) -> list[tuple[str, str, str]]:
    """
    Input:  raw ZIP bytes containing PDF files
    Output: list of (candidate_filename, owner_name, resume_text) — one per PDF found in ZIP
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
                candidate_filename = os.path.splitext(os.path.basename(name))[0]
                owner_name = extract_owner_name(text)
                results.append((candidate_filename, owner_name, text))
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file: {e}")
    return results
