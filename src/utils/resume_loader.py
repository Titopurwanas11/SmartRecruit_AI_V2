import io
import zipfile
import logging
import os
import fitz  # PyMuPDF
import re

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


def extract_owner_name_from_pdf(pdf_bytes: bytes, filename: str = "file.pdf") -> str:
    """
    Try to infer the candidate name from the largest font near the top of the PDF.
    Falls back to plain-text extraction if the font-based heuristic fails.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        candidates = []

        for page in doc:
            page_dict = page.get_text("dict")
            for block in page_dict.get("blocks", []):
                if block.get("type") != 0:
                    continue
                block_bbox = block.get("bbox", [0, 0, 0, 0])
                block_y = block_bbox[1]
                if block_y > 250:
                    continue

                for line in block.get("lines", []):
                    line_text = "".join(span.get("text", "") for span in line.get("spans", []))
                    line_text = line_text.strip()
                    if not line_text:
                        continue

                    line_size = max((span.get("size", 0) for span in line.get("spans", [])), default=0)
                    if line_size < 12:
                        continue

                    words = line_text.split()
                    if len(words) > 4 or len(words) < 2:
                        continue
                    if any(token in line_text.lower() for token in [
                        "skills", "experience", "education", "summary", "contact", "profile",
                        "work history", "professional", "additional information", "resume"
                    ]):
                        continue
                    if not all(word[0].isupper() and word.isalpha() for word in words):
                        continue
                    candidates.append((line_size, -block_y, line_text))

        if candidates:
            return max(candidates)[2].title()
    except Exception as e:
        logger.error(f"Failed to extract owner name from {filename} using font heuristic: {e}")

    fallback_text = extract_text_from_pdf(pdf_bytes, filename)
    return extract_owner_name(fallback_text)


def extract_contact_info(text: str) -> str:
    """
    Extract email and phone number from text using regex.
    """
    # Email regex
    email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    email = email_match.group(0) if email_match else ""
    
    # Phone regex (look for 10-15 digits, optionally with spaces/dashes/plus)
    phone_match = re.search(r'\+?[\d\s-]{10,15}', text)
    phone = phone_match.group(0).strip() if phone_match else ""
    
    if email and phone:
        return f"{email} | {phone}"
    elif email:
        return email
    elif phone:
        return phone
    return "Not Found"


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
                owner_name = extract_owner_name_from_pdf(pdf_bytes, name)
                results.append((candidate_filename, owner_name, text))
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file: {e}")
    return results
