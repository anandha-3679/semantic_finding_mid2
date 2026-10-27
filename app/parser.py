import io, re
from pathlib import Path
from docx import Document

_ocr = None

def _ocr_page(page):
    """Render a PDF page to an image and read the text with OCR."""
    global _ocr
    if _ocr is None:
        from rapidocr_onnxruntime import RapidOCR
        _ocr = RapidOCR()
    png = page.get_pixmap(dpi=200).tobytes("png")
    result, _ = _ocr(png)
    if not result:
        return ""
    return "\n".join(line[1] for line in result)  # line = [box, text, confidence]

def extract(filename: str, data: bytes):
    """Returns list of (page_or_None, paragraph_text)."""
    ext = Path(filename).suffix.lower()
    out = []
    if ext == ".pdf":
        import fitz  # PyMuPDF
        doc = fitz.open(stream=data, filetype="pdf")
        for p, page in enumerate(doc, 1):
            text = page.get_text("text") or ""
            if len(text.strip()) < 50:          # little/no real text -> scanned page
                text = _ocr_page(page)
            for para in re.split(r"\n\s*\n", text):
                out.append((p, para.replace("\n", " ")))
    elif ext == ".docx":
        for para in Document(io.BytesIO(data)).paragraphs:
            out.append((None, para.text))
    elif ext in (".txt", ".md"):
        text = data.decode("utf-8", errors="ignore")
        for para in re.split(r"\n\s*\n", text):
            out.append((None, para.replace("\n", " ")))
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return out