"""
detect_format.py

Decides whether a PDF is:
  - "text_based"  -> has a real, selectable text layer (e.g. invoices)
  - "image_based" -> is essentially a scanned photo with no text layer
                      (e.g. handwritten claim forms)

Why this matters: it tells pipeline.py which extraction method to run next.
Text-based PDFs go to extract_pdf_text.py.
Image-based PDFs go to extract_ocr.py (and optionally extract_vlm.py).
"""

import pdfplumber  


def get_raw_text(pdf_path: str) -> str:
    """
    Opens the PDF and pulls out whatever text layer exists, across all pages.
    If the PDF has no text layer (it's just an embedded image), this
    will return an empty or near-empty string.
    """
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()  # returns None if no text found
            if page_text:
                all_text.append(page_text)
    return "\n".join(all_text)


def detect_format(pdf_path: str, min_chars: int = 20) -> str:
    """
A minimum text-length threshold is used instead of checking for zero
characters only. During evaluation, the scanned claim forms returned
0 extractable characters, while the digitally generated invoices
returned between 393 and 518 characters.

A threshold of 20 characters was selected to provide a safety margin
for scanned PDFs that may contain a small amount of extractable
metadata or OCR artefacts while still correctly distinguishing them
from text-based PDFs.
    """
    text = get_raw_text(pdf_path)
    stripped_length = len(text.strip())

    if stripped_length < min_chars:
        return "image_based"
    return "text_based"


if __name__ == "__main__":
    
    #manual test
    import sys

    if len(sys.argv) != 2:
        print("Usage: python detect_format.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = detect_format(path)
    print(f"{path} -> {result}")