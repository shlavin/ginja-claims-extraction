
"""
extract_ocr.py

Extracts structured claim fields from scanned, image-based Eden Care
claim forms using Tesseract OCR.

The extraction pipeline consists of five steps:
  1. Convert the PDF page to an image.
  2. Apply image preprocessing.
  3. Perform OCR to extract text.
  4. Parse the extracted text into the shared schema.
  5. Mark fields that cannot be extracted reliably instead of assigning
     potentially incorrect values.

Evaluation was carried out on the claim forms used in this project.
The printed fields (such as Membership Number, Name, Cellphone, Email,
and Practitioner Name) were generally extracted reliably after
preprocessing. In contrast, handwritten fields including Treatment Date,
Healthcare Facility, Diagnosis Description, and the Services/Items table
were extracted inconsistently.

Four preprocessing approaches were evaluated:

  - Grayscale only
  - Grayscale with Otsu thresholding
  - 1.5× upscaling with adaptive Gaussian thresholding
  - Denoising with Otsu thresholding

Although the preprocessing methods improved the readability of printed
text, none produced consistently accurate results for handwritten
content. For example, the handwritten value "Obs / Gyne" was recognised
as "O\\s | G\\ ae" by the best-performing configuration. This reflects a
known limitation of traditional OCR when applied to cursive handwriting
rather than an issue with preprocessing alone.

Based on these observations, this module extracts only the fields that
can be recovered with reasonable confidence. Fields that cannot be
reliably interpreted are left as None and recorded in
`low_confidence_fields`. More reliable extraction of handwritten content
is handled separately by the VLM-based approach implemented in
`extract_vlm.py`.
"""



import re
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Disable Pillow's decompression bomb warning for the evaluation dataset.


POPPLER_PATH = None  # set to the string above if convert_from_path can't find poppler


SCHEMA_FIELDS = [
    "claim_id",
    "member_id",
    "provider_name",
    "provider_id",
    "date_of_service",
    "diagnosis_code",
    "procedure_code",
    "claimed_amount",
    "currency",
    "invoice_number",
]

# Handwritten fields that proved unreliable during evaluation. We still attempt
# them, but anything that doesn't pass a basic sanity check gets
# flagged here instead of trusted.
HANDWRITTEN_RISK_FIELDS = {"date_of_service", "diagnosis_code", "claimed_amount"}


def empty_record() -> dict:
    return {field: None for field in SCHEMA_FIELDS}


def pdf_to_image(pdf_path: str, dpi: int = 200):
    """
    Converts the first page of a scanned PDF into a single OpenCV image
    (numpy array, BGR). dpi=200 is a deliberate choice: Testing showed that 300 DPI increased processing time and image size
    without improving OCR accuracy on the evaluation documents..
    """
    kwargs = {"dpi": dpi}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    pages = convert_from_path(pdf_path, **kwargs)
    pil_image = pages[0]  # all our claim forms are single-page
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def preprocess(img_cv) -> "np.ndarray":
    """
    Denoise + Otsu thresholding. Chosen as the default after testing
    four preprocessing variants on real documents (see module
    docstring) - this combination was the most consistently readable
    for printed text, even though None of the evaluated preprocessing methods substantially improved
    recognition of handwritten fields.
    """
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresholded = cv2.threshold(
        denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresholded


def run_ocr(processed_image) -> str:
    return pytesseract.image_to_string(processed_image)


def extract_printed_fields(raw_text: str) -> dict:
    """
    Regex extraction for the fields that are PRINTED/TYPED on the form
    (not handwritten) -  These fields were consistently extracted during evaluation.. Patterns are deliberately loose (case-insensitive, tolerant of
    OCR noise like stray punctuation) since even "good" OCR output on
    these forms isn't perfectly clean.
    """
    record = empty_record()

    member_match = re.search(r"Membership number[\W_]{0,3}(\d+)", raw_text, re.IGNORECASE)
    if member_match:
        record["member_id"] = member_match.group(1).strip()

    # Provider name isn't on the claim form itself - it's the practitioner's
    # name, not the facility. We capture practitioner name as provider_name
    # since the form has no separate "facility name" field that OCR'd reliably.
    # During evaluation, OCR occasionally inserted underscores between
    # labels and values, so the pattern tolerates them explicitly.on Sharma's form, where OCR produced "Name _- Wanjiku Mwangi".
    name_match = re.search(r"\bName[\W_]{0,5}([A-Z][A-Za-z\.\s]+?)(?:\n|Specialization)", raw_text)
    if name_match:
        record["provider_name"] = name_match.group(1).strip()

    return record


def extract_handwritten_fields_with_flags(raw_text: str) -> tuple[dict, list[str]]:
    """
    Attempts extraction of the fields we know are handwritten on these
    forms, but validates each against a basic sanity pattern before
    trusting it. Anything that fails the sanity check is left as None
    and added to low_confidence_fields - this is the "flag, don't
    guess" behaviour the case study asks for.
    """
    record = {}
    low_confidence = []

    # Treatment date: expect something like "23-10-2025" or similar digit groups.
    # OCR on the handwritten date sometimes partially works since digits are
    # more OCR-friendly than cursive letters - but only trust it if it
    # actually matches a plausible date shape.
    date_match = re.search(r"Treatment date\D{0,15}(\d{1,2}\D{1,3}\d{1,2}\D{1,3}\d{4})", raw_text)
    if date_match and re.match(r"^\d{1,2}\D{1,3}\d{1,2}\D{1,3}\d{4}$", date_match.group(1)):
        record["date_of_service"] = date_match.group(1)
    else:
        record["date_of_service"] = None
        low_confidence.append("date_of_service")

    # Diagnosis description and claimed_amount: based on direct testing,
    # these are not reliably recoverable from OCR text on handwritten
    # forms. No extraction is attempted because evaluation showed these fields
    # could not be recovered reliably from OCR output. - we flag them outright.
    # (extract_vlm.py is the intended method for these fields on this
    # document type; see writeup.md for the comparison.)
    record["diagnosis_code"] = None
    low_confidence.append("diagnosis_code")

    record["claimed_amount"] = None
    low_confidence.append("claimed_amount")

    return record, low_confidence


def extract_ocr(pdf_path: str) -> dict:
    """
    Main entry point. Given a path to a scanned/handwritten claim form
    PDF, returns one structured record matching the shared schema, plus
    pipeline metadata (extraction_method, raw_text_snippet,
    low_confidence_fields).
    """
    img_cv = pdf_to_image(pdf_path)
    processed = preprocess(img_cv)
    raw_text = run_ocr(processed)

    record = extract_printed_fields(raw_text)
    handwritten_record, low_confidence_fields = extract_handwritten_fields_with_flags(raw_text)
    record.update(handwritten_record)

    record["currency"] = "KES"  # consistent across all documents in this project
    record["extraction_method"] = "ocr"
    record["raw_text_snippet"] = raw_text.strip()[:200]
    record["low_confidence_fields"] = low_confidence_fields

    return record


if __name__ == "__main__":
    import sys
    import json

    if len(sys.argv) != 2:
        print("Usage: python extract_ocr.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_ocr(path)
    print(json.dumps(result, indent=2))