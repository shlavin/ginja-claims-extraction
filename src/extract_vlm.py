
"""
extract_vlm.py

Extracts structured claim fields from documents using a Vision Language
Model (VLM) hosted on Groq. This module is intended for documents that
are difficult to process using traditional OCR, particularly the
handwritten Eden Care claim forms used in this project.

Unlike OCR, which converts image pixels into characters, a VLM analyses
the document as a whole and uses visual and textual context to interpret
its contents. During evaluation, this approach was more effective at
extracting handwritten fields such as treatment dates, diagnosis
descriptions, and claimed amounts that could not be recovered reliably
using Tesseract OCR.

The document page is rendered as an image and submitted to Groq's vision
model, which returns structured JSON matching the shared extraction
schema used throughout the pipeline.

Configuration
-------------
A valid Groq API key is required and should be stored in the project's
`.env` file as:

    GROQ_API_KEY=<your_api_key>

The implementation uses the
`meta-llama/llama-4-scout-17b-16e-instruct` vision model because it
supports image input and structured JSON responses, which simplifies the
conversion of model output into the project's schema. Since Groq updates
its available models over time, the configured model should be verified
against the current documentation if this project is revisited.

Image Resolution
----------------
The input image resolution was selected after evaluating several DPI
values. Groq limits both image resolution (33 megapixels) and encoded
image size (4 MB). Rendering the claim forms at their default DPI
exceeded these limits, while rendering at 130 DPI kept the images within
the allowed size without noticeably affecting readability. This value is
therefore used as the default throughout the module.
"""



import os
import json
import base64
import io

from pdf2image import convert_from_path
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY from .env into the environment

Image.MAX_IMAGE_PIXELS = None

# Same shell-PATH caveat as extract_ocr.py - see that file's header.
POPPLER_PATH = None  # if the need arises that is doesnt work from powershell set to  r"C:\poppler-26.02.0\Library\bin" 

MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
RENDER_DPI = 130  # chosen empirically to stay under Groq's 33MP / 4MB limits

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


def empty_record() -> dict:
    return {field: None for field in SCHEMA_FIELDS}


def pdf_to_base64_jpeg(pdf_path: str, dpi: int = RENDER_DPI) -> str:
    """
    Converts the first page of the PDF into a JPEG, encoded as base64,
    ready to send to Groq's vision API as a data URL.
    """
    kwargs = {"dpi": dpi}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    pages = convert_from_path(pdf_path, **kwargs)
    image = pages[0]

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_prompt() -> str:
    
    """
Builds the prompt used for structured extraction with the VLM.

The prompt specifies the expected output schema, instructs the model to
return null for missing or uncertain values, and requires the response
to be a single JSON object. These instructions were refined during
testing to improve the consistency of the extracted fields and ensure
that the response could be parsed reliably as JSON.
    """


    fields_description = """
    - claim_id: string, unique claim identifier if present, else null
    - member_id: string, the "Membership number" field
    - provider_name: string, the medical practitioner's name (printed, under "Medical Practitioner Details")
    - provider_id: string, RMDC Reg No if filled in, else null
    - date_of_service: string, the handwritten "Treatment date" field, in whatever format it is written (do not try to reformat it yourself)
    - diagnosis_code: string, this form has no ICD code field filled in for these documents in most cases - if the ICD Code box is empty, return null. Do NOT put the diagnosis description text here.
    - procedure_code: string, RMPC Procedure Code column in the services table, if filled in, else null
    - claimed_amount: number, the value in the "Total" row of the "Total Claimed" column at the bottom of the services table. Return it as a plain number, no currency symbols or commas.
    - currency: string, always "KES" for these documents
    - invoice_number: string, if present anywhere on the form, else null
    """

    return f"""You are extracting structured data from a scanned, partially handwritten health insurance claim form.

Some fields on this form are PRINTED/TYPED (e.g. Membership number, Surname, Practitioner Name) and some are HANDWRITTEN in pen (e.g. Specialization, Treatment date, Final Diagnosis Description, and the Services/Items table with procedure names and billed/claimed amounts). Read the handwriting carefully - it may be cursive or stylized.

Extract exactly these fields:
{fields_description}

Rules:
- If a field is not present, illegible, or you are not confident in your reading, return null for that field. Do NOT guess or invent a plausible-looking value.
- Return ONLY a single valid JSON object with exactly these keys: {", ".join(SCHEMA_FIELDS)}
- Do not include any explanation, markdown formatting, or text outside the JSON object.
"""


def extract_vlm(pdf_path: str) -> dict:
    """
    Main entry point. Sends the document image to Groq's vision model
    and parses the JSON response into the shared schema.
    """
    base64_image = pdf_to_base64_jpeg(pdf_path)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    completion = client.chat.completions.create(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": build_prompt()},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        temperature=0,  # deterministic extraction, not creative writing
        max_completion_tokens=1024,
        response_format={"type": "json_object"},
    )

    raw_response = completion.choices[0].message.content

    record = empty_record()
    parse_error = None
    try:
        parsed = json.loads(raw_response)
        for field in SCHEMA_FIELDS:
            record[field] = parsed.get(field)
    except json.JSONDecodeError as e:
        # If the model ever breaks the "JSON only" instruction, we
        # record the failure rather than crashing the whole pipeline -
        # same "flag, don't silently fail" principle as extract_ocr.py.
        parse_error = str(e)

    record["extraction_method"] = "vlm"
    record["raw_text_snippet"] = raw_response[:200]
    if parse_error:
        record["parse_error"] = parse_error

    return record


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_vlm.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_vlm(path)
    print(json.dumps(result, indent=2))