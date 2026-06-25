"""
extract_vlm.py

Pulls structured claim fields out of documents using a Vision Language
Model (VLM) on Groq, instead of OCR or plain text extraction. I built
this originally for the toughest documents in the set - the handwritten
Eden Care claim forms, where extract_ocr.py just couldn't read the
handwritten fields reliably (diagnosis description, treatment date,
claimed amounts). It also runs on the clean Nairobi Lifecare invoices
though, so every document ends up with a result from all three
extraction methods (pdf_text, ocr, vlm) - that way the method
comparison is a real three-way comparison, not just "here's what we
did for the hard cases."

How this is different from extract_ocr.py:
  - OCR is just pixels -> characters, with zero understanding of what
    it's looking at. It doesn't know "Obs / Gyne" is a medical
    specialization - it's just pattern-matching shapes to letters.
  - A VLM looks at the whole page and reasons about it more like a
    person would: it knows what a claims form is supposed to look
    like, has a sense of what should be in a diagnosis field, and can
    use that context to make a far better guess at messy handwriting
    than character-by-character OCR ever could.

The approach here is to send the rendered page image straight to
Groq's vision model and ask it to hand back strict JSON matching our
shared schema.

GROQ API SETUP:
  1. Sign up free at console.groq.com, no card required.
  2. Create an API key under "API Keys" in the console.
  3. Add it to a .env file in the project root:
       GROQ_API_KEY=gsk_your_key_here
  4. pip install groq python-dotenv

MODEL CHOICE:
  meta-llama/llama-4-scout-17b-16e-instruct - picked because, as of
  writing, it's one of only two vision-capable models on Groq, and it
  explicitly supports JSON mode (response_format = json_object), which
  is exactly what this task needs. Worth double-checking
  console.groq.com/docs/vision before you submit your own version of
  this, since Groq's supported model list shifts around over time.

IMAGE SIZE CONSTRAINTS (figured these out by trial and error - notes
below):
  Groq caps images at 33 megapixels and also caps the base64-encoded
  payload at 4MB. Our claim form PDFs, rendered at their default
  conversion DPI, blew past BOTH limits (58 megapixels, 6MB+ encoded).
  After testing a few values, dpi=130 turned out to be the sweet spot -
  it lands comfortably under both caps (~25 megapixels, ~3.3MB encoded)
  while still being legible.
"""

import os
import json
import base64
import io

import groq
from groq import Groq
from pdf2image import convert_from_path
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # pulls GROQ_API_KEY out of .env and into the environment

Image.MAX_IMAGE_PIXELS = None

# Same shell-PATH caveat as in extract_ocr.py - see that file's header.
POPPLER_PATH = None  # set to e.g. r"C:\poppler-26.02.0\Library\bin" if needed

MODEL_ID = "meta-llama/llama-4-scout-17b-16e-instruct"
RENDER_DPI = 130  # found by testing - keeps us under Groq's 33MP / 4MB limits


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
    Converts the first page of the PDF into a JPEG, then base64-encodes
    it so it's ready to hand to Groq's vision API as a data URL.
    """
    kwargs = {"dpi": dpi}
    if POPPLER_PATH:
        kwargs["poppler_path"] = POPPLER_PATH

    pages = convert_from_path(pdf_path, **kwargs)
    image = pages[0]

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_prompt(document_type: str = "claim_form") -> str:
    """
    Builds the extraction prompt. A few notes on why it's written this
    way - these came out of actually testing against the real
    documents, not just guessing what might work:

    - Every field is listed explicitly with its expected type/format,
      rather than something vague like "extract the claim details" -
      vague prompts gave inconsistent key names when I tried them
      early on.
    - The model is explicitly told what to put when a field's missing
      (null), because otherwise it tends to fill the gap with a
      plausible-sounding guess instead of admitting it doesn't know -
      which is exactly the silent-wrong-value failure mode this case
      study is warning against.
    - We ask for ONLY the JSON object, nothing else, because any
      preamble text ("Here is the extracted data:") breaks a naive
      json.loads() call.
    - The prompt tells the model what kind of document it's looking
      at and whether to expect handwriting, which gives it useful
      context for reading the page - the same way a human reviewer
      would expect handwriting on a claim form but not on a
      system-generated invoice.

    document_type: "claim_form" (handwritten Eden Care forms) or
    "invoice" (clean, typed Nairobi Lifecare invoices). This actually
    matters: telling the model a clean printed invoice is "partially
    handwritten" would send it hunting for handwriting that isn't
    there, and the same problem in reverse for claim forms.
    """
    fields_description = """
    - claim_id: string, unique claim identifier if present, else null
    - member_id: string, the member/insurance number field
    - provider_name: string, the medical practitioner's or hospital's name
    - provider_id: string, a practitioner/provider registration number if filled in, else null
    - date_of_service: string, the treatment/visit date, in whatever format it is written (do not try to reformat it yourself)
    - diagnosis_code: string, an ICD code if one is explicitly filled in - if there is only a diagnosis DESCRIPTION (free text, not a code) or the code box is empty, return null. Do NOT put diagnosis description text here.
    - procedure_code: string, a procedure/CPT code if filled in, else null
    - claimed_amount: number, the total amount billed/claimed (use the document's stated Total, not your own sum). Return it as a plain number, no currency symbols or commas.
    - currency: string, always "KES" for these documents
    - invoice_number: string, if present anywhere on the document, else null
    """

    if document_type == "invoice":
        document_description = (
            "You are extracting structured data from a clean, system-generated "
            "hospital outpatient invoice (typed text, no handwriting)."
        )
    else:
        document_description = (
            "You are extracting structured data from a scanned, partially "
            "handwritten health insurance claim form. Some fields on this form "
            "are PRINTED/TYPED (e.g. Membership number, Surname, Practitioner "
            "Name) and some are HANDWRITTEN in pen (e.g. Specialization, "
            "Treatment date, Final Diagnosis Description, and the Services/"
            "Items table with procedure names and billed/claimed amounts). "
            "Read the handwriting carefully - it may be cursive or stylized."
        )

    return f"""{document_description}

Extract exactly these fields:
{fields_description}

Rules:
- If a field is not present, illegible, or you are not confident in your reading, return null for that field. Do NOT guess or invent a plausible-looking value.
- Return ONLY a single valid JSON object with exactly these keys: {", ".join(SCHEMA_FIELDS)}
- Do not include any explanation, markdown formatting, or text outside the JSON object.
"""


def extract_vlm(pdf_path: str, document_type: str = "claim_form") -> dict:
    """
    Main entry point. Sends the document image to Groq's vision model
    and parses the JSON it sends back into the shared schema.

    document_type: "claim_form" or "invoice" - gets passed through to
    build_prompt() so the model is told accurately what kind of
    document it's looking at. Defaults to "claim_form" so existing
    callers keep their original behaviour.

    On error handling: the groq SDK already retries transient failures
    (connection errors, 408/409/429/5xx) twice with backoff before it
    ever raises - so by the time an exception gets here, it's already
    survived the SDK's own retry logic. We still catch Groq's specific
    exception types to give a clearer error message (e.g. "bad API
    key" vs "rate limited" vs "network unreachable") than a generic
    Exception string would, which matters when you're trying to debug
    fast against a deadline. pipeline.py's process_document() wraps
    this in its own try/except too, as a final safety net, so one
    document's API failure can never take down the whole batch run.
    """
    base64_image = pdf_to_base64_jpeg(pdf_path)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    try:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": build_prompt(document_type)},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                }
            ],
            temperature=0,  # we want deterministic extraction, not creative writing
            max_completion_tokens=1024,
            response_format={"type": "json_object"},
        )
    except groq.AuthenticationError as e:
        raise RuntimeError(
            "Groq authentication failed - check that GROQ_API_KEY in .env "
            f"is set correctly. Original error: {e}"
        ) from e
    except groq.RateLimitError as e:
        raise RuntimeError(
            "Groq rate limit exceeded (the SDK already retried this twice "
            f"automatically before giving up). Original error: {e}"
        ) from e
    except groq.APIConnectionError as e:
        raise RuntimeError(
            f"Could not reach Groq's API - check network connectivity. Original error: {e}"
        ) from e

    raw_response = completion.choices[0].message.content

    record = empty_record()
    parse_error = None
    try:
        parsed = json.loads(raw_response)
        for field in SCHEMA_FIELDS:
            record[field] = parsed.get(field)
    except json.JSONDecodeError as e:
        # If the model ever ignores the "JSON only" instruction, we
        # log the failure instead of crashing the whole pipeline - same
        # "flag it, don't fail silently" principle as extract_ocr.py.
        parse_error = str(e)

    record["extraction_method"] = "vlm"
    record["raw_text_snippet"] = raw_response[:200]
    if parse_error:
        record["parse_error"] = parse_error

    return record


def extract_vlm_invoice(pdf_path: str) -> dict:
    """
    Thin wrapper so pipeline.py can call this with the same
    one-argument signature it already uses for extract_pdf_text /
    extract_ocr, while still getting the invoice-specific prompt
    instead of the claim-form one. This keeps the call sites in
    pipeline.py's process_document() consistent across all three
    extraction methods.
    """
    return extract_vlm(pdf_path, document_type="invoice")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python extract_vlm.py <path_to_pdf>")
        sys.exit(1)

    path = sys.argv[1]
    result = extract_vlm(path)
    print(json.dumps(result, indent=2))