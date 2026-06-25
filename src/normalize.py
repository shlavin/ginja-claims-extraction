
"""
normalize.py

Takes the raw output from any extraction method (PDF text, OCR, or VLM)
and converts values into a consistent format before validation.

The purpose of this module is only to standardize the data. It does not
decide whether a value is correct or reasonable—that is handled later by
validate.py. For example, different date formats are all converted into
the same ISO format, regardless of which extraction method produced
them.

Keeping normalization separate from extraction makes the pipeline easier
to maintain. Each extractor only needs to produce the expected field
names, while this file handles differences in formatting.
"""

import re
from dateutil import parser as date_parser


def normalize_date(raw_value) -> str | None:
    """
    Converts a date from the various formats found in the project into
    the ISO format (YYYY-MM-DD).

    If the value is missing or cannot be parsed, None is returned
    instead of trying to guess the intended date.

    dayfirst=True is used because the claim documents processed in this
    project follow the common East African date format (DD-MM-YYYY).
    This avoids dates being interpreted incorrectly using the US
    month-first convention.
    """
    if not raw_value or not str(raw_value).strip():
        return None

    try:
        parsed = date_parser.parse(str(raw_value).strip(), dayfirst=True)
        return parsed.date().isoformat()
    except (date_parser.ParserError, ValueError, OverflowError):
        return None


def normalize_amount(raw_value) -> float | None:
    """
    Converts different amount formats, such as "4,300", "KES 9548.97",
    or numeric values returned by the VLM, into a float.

    Returns None if no valid numeric value can be extracted.
    """
    if raw_value is None:
        return None

    # Values that are already numeric only need to be converted to float.
    if isinstance(raw_value, (int, float)):
        return float(raw_value)

    # Remove currency symbols and other non-numeric characters while
    # keeping digits and decimal points.
    cleaned = re.sub(r"[^\d.]", "", str(raw_value))
    if not cleaned:
        return None

    try:
        return float(cleaned)
    except ValueError:
        return None


def normalize_text(raw_value) -> str | None:
    """
    Performs basic cleanup on text fields by removing leading and
    trailing whitespace and replacing repeated spaces with a single one.

    The original capitalization is left unchanged since altering names
    or codes could introduce unnecessary errors.
    """
    if raw_value is None:
        return None

    text = str(raw_value).strip()
    text = re.sub(r"\s+", " ", text)
    return text if text else None


def normalize_record(record: dict) -> dict:
    """
    Applies the appropriate normalization step to each field in a
    record and returns a cleaned copy.

    Fields that do not require normalization are copied over without
    modification.
    """
    normalized = dict(record)  # Create a copy so the original record is unchanged.

    normalized["date_of_service"] = normalize_date(record.get("date_of_service"))
    normalized["claimed_amount"] = normalize_amount(record.get("claimed_amount"))

    for text_field in ["member_id", "provider_name", "provider_id",
                        "claim_id", "diagnosis_code", "procedure_code",
                        "invoice_number", "currency"]:
        if text_field in record:
            normalized[text_field] = normalize_text(record.get(text_field))

    return normalized


if __name__ == "__main__":
    # Simple example using values from the project to verify that the
    # normalization functions produce the expected output.
    import json

    sample_record = {
        "claim_id": " 20011942 ",
        "member_id": "1536500",
        "provider_name": "Wanjiku Mwangi",
        "provider_id": None,
        "date_of_service": "28-08-2026",
        "diagnosis_code": None,
        "procedure_code": None,
        "claimed_amount": "17,000",
        "currency": "KES",
        "invoice_number": None,
        "extraction_method": "vlm",
    }

    print(json.dumps(normalize_record(sample_record), indent=2))

