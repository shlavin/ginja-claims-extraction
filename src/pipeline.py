"""
pipeline.py

The main entry point for this project. Run this file to process every
PDF in data/claim_forms/ and data/invoices/, producing one structured,
validated JSON record per (document, extraction method) pair, written
to output/results.json.

Routing logic:
  - Text-based PDFs (invoices): extract_pdf_text.py AND extract_vlm.py
    (invoice-aware prompt) are both run.
  - Image-based PDFs (claim forms): extract_ocr.py AND extract_vlm.py
    (claim-form-aware prompt) are both run.
  So every document in the project ends up with results from at least
  two methods, and the VLM result exists for every single document -
  that gives a genuine three-way comparison across the whole dataset,
  not just on the harder claim forms. (OCR doesn't run on invoices:
  there's no real ambiguity about the best traditional method for a
  clean, typed PDF that already has a real text layer - pdf_text
  already handles that case correctly, and OCR-ing a PDF that's
  already text-based would just add noise for no benefit, since OCR
  is recovering text from pixels that's already sitting there
  natively in the PDF.)

Every record goes through the same three stages no matter which
extraction method produced it:
  raw extraction -> normalize_record() -> validate_record()
Then, once the whole batch is done, check_duplicate_claim_ids() runs
a single pass over everything and merges any duplicate flags back in.

KNOWN ASYMMETRY (documented on purpose, see writeup.md):
  claim_id is genuinely absent on the Nairobi Lifecare invoices -
  these are hospital billing documents, not Eden Care claims-system
  documents, so there's no claim_id field on them to extract in the
  first place. The Eden Care claim forms DO have an equivalent (the
  printed "Visit ID"), which extract_vlm.py maps to claim_id. We
  deliberately did NOT substitute invoice_number as a stand-in
  claim_id for invoices - the case study's own field table treats
  claim_id and invoice_number as distinct, separately-meaningful
  fields, so inventing a value where none exists would go against the
  explicit instruction to "record what is missing rather than
  failing" with a guess.
"""

import os
import json
import time
import traceback

from detect_format import detect_format
from extract_pdf_text import extract_pdf_text
from extract_ocr import extract_ocr
from extract_vlm import extract_vlm, extract_vlm_invoice
from normalize import normalize_record
from validate import validate_record, check_duplicate_claim_ids

CLAIM_FORMS_DIR = os.path.join("data", "claim_forms")
INVOICES_DIR = os.path.join("data", "invoices")
OUTPUT_PATH = os.path.join("output", "results.json")

SCHEMA_FIELDS = [
    "claim_id", "member_id", "provider_name", "provider_id",
    "date_of_service", "diagnosis_code", "procedure_code",
    "claimed_amount", "currency", "invoice_number",
]


def compute_confidence_score(record: dict, validation_flags: list[str]) -> float:
    """
    A simple, explainable confidence score: the fraction of schema
    fields that came back successfully extracted (not None), minus a
    penalty for each validation flag raised.

    Kept intentionally simple rather than building some learned or
    weighted model - the case study asks for confidence_score to
    "reflect how confident you are in the extraction," and at this
    project's scale, a transparent formula anyone can check by hand
    is more trustworthy than an opaque score would be.
    """
    filled = sum(1 for field in SCHEMA_FIELDS if record.get(field) is not None)
    completeness = filled / len(SCHEMA_FIELDS)

    penalty = 0.1 * len(validation_flags)
    score = max(0.0, completeness - penalty)
    # Note: on this project's actual OCR results, completeness and penalty
    # come out almost exactly equal (0.3 vs 0.3 for a record missing all 3
    # required fields), which lands on a 0.0 score. I checked this
    # deliberately against the real pipeline output - it's the formula
    # working as intended, not a clamping artifact: OCR genuinely failed
    # every required field on these documents, so a 0.0 "do not trust
    # without review" signal is the correct outcome here.
    return round(score, 2)


def process_document(filepath: str, extraction_fn, method_name: str) -> dict:
    """
    Runs one extraction method against one document, then normalizes
    and validates whatever comes back. Wrapped in a try/except so one
    failing document/method pairing can't take down the whole batch -
    a failure gets recorded as its own flagged result instead, in
    line with the case study's "do not silently fail" principle.
    """
    source_file = os.path.basename(filepath)
    start = time.time()

    try:
        raw_record = extraction_fn(filepath)
        normalized = normalize_record(raw_record)
        flags = validate_record(normalized)
    except Exception as e:
        # Record the failure itself as a result, rather than letting
        # one bad document/method pairing crash the whole pipeline run.
        return {
            "source_file": source_file,
            "extraction_method": method_name,
            "fields_extracted": {},
            "fields_missing": SCHEMA_FIELDS,
            "validation_flags": [f"extraction_failed: {e}"],
            "confidence_score": 0.0,
            "raw_text_snippet": None,
            "processing_time_seconds": round(time.time() - start, 2),
            "error_traceback": traceback.format_exc(limit=3),
        }

    fields_extracted = {k: v for k, v in normalized.items() if k in SCHEMA_FIELDS and v is not None}
    fields_missing = [k for k in SCHEMA_FIELDS if normalized.get(k) is None]

    result = {
        "source_file": source_file,
        "extraction_method": method_name,
        "claim_id": normalized.get("claim_id"),  # kept at the top level so duplicate-checking can get to it easily
        "fields_extracted": fields_extracted,
        "fields_missing": fields_missing,
        "validation_flags": flags,
        "confidence_score": compute_confidence_score(normalized, flags),
        "raw_text_snippet": raw_record.get("raw_text_snippet"),
        "processing_time_seconds": round(time.time() - start, 2),
    }

    # Carry over method-specific extras if they're present, but don't fail if they're not
    if "low_confidence_fields" in raw_record:
        result["low_confidence_fields"] = raw_record["low_confidence_fields"]
    if "parse_error" in raw_record:
        result["parse_error"] = raw_record["parse_error"]

    return result


def list_pdfs(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return [
        os.path.join(directory, f)
        for f in sorted(os.listdir(directory))
        if f.lower().endswith(".pdf")
    ]


def run_pipeline() -> list[dict]:
    results = []

    # --- Invoices: text-based. Run pdf_text AND vlm (invoice-aware prompt) ---
    for filepath in list_pdfs(INVOICES_DIR):
        detected = detect_format(filepath)
        if detected != "text_based":
            print(f"WARNING: {filepath} in invoices/ was detected as {detected}, "
                  f"not text_based as expected. Running pdf_text anyway, "
                  f"but flag this for review.")

        for extraction_fn, method_name in [(extract_pdf_text, "pdf_text"), (extract_vlm_invoice, "vlm")]:
            result = process_document(filepath, extraction_fn, method_name)
            results.append(result)
            print(f"Processed {os.path.basename(filepath)} [{method_name}] "
                  f"-> confidence {result['confidence_score']}")

    # --- Claim forms: image-based, run BOTH ocr and vlm for comparison ---
    for filepath in list_pdfs(CLAIM_FORMS_DIR):
        detected = detect_format(filepath)
        if detected != "image_based":
            print(f"WARNING: {filepath} in claim_forms/ was detected as {detected}, "
                  f"not image_based as expected. Running ocr+vlm anyway.")

        for extraction_fn, method_name in [(extract_ocr, "ocr"), (extract_vlm, "vlm")]:
            result = process_document(filepath, extraction_fn, method_name)
            results.append(result)
            print(f"Processed {os.path.basename(filepath)} [{method_name}] "
                  f"-> confidence {result['confidence_score']}")

    # --- Duplicate claim_id check across the whole batch ---
    duplicate_flags = check_duplicate_claim_ids(results)
    for result in results:
        claim_id = result.get("claim_id")
        if claim_id in duplicate_flags:
            result["validation_flags"] = result["validation_flags"] + duplicate_flags[claim_id]

    return results


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    results = run_pipeline()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nWrote {len(results)} records to {OUTPUT_PATH}")


def _self_test_duplicate_detection():
    """
    Our real dataset (3 claim forms, 3 invoices, all with distinct
    claim_ids) never actually triggers check_duplicate_claim_ids()
    with a true positive - every claim_id in this project happens to
    be unique. So this function exists to run directly and confirm
    the duplicate-detection logic itself actually works, using
    synthetic data with a deliberately repeated claim_id. It's here so
    there's a record that the check was verified, not just written
    and assumed to work.
    """
    synthetic_results = [
        {"source_file": "a.pdf", "extraction_method": "pdf_text", "claim_id": "20011941", "validation_flags": []},
        {"source_file": "b.pdf", "extraction_method": "ocr", "claim_id": "20011942", "validation_flags": []},
        {"source_file": "c.pdf", "extraction_method": "vlm", "claim_id": "20011941", "validation_flags": []},
    ]
    duplicate_flags = check_duplicate_claim_ids(synthetic_results)
    assert "20011941" in duplicate_flags, "Expected claim_id 20011941 to be flagged as duplicate"
    assert "20011942" not in duplicate_flags, "20011942 is unique and should NOT be flagged"
    print("Self-test passed: duplicate claim_id '20011941' correctly detected,")
    print(f"  flag message: {duplicate_flags['20011941']}")
    print("  unique claim_id '20011942' correctly NOT flagged.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test_duplicate_detection()
    else:
        main()