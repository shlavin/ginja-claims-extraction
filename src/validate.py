
"""
validate.py

Runs validation on records after they have been normalized.

Instead of fixing or removing bad data, this module records anything
that looks suspicious or incomplete. The idea is to keep every record
and attach validation flags so that a person or another system can
review them later if needed.

The checks performed here include:
  - Required fields are present
  - Claimed amounts are reasonable
  - Dates are valid and not in the future
  - Diagnosis codes follow the expected format
  - Duplicate claim IDs are detected within a batch
"""

import re
from datetime import date, datetime

REQUIRED_FIELDS = ["claim_id", "member_id", "claimed_amount", "date_of_service"]

# Claims in this dataset are normally much lower than this (roughly
# between 800 and 17,000 KES). The limit is intentionally generous so
# that only obvious mistakes, such as an extra zero being entered, are
# flagged while still allowing unusually expensive claims through.
MAX_PLAUSIBLE_AMOUNT = 1_000_000  # KES

# Basic pattern for ICD-10 diagnosis codes such as "J18.9" or "K29.0".
# It only checks the general structure rather than validating every
# possible ICD-10 code, which helps avoid rejecting valid codes that
# are outside the sample used during testing.
ICD10_PATTERN = re.compile(r"^[A-Z]\d{2}(\.\d+)?$")


def check_required_fields(record: dict) -> list[str]:
    flags = []
    for field in REQUIRED_FIELDS:
        if record.get(field) is None:
            flags.append(f"required field '{field}' is missing")
    return flags


def check_amount(record: dict) -> list[str]:
    flags = []
    amount = record.get("claimed_amount")

    if amount is None:
        return flags  # already flagged by check_required_fields

    if amount <= 0:
        flags.append(f"claimed_amount is not positive: {amount}")
    elif amount > MAX_PLAUSIBLE_AMOUNT:
        flags.append(f"claimed_amount exceeds plausible range: {amount}")

    return flags


def check_date(record: dict) -> list[str]:
    """
    Verifies that date_of_service is a valid ISO date and that it is
    not later than today's date.

    This check was added after noticing that one extracted claim had a
    service date in 2026 even though the supporting documents were from
    2025. Rather than trying to guess the correct value, the validator
    simply reports the inconsistency.
    """
    flags = []
    raw_date = record.get("date_of_service")

    if raw_date is None:
        return flags  # already flagged by check_required_fields

    try:
        parsed_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        flags.append(f"date_of_service is not a valid ISO date: {raw_date!r}")
        return flags

    if parsed_date > date.today():
        flags.append(f"date_of_service is in the future: {raw_date}")

    return flags


def check_codes(record: dict) -> list[str]:
    """
    Checks the format of diagnosis_code when one is available.

    Missing diagnosis codes are not treated as validation errors here
    because many claim forms in this dataset leave that field blank.
    The check only flags codes that are present but do not resemble an
    ICD-10 code.
    """
    flags = []
    diagnosis = record.get("diagnosis_code")

    if diagnosis is not None and not ICD10_PATTERN.match(diagnosis):
        flags.append(f"diagnosis_code does not match expected ICD-10 shape: {diagnosis!r}")

    return flags


def check_duplicate_claim_ids(records: list[dict]) -> dict[str, list[str]]:
    """
    Looks through the entire batch and identifies claim IDs that appear
    more than once.

    Since duplicate detection depends on comparing records with one
    another, it is handled separately from the single-record validation
    functions.
    """
    seen = {}
    duplicate_flags = {}

    for record in records:
        claim_id = record.get("claim_id")
        if claim_id is None:
            continue
        seen.setdefault(claim_id, 0)
        seen[claim_id] += 1

    for claim_id, count in seen.items():
        if count > 1:
            duplicate_flags[claim_id] = [f"claim_id '{claim_id}' appears {count} times in this batch"]

    return duplicate_flags


def validate_record(record: dict) -> list[str]:
    """
    Runs every validation check that only requires a single record and
    combines the resulting flags into one list.

    Duplicate claim ID checking is not included here because it needs
    access to the full batch of records.
    """
    flags = []
    flags += check_required_fields(record)
    flags += check_amount(record)
    flags += check_date(record)
    flags += check_codes(record)
    return flags


if __name__ == "__main__":
    import json

    # Example record taken from the project after normalization. The
    # service date is intentionally left as the extracted value so the
    # validator demonstrates how a future-date issue is flagged.
    sharma_vlm_normalized = {
        "claim_id": "20011942",
        "member_id": "1536500",
        "provider_name": "Wanjiku Mwangi",
        "provider_id": None,
        "date_of_service": "2026-08-28",
        "diagnosis_code": None,
        "procedure_code": None,
        "claimed_amount": 17000.0,
        "currency": "KES",
        "invoice_number": None,
    }

    flags = validate_record(sharma_vlm_normalized)
    print(json.dumps({"record": sharma_vlm_normalized, "validation_flags": flags}, indent=2))

