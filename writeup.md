# Write-up: Healthcare Claims Document Extraction

## My approach

I built this around one idea: every document gets routed to whichever extraction method(s) fit its format, but every extracted record — no matter which method produced it — goes through the same two downstream stages: `normalize.py` cleans raw values into consistent types, and `validate.py` flags completeness and plausibility issues. Because of that split, adding the VLM as a third extraction method partway through didn't require touching normalization or validation at all — both already worked on any dict matching the shared schema.

`detect_format.py` checks whether `pdfplumber` can pull more than a small character threshold off the page. Below that threshold, I treat the PDF as image-based. I didn't just trust this heuristic — I checked it against the raw PDF byte structure directly (presence of `/Font` objects and text operators vs. a single embedded `/Image` with no text layer) for all six of my documents before relying on it, and it matched in every case.

Text-based invoices go to `extract_pdf_text.py`, which combines regex for header fields (Membership Number, Invoice No.) with text-anchored parsing for the line-items table. Image-based claim forms go to `extract_ocr.py` (Tesseract plus OpenCV preprocessing) and `extract_vlm.py` (Groq's Llama 4 Scout vision model). I ran all three invoices through the VLM too, so every document in the project has results from at least two methods — that gave me a real comparison across the whole dataset, not just on the documents that were already hard.

## The hardest challenge: handwriting, and knowing when to stop trying

The claim forms mix printed labels with handwritten values. Specialization, Treatment date, Final Diagnosis Description, and the entire Services/Items table (procedure names, billed amounts, claimed amounts) are all handwritten in pen. My first OCR pass on these fields produced near-total noise — "Bilateral Breast Cysts" came back as unreadable fragments, and the services table collapsed into nonsense since Tesseract had no concept that those numbers belonged in distinct columns.

I didn't want to assume this was just a preprocessing-tuning problem, so I tested it properly: four preprocessing strategies (raw grayscale, Otsu global thresholding, upscale 1.5x with adaptive Gaussian thresholding, denoise plus Otsu thresholding), run against the same document, comparing OCR output on the same handwritten lines each time. All four left the handwriting effectively unreadable. The best result I got, on "Obs / Gyne", was the string `"O\s | G\ ae"` — recognizable to a human who already knows the answer, but nothing a parser could safely pull a value from.

That led to a deliberate design call: `extract_ocr.py` doesn't try to regex a value out of garbled handwritten OCR text. It extracts what it can reliably get (printed fields like member ID and practitioner name) and explicitly lists everything else in `low_confidence_fields`, set to `null` instead of populated with a guess. This cost me extraction coverage on OCR, but it avoided the failure mode the case study explicitly warns against — silently inserting a wrong value into a claims record.

The VLM did substantially better on these same fields, but it brought a different, more subtle failure mode: confident, well-formatted, wrong answers, which I get into below.

## Method comparison

All figures below are from two full runs of `pipeline.py` against my six source documents (three Eden Care claim forms, three Nairobi Lifecare invoices).

### Accuracy on invoices (text-based)

`pdf_text` and `vlm` produced identical extracted values across all three invoices in both runs — same member ID, provider name, date, and claimed amount every time, including Deepak's invoice with a decimal claimed amount (9548.97). That's a useful data point on its own: the VLM's accuracy advantage isn't limited to messy documents. It gets the easy cases right too, it just costs more time and a paid API call to do it (see Speed, below).

I'll flag one limitation in both methods here rather than let it hide: `provider_name` on invoices isn't really "extracted" by `pdf_text` so much as inferred — there's no labelled provider field on these invoices, so my code checks for "nairobi lifecare" in the text and hardcodes the hospital name if found. That works because all three invoices are from the same hospital, but it's not a generalizable extraction. Also, `provider_id` and `procedure_code` came back empty across every method and every document in my dataset — none of the six source documents appear to contain those fields at all, so this isn't a case of either method failing; there was nothing there to extract.

### Accuracy on claim forms (image-based)

This is where the methods diverge sharply.

| Field | OCR result | VLM result |
|---|---|---|
| `member_id` | Correct on all three forms | Correct on all three forms |
| `provider_name` | Correct on all three forms, after fixing two regex bugs (see below) | Correct on all three forms |
| `claimed_amount` | `null` on all three forms, including Deepak's claim (9548.97), the most challenging value in the dataset due to a hand-drawn circle correction on the form | Correct on all three forms, including Deepak's decimal value |
| `diagnosis_code` | `null` — correctly identifies the ICD code field as blank on all three forms | `null` — also correctly identifies the field as blank on all three forms |
| `date_of_service` | `null`, with a low-confidence flag, on all three forms | Correct on Deepak's form in both runs. Wrong on Wanjiku's form in both runs, with two different wrong values across the two runs. Wrong on Sharma's form in both runs, with the same wrong value both times |

Every OCR claim-form record landed at a `confidence_score` of 0.0 across all three documents in both runs. That's not a clamping bug — it's an accurate reflection that every required field (`claimed_amount`, `date_of_service`) was genuinely missing. The VLM's claim-form confidence scores were 0.6 when no validation flag fired (Deepak, both runs) and 0.5 when the future-date flag fired (Sharma and one of Wanjiku's two runs).

### The VLM's failure mode is different in kind, not just degree

OCR's failures were obvious — garbled text nobody would mistake for a real value, and my pipeline correctly flagged these instead of guessing. The VLM's failures were the opposite: clean, plausible, well-typed JSON with a wrong value sitting inside it, and the model never flagged its own readings as uncertain. Specifically, across my two full pipeline runs:

- **Wanjiku's claim form** (actual handwritten date: `23-10-2025`). One run returned `"2025-11-23"` — the month misread, 10 read as 11, year correct. The other run returned `"2026-10-23"` — the year misread, 2025 read as 2026, month correct. Both are plausible misreadings of ambiguous handwritten digits, and both are wrong. Every other field the VLM extracted for this document — `claim_id`, `member_id`, `provider_name`, `claimed_amount` — was identical across both runs. Only the date varied.
- **Sharma's claim form** (actual date: `28-08-2025`). Both runs returned `"2026-08-28"` — the same year misread, consistently, both times.
- **Deepak's claim form** — correct in both runs.

So the instability I found isn't "the VLM is generally non-deterministic." It's narrower than that: two of three documents gave stable results across runs (one correct, one consistently wrong the same way), and one document's date field specifically gave two different wrong answers. That's a more useful finding than a blanket "non-deterministic" claim, because it suggests whatever is driving the instability is tied to something about that specific document or that specific field — not a general property of calling the model.

This matters because the VLM's output, taken at face value, is more dangerous in a real claims pipeline than OCR's obvious garbage. A human reviewer has far less reason to double-check a clean, well-formatted date than an obviously corrupted string. This is exactly why I built `validate.py` as an independent stage: it caught both of the future-dated errors (`2026-08-28` and `2026-10-23`) automatically, without needing to know what the "correct" value was — it only needed to know that a treatment date after the claim's filing date is logically impossible. It did not catch the `2025-11-23` error, since that date, while still wrong, isn't implausible on its face. That's a real, honest limit on what automated validation can do: it catches impossible values, not merely incorrect ones.

I also want to be precise about what `temperature=0` did and didn't buy me here. I set it expecting fully deterministic output, and got that for two of three claim forms — but not the third. That tells me `temperature=0` reduces variance at the API-provider level for vision-model inference, but it doesn't guarantee it, at least not for Groq's hosted Llama 4 Scout.

### Speed

| Method | Typical time per document |
|---|---|
| `pdf_text` | 0.02–0.04 seconds |
| `ocr` | 3.3–50 seconds (highly variable — see below) |
| `vlm` | 2.7–5.3 seconds for invoices; 3.5–64 seconds for claim forms, with high run-to-run variance on the same document |

The OCR timing spread is fully explained by source image resolution: Wanjiku's claim form has a much higher-resolution embedded scan than Deepak's or Sharma's, and Tesseract's runtime scales with pixel count — 41–50 seconds for Wanjiku across my two runs, versus 3–5 seconds for the other two.

The VLM's timing variance is a different story, and it's the part I find most interesting. Image size is standardized to `dpi=130` before sending to Groq regardless of the source PDF's native resolution, so the VLM doesn't show the same 10x spread tied to file size that OCR does. But it still varied a lot run to run on the exact same document with the exact same result: Sharma's claim form took 4.25 seconds in one run and 31.85 seconds in the other, returning the identical (wrong) date both times. Since the output didn't change but the latency did, this points squarely at fluctuating load on Groq's API rather than anything in my own code.

On invoices specifically, `pdf_text` was over 100x faster than the VLM for an identical result — a clear, quantified case for preferring deterministic text extraction whenever a usable text layer exists, and saving the VLM for documents where it actually earns its cost.

### What each method got wrong, summarized

- **pdf_text**: handled header fields and line-item totals reliably once I switched away from `extract_table()`. Its real limitation isn't a bug so much as a design shortcut — `provider_name` is inferred from a hospital-name string match rather than parsed from a labelled field, and it never attempts `provider_id` or `procedure_code` at all, since neither field appears anywhere in my invoice documents.
- **ocr**: correctly extracts printed fields, cannot extract handwritten fields at all, and I chose not to guess at them rather than risk a silent wrong value.
- **vlm**: extracts both printed and handwritten fields well, including amounts that defeated OCR entirely — but it's the only method that produced confident, clean, incorrect values, specifically on the handwritten date field, without surfacing any indication of uncertainty about that specific reading.

## Other bugs I found and fixed along the way

- **`pdfplumber.extract_table()` inconsistency.** Two of three invoices had their line-items table correctly detected; the third (Wanjiku's) did not, despite all three coming from the same hospital billing template. I fixed this by parsing line items from raw text, anchored on the literal "Service Description" and "Total" strings, instead of depending on table detection.
- **Regex failures from OCR noise characters.** `member_id` and `provider_name` extraction initially failed on two of three claim forms because OCR introduced stray underscore and smart-quote characters between a field's printed label and its value — e.g. `"Name _- Wanjiku Mwangi"`. The underscore case turned out to be a genuine regex gotcha: `\W` (non-word character) doesn't match `_`, because regex treats underscore as a word character. I fixed it by using `[\W_]` instead of `\W` alone.
- **Image size exceeding Groq's API limits.** Claim form images rendered at the default PDF-to-image DPI exceeded both Groq's 33-megapixel resolution cap and its 4MB base64 payload limit. I found and fixed this before making my first real API call, by testing several DPI values and settling on 130.

## What I'd do differently with more time

- **Run multiple VLM passes per document and reconcile disagreements.** Since I directly observed the VLM isn't fully deterministic on at least one document, a majority-vote or confidence-weighted approach across 2–3 calls per document could catch cases like Wanjiku's date, where different runs disagree with each other — a strong signal that a field deserves human review even without knowing the ground truth in advance.
- **Cross-validate `claimed_amount` against the invoice for the same patient, where both exist.** Deepak's claim form and his Nairobi Lifecare invoice both state 9548.97, and Sharma's and Wanjiku's claim/invoice pairs likewise agree on amount. A validation rule flagging disagreement between a claim form and its corresponding invoice (when both exist for the same `member_id`) would catch a class of errors neither document type alone can.
- **Handle password-protected or encrypted PDFs explicitly.** None of my six test documents were encrypted, so this never came up in testing, but it's realistic for a production pipeline — a provider or member could submit a password-protected PDF. `pdfplumber.open()` and `pdf2image`'s underlying Poppler call both accept a `password` parameter, but right now nothing in `detect_format.py` or any extractor checks for encryption up front, so an encrypted file would currently raise an unhandled exception and get lumped into `pipeline.py`'s generic `extraction_failed` flag — indistinguishable from any other crash. I'd add an explicit encryption check, then either try a small set of provider-supplied default passwords, prompt for one, or — more realistically for an unattended batch pipeline — flag the document distinctly as `requires_manual_review: password_protected`.
- **Build a small ground-truth labeled set**, even just my six documents hand-verified, to compute precision/recall per field per method numerically rather than the qualitative correct/incorrect comparison I've done here.
- **Add automatic retry-with-different-DPI logic** to `extract_ocr.py` and `extract_vlm.py` rather than a single fixed DPI, so future documents with very different native resolutions don't need manual re-tuning.
