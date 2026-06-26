# Ginja AI Case Study — Healthcare Claims Document Extraction

A pipeline that extracts structured data from healthcare claim documents (Eden Care claim forms and Nairobi Lifecare invoices) using three different extraction methods — direct PDF text parsing, OCR, and a Vision Language Model — then normalizes and validates the results.

## What this project does

Given a folder of claim forms (scanned, handwritten) and invoices (clean, typed PDFs), the pipeline:

1. Detects whether each PDF has a real text layer or is image-only
2. Routes it to the appropriate extraction method(s)
3. Normalizes raw extracted values into consistent types (ISO dates, float amounts)
4. Validates the result for completeness and plausibility
5. Writes one structured record per (document, method) pair to `output/results.json`

Every image-based claim form and every text-based invoice is run through **both** of its applicable traditional method **and** the VLM, so every document in the dataset has a result from multiple methods — supporting a genuine side-by-side method comparison rather than one limited to only the hardest documents.

## Project structure

```
ginja-claims-extraction/
├── data/
│   ├── claim_forms/        # 3 Eden Care PDFs (scanned, handwritten)
│   └── invoices/            # 3 Nairobi Lifecare PDFs (clean, typed)
├── src/
│   ├── detect_format.py     # text-based vs image-based detection
│   ├── extract_pdf_text.py  # text-layer extraction (invoices)
│   ├── extract_ocr.py       # Tesseract OCR (claim forms)
│   ├── extract_vlm.py       # Groq vision model (both document types)
│   ├── normalize.py         # type/format cleanup
│   ├── validate.py          # completeness + plausibility checks
│   └── pipeline.py          # wires everything together
├── output/
│   └── results.json         # final structured output
├── README.md
└── writeup.md
└── requirements.txt
└── .gitignore
```

## Setup

### 1. Python environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows PowerShell
# or: source venv/Scripts/activate   # Git Bash on Windows
```

### 2. Install dependencies

```bash
pip install pdfplumber pymupdf pytesseract pillow opencv-python pandas python-dateutil groq python-dotenv pdf2image
```

### 3. Install Tesseract OCR (the actual engine, not just the Python wrapper)

**Windows:**
```powershell
winget install -e --id UB-Mannheim.TesseractOCR
```
Or download the installer from the [UB-Mannheim Tesseract wiki](https://github.com/UB-Mannheim/tesseract/wiki).

If `pytesseract` can't find it automatically, set the path explicitly at the top of `extract_ocr.py`:
```python
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
```

### 4. Install Poppler (required by `pdf2image` to rasterize PDF pages)

Download from the [oschwartz10612 Windows build](https://github.com/oschwartz10612/poppler-windows/releases), extract it somewhere permanent (e.g. `C:\poppler-26.02.0`), and either add the `Library\bin` folder to PATH or set `POPPLER_PATH` directly in `extract_ocr.py` / `extract_vlm.py`.

Verify both are working:
```bash
tesseract --version
pdftoppm -v
```

### 5. Set up Groq (for VLM extraction)

1. Sign up free at [console.groq.com](https://console.groq.com) (no card required)
2. Create an API key under **API Keys**
3. Create a `.env` file in the project root (already gitignored):
```
GROQ_API_KEY=gsk_your_key_here
```

Model used: `meta-llama/llama-4-scout-17b-16e-instruct`, one of two vision-capable models currently available on Groq (the other being `qwen/qwen3.6-27b`). Check [console.groq.com/docs/vision](https://console.groq.com/docs/vision) for the current list before re-running this — Groq's supported models change over time.

## Running the pipeline

```bash
python src/pipeline.py
```

This processes every PDF in `data/claim_forms/` and `data/invoices/` and writes `output/results.json`. Console output shows per-document, per-method progress and confidence scores as it runs. The file is fully overwritten (not appended) on each run, so there's no need to manually delete it beforehand.

To verify the duplicate-claim-id detection logic on its own (our real dataset has no duplicate claim_ids, so this never naturally triggers):
```bash
python src/pipeline.py --self-test
```

To run any individual extraction method against a single document (useful for debugging):
```bash
python src/detect_format.py data/claim_forms/claim_form_Wanjiku_Kamua.pdf
python src/extract_pdf_text.py data/invoices/Nairobi_Lifecare_Hospital_Invoice_Wanjiku_Kamau.pdf
python src/extract_ocr.py data/claim_forms/claim_form_Wanjiku_Kamua.pdf
python src/extract_vlm.py data/claim_forms/claim_form_Wanjiku_Kamua.pdf
```

## Test documents used

| File | Format | Notes |
|---|---|---|
| `claim_form_Wanjiku_Kamua.pdf` | Image-based (scanned, handwritten) | Eden Care claim form |
| `Sharma_Siddharth_Claim_Form.pdf` | Image-based (scanned, handwritten) | Eden Care claim form |
| `Deepak_Bodkhe_Claim_Form.pdf` | Image-based (scanned, handwritten) | Eden Care claim form |
| `Nairobi_Lifecare_Hospital_Invoice_Wanjiku_Kamau.pdf` | Text-based (typed) | Hospital invoice |
| `Nairobi_Lifecare_Hospital_Invoice_Sharma_Siddharth.pdf` | Text-based (typed) | Hospital invoice |
| `Nairobi_Lifecare_Hospital_Invoice_Deepak_Bodkhe.pdf` | Text-based (typed) | Hospital invoice |

Format was confirmed by direct inspection of the underlying PDF structure (presence/absence of `/Font` objects and text operators vs. a single embedded `/Image` object with no text layer), not just by `detect_format.py`'s output — both agreed in all 6 cases.

## Why these libraries

- **pdfplumber** for text extraction — handles both raw text and (in principle) tables. In practice, we found its table detection unreliable across documents that looked visually identical (see Known Limitations), so line-item parsing falls back to text-anchored regex rather than `extract_table()`.
- **pytesseract / Tesseract** — free, open-source, the most widely supported OCR engine with a mature Python wrapper.
- **OpenCV** for preprocessing — denoising and Otsu thresholding, chosen after testing four preprocessing variants against real documents (see writeup.md).
- **pdf2image / Poppler** — the standard way to rasterize a PDF page into an image for OCR/VLM input, since neither Tesseract nor Groq's vision API accept PDFs directly.
- **Groq + Llama 4 Scout** — free vision-language model access, with JSON mode support that simplifies parsing structured output.
- **dateutil** — handles the wide variety of date formats we encountered ("12 Feb 2026", "23-11-2025", "08 - 08 - 2025") without hand-rolled regex for each.

## Known limitations and edge cases

**Shell-dependent PATH resolution on Windows.** `extract_ocr.py` and `extract_vlm.py` depend on Poppler being resolvable via `pdf2image`. In testing, the exact same script succeeded when run from Git Bash (MINGW64) but failed in PowerShell with `"Unable to get page count. Is poppler installed and in PATH?"` — even though Poppler was correctly installed. Windows does not always propagate PATH updates consistently across PowerShell, Command Prompt, and Git Bash sessions opened at different times. **Run this project from Git Bash on Windows**, or set `POPPLER_PATH` explicitly at the top of `extract_ocr.py` / `extract_vlm.py` if running from PowerShell.

**`pdfplumber`'s table detection is inconsistent across visually-identical documents.** Two of our three invoices (Sharma's, Deepak's) had their line-items table correctly detected by `extract_table()`. The third (Wanjiku's) did not — `extract_table()` only found the header block, not the services table, even though both PDFs were generated by the same hospital billing system and look identical to a human. We fixed this by parsing line items directly from raw text, anchored on the literal "Service Description" and "Total" strings, rather than relying on table detection at all.

**OCR cannot reliably read handwritten fields.** We tested four preprocessing strategies (raw grayscale, Otsu thresholding, upscale + adaptive thresholding, denoise + Otsu) against the same claim form. All four left handwritten fields (Specialization, Treatment date, Final Diagnosis Description, the entire Services/Items table) unreadable or only partially readable. Printed/typed fields (Membership number, Surname, Practitioner Name) OCR'd reliably across all three claim forms after denoise+Otsu preprocessing. Rather than guess at garbled handwriting, `extract_ocr.py` explicitly flags these fields as `low_confidence_fields` and leaves them `null`.

**The VLM is not perfectly accurate, and is not fully deterministic even at `temperature=0`.** Across two separate full pipeline runs, the VLM's `date_of_service` reading for the same document (`claim_form_Wanjiku_Kamua.pdf`) returned two different wrong values (`"23-11-2025"` in one run, `"23-10-2026"` in another; the actual handwritten date is `23-10-2025`). The VLM's errors were plausible misreadings of ambiguous handwritten digits (10 vs 11, 2025 vs 2026), not random noise — but they were also clean, well-formatted, and not flagged by the model itself as low-confidence, unlike OCR's obviously garbled output. See writeup.md for the full discussion of why this matters.

**`claim_id` is genuinely absent on invoices.** Nairobi Lifecare invoices are hospital billing documents, not Eden Care claims-system documents, so they have no `claim_id` field to extract — only the Eden Care claim forms have an equivalent (`Visit ID`, which `extract_vlm.py` maps to `claim_id`). We deliberately did not substitute `invoice_number` as a stand-in `claim_id` for invoices, since the case study's field table treats them as distinct fields with different meanings; inventing a value would contradict the instruction to record what's missing rather than guess. This means every invoice record is flagged with `"required field 'claim_id' is missing"` and scores a correspondingly lower `confidence_score` — this is intentional, not a bug.

**Groq's image size limits required tuning.** Groq enforces a 33-megapixel resolution limit and a 4MB limit on base64-encoded image payloads. Our claim form PDFs, rendered at a typical conversion DPI, exceeded both (58 megapixels, 6MB+ encoded). We settled on `dpi=130` after testing several values — it stays comfortably under both caps across all three claim forms while remaining legible.

**Large embedded images significantly slow down OCR.** One claim form's processing time was over 10x longer than the other two (50s vs ~5s) purely because its source PDF had a higher-resolution embedded image. VLM processing time was comparatively stable across documents since image size is standardized before sending to Groq.

**Duplicate claim_id detection is untested against a true positive in the real dataset**, since all 6 source documents have unique claim_ids. `pipeline.py --self-test` verifies the detection logic works correctly using synthetic data with a deliberately repeated claim_id.

**Validation checks plausibility, not correctness.** `validate.py`'s future-date check successfully caught the VLM's `2026-08-28` and `2026-10-23` errors (both genuinely impossible — in the future relative to the claim filing date). It did *not* catch the VLM's `2025-11-23` error in an earlier run, because that date, while factually wrong, is not implausible on its face. This is a deliberate scope limitation: automated validation can catch values that are logically impossible, not values that are merely incorrect but plausible-looking.

**Provider_id and procedure_code have no dedicated extraction logic in extract_pdf_text.py or extract_ocr.py.** Both fields are defined in the shared schema and requested in the VLM prompt, but on all six of our source documents, the corresponding form fields (RMDC Reg No, RMPC Procedure Code) are genuinely left blank by the practitioner — so every method correctly returns null for them, and we never had a real filled-in example to write or test a regex pattern against. If a future document did have these fields filled in, the VLM would likely still extract them correctly (since the prompt already asks for them), but extract_pdf_text.py and extract_ocr.py would currently miss them even with a perfect OCR/text read, since no pattern-matching rule exists yet for either field.