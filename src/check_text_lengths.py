from pathlib import Path
import pdfplumber


def get_raw_text(pdf_path: str) -> str:
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)

    return "\n".join(all_text)


def check_folder(folder):
    print(f"\nChecking: {folder}\n")

    for pdf in sorted(Path(folder).glob("*.pdf")):
        text = get_raw_text(str(pdf))
        stripped = len(text.strip())

        print(f"{pdf.name}")
        print(f"Characters: {stripped}")

        # Show exactly what text was extracted
        preview = repr(text.strip()[:120])
        print(f"Preview: {preview}")
        print("-" * 60)


check_folder("data/claim_forms")
check_folder("data/invoices")