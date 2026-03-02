from pathlib import Path
import json
import pdfplumber
import csv

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# PDF_PATH = PROJECT_ROOT / "data" / "raw" / "pm" / "Pharmacy_Policy_Guidelines.pdf"
# PARSED_JSON = PROJECT_ROOT / "data" / "raw" / "pm" / "parse_raw_json" / "Pharmacy_Policy_Guidelines.json"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pm"
PARSED_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "pm" / "parse_raw_json"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "pm" / "parse_csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Detect Pages That Likely Contain Tables
# ============================================================

def page_likely_has_table(text: str) -> bool:
    """
    Heuristic detection:
    - multiple lines with repeated spacing
    - multiple numeric columns
    - OR typical table keywords
    """
    if not text:
        return False

    keywords = ["table", "origin code", "field", "corresponding"]
    if any(k.lower() in text.lower() for k in keywords):
        return True

    lines = text.splitlines()
    multi_space_lines = [l for l in lines if "  " in l]

    if len(multi_space_lines) > 5:
        return True

    return False


def get_pages_with_tables(parsed_json_path: Path):
    pages = []

    data = json.loads(parsed_json_path.read_text(encoding="utf-8"))

    for page in data.get("pages", []):
        page_no = page.get("page")
        text = page.get("text", "")

        if page_likely_has_table(text):
            pages.append(page_no)

    return pages


# ============================================================
# Extract Tables From Original PDF
# ============================================================

def extract_tables_from_pdf(pdf_path: Path, page_numbers: list[int], policy_out_dir: Path):
    with pdfplumber.open(pdf_path) as pdf:

        for page_no in page_numbers:
            if page_no - 1 >= len(pdf.pages):
                continue

            page = pdf.pages[page_no - 1]
            tables = page.extract_tables()

            if not tables:
                continue

            for idx, table in enumerate(tables, start=1):

                cleaned_table = [
                    [cell.strip() if cell else "" for cell in row]
                    for row in table
                ]

                out_path = policy_out_dir / f"{pdf_path.stem}_page_{page_no}_table_{idx}.csv"

                with out_path.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerows(cleaned_table)

                print(f"Saved: {out_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    if not PDF_DIR.exists():
        print("PDF directory not found.")
        return

    if not PARSED_JSON_DIR.exists():
        print("Parsed JSON directory not found.")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found.")
        return

    for pdf_path in pdf_files:

        policy_name = pdf_path.stem
        parsed_json_path = PARSED_JSON_DIR / f"{policy_name}.json"

        if not parsed_json_path.exists():
            print(f"Skipping {policy_name}: no matching parsed JSON found.")
            continue

        print(f"\nProcessing policy: {policy_name}")

        # Create subfolder for this policy
        policy_out_dir = OUT_DIR / policy_name
        policy_out_dir.mkdir(parents=True, exist_ok=True)

        print("Detecting pages with potential tables...")
        pages = get_pages_with_tables(parsed_json_path)

        if not pages:
            print("No likely table pages detected.")
            continue

        print(f"Pages flagged for extraction: {pages}")

        print("Extracting tables from original PDF...")
        extract_tables_from_pdf(pdf_path, pages, policy_out_dir)

    print("\nAll policies processed.")


if __name__ == "__main__":
    main()