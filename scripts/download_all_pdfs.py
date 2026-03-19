"""Download all PDFs from metadata_filled.csv source_url, organized by category."""
import csv
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META_FILE = ROOT / "data" / "metadata" / "metadata_filled.csv"
RAW_DIR = ROOT / "data" / "raw"

# Category -> subdirectory mapping
CATEGORY_DIR = {
    "Medicaid Update": "medicaid_update",
    "Children Wavier": "children_waiver",
    "Pharmacy": "pharmacy",
    "Pharmacy Billing": "pharmacy_billing",
}


def main():
    with open(META_FILE, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"[INFO] {len(rows)} documents in metadata")

    for row in rows:
        cat = row.get("category", "unknown")
        subdir = CATEGORY_DIR.get(cat, cat.lower().replace(" ", "_"))
        out_dir = RAW_DIR / subdir
        out_dir.mkdir(parents=True, exist_ok=True)

        file_name = row["file_name"]
        url = row["source_url"]
        dest = out_dir / file_name

        if dest.exists():
            print(f"[SKIP] {file_name} (already exists)")
            continue

        print(f"[DL]  {file_name} <- {url}")
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })
            with urllib.request.urlopen(req) as resp, open(dest, "wb") as out_f:
                out_f.write(resp.read())
        except Exception as e:
            print(f"[ERR] {file_name}: {e}")

    print("[DONE] All downloads complete")


if __name__ == "__main__":
    main()
