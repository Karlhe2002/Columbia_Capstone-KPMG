"""Batch parse + chunk with title checks every 10 files."""
import sys, json, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from healthcare_rag_llm.doc_parsing import parse_file
from healthcare_rag_llm.chunking.section_semantic_chunking import section_semantic_chunking

ROOT = Path(__file__).resolve().parents[1]

BATCH_GROUPS = [
    {
        "name": "Medicaid Updates",
        "raw_dir": ROOT / "data" / "raw" / "medicaid_update",
        "out_dir": ROOT / "data" / "processed" / "medicaid_update",
        "chunk_dir": ROOT / "data" / "chunks" / "section_semantic_chunking_result",
    },
    {
        "name": "Children Waiver",
        "raw_dir": ROOT / "data" / "raw" / "children_waiver",
        "out_dir": ROOT / "data" / "processed" / "children_waiver",
        "chunk_dir": ROOT / "data" / "chunks" / "section_semantic_chunking_result",
    },
    {
        "name": "Pharmacy",
        "raw_dir": ROOT / "data" / "raw" / "pharmacy",
        "out_dir": ROOT / "data" / "processed" / "pharmacy",
        "chunk_dir": ROOT / "data" / "chunks" / "section_semantic_chunking_result",
    },
    {
        "name": "Pharmacy Billing",
        "raw_dir": ROOT / "data" / "raw" / "pharmacy_billing",
        "out_dir": ROOT / "data" / "processed" / "pharmacy_billing",
        "chunk_dir": ROOT / "data" / "chunks" / "section_semantic_chunking_result",
    },
]

BATCH_SIZE = 10

# Files that hang during parsing (stuck in pdfium rendering)
SKIP_FILES = set()


def check_titles(chunk_dir, jsonl_files):
    """Print section titles from recently chunked files for review."""
    print("\n" + "=" * 60)
    print("TITLE CHECK")
    print("=" * 60)
    for jf in jsonl_files:
        path = chunk_dir / jf
        if not path.exists():
            continue
        with open(path) as f:
            chunks = [json.loads(l) for l in f]
        seen = []
        for c in chunks:
            t = c.get("section_title", "")
            if t and t not in seen:
                seen.append(t)
        print(f"\n  {jf} ({len(chunks)} chunks, {len(seen)} sections):")
        for t in seen[:8]:
            print(f"    - {t[:80]}")
        if len(seen) > 8:
            print(f"    ... and {len(seen) - 8} more")
    print("=" * 60)


def main():
    for group in BATCH_GROUPS:
        name = group["name"]
        raw_dir = group["raw_dir"]
        out_dir = group["out_dir"]
        chunk_dir = group["chunk_dir"]

        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_dir.mkdir(parents=True, exist_ok=True)

        pdfs = sorted(raw_dir.glob("*.pdf"))
        if not pdfs:
            print(f"\n[SKIP] {name}: no PDFs found in {raw_dir}")
            continue

        print(f"\n{'#' * 60}")
        print(f"# {name}: {len(pdfs)} files")
        print(f"{'#' * 60}")

        batch_jsonl = []

        for i, pdf in enumerate(pdfs):
            if pdf.name in SKIP_FILES:
                print(f"[SKIP] {pdf.name} (in skip list - hangs during parse)")
                continue

            # Step 1: Parse
            json_out = out_dir / (pdf.stem + ".json")
            if json_out.exists():
                print(f"[SKIP PARSE] {pdf.name} (already parsed)")
            else:
                print(f"[PARSE {i+1}/{len(pdfs)}] {pdf.name}")
                try:
                    import signal

                    def _timeout_handler(signum, frame):
                        raise TimeoutError(f"Parse timed out for {pdf.name}")

                    signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(120)  # 2 minute timeout per file
                    parse_file(pdf, save_txt=True, save_json=True, out_dir=str(out_dir))
                    signal.alarm(0)  # cancel alarm
                except TimeoutError as e:
                    signal.alarm(0)
                    print(f"  [TIMEOUT] {pdf.name} - skipping (took >120s)")
                    continue
                except Exception as e:
                    signal.alarm(0)
                    print(f"  [ERR] {e}")
                    continue

            # Step 2: Chunk
            jsonl_name = pdf.stem + ".chunks.jsonl"
            jsonl_out = chunk_dir / jsonl_name
            if jsonl_out.exists():
                print(f"  [SKIP CHUNK] {jsonl_name} (already chunked)")
            else:
                # Pharmacy uses TXT input, others use JSON
                if name in ("Pharmacy", "Pharmacy Billing"):
                    txt_file = out_dir / (pdf.stem + ".txt")
                    if txt_file.exists():
                        preset = "pharmacy_policy" if name == "Pharmacy" else "pharmacy_billing"
                        section_semantic_chunking(
                            txt_path=str(txt_file),
                            output_dir=str(chunk_dir),
                            category=name,
                            max_chunk_chars=1200,
                            similarity_threshold=0.35,
                            preset=preset,
                            verbose=True,
                        )
                else:
                    json_file = out_dir / (pdf.stem + ".json")
                    if json_file.exists():
                        section_semantic_chunking(
                            json_path=str(json_file),
                            output_dir=str(chunk_dir),
                            max_chunk_chars=1200,
                            similarity_threshold=0.35,
                            verbose=True,
                        )

            batch_jsonl.append(jsonl_name)

            # Step 3: Check titles every BATCH_SIZE files
            if (i + 1) % BATCH_SIZE == 0 or i == len(pdfs) - 1:
                check_titles(chunk_dir, batch_jsonl)
                batch_jsonl = []
                if (i + 1) < len(pdfs):
                    print(f"\n--- Batch done ({i+1}/{len(pdfs)}), continuing... ---\n")

    print("\n\nALL DONE!")


if __name__ == "__main__":
    main()
