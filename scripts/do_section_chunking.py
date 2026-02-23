from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from healthcare_rag_llm.chunking.section_chunking import section_chunking


def main():
    ap = argparse.ArgumentParser(
        description="Chunk a policy-guidelines TXT file by ToC section boundaries"
    )
    ap.add_argument(
        "--input", "-i",
        default="data/processed/pharmacy/Pharmacy_Policy_Guidelines.txt",
        help="Path to input TXT file",
    )
    ap.add_argument(
        "--output", "-o",
        default="data/chunks/section_chunking_result",
        help="Output directory for JSONL chunks",
    )
    ap.add_argument(
        "--doc-id", default="",
        help="doc_id for JSONL records (defaults to <stem>.pdf)",
    )
    ap.add_argument(
        "--category", default="pharmacy",
        help="Category tag for each chunk record",
    )
    ap.add_argument(
        "--max-chars", "-s", type=int, default=0,
        help="Max characters per chunk (0 = no limit, each section is one chunk)",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = ap.parse_args()

    txt_path = Path(args.input)
    if not txt_path.is_absolute():
        txt_path = ROOT / txt_path

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    print(f"[INFO] Section chunking: {txt_path} -> {out_dir}")

    section_chunking(
        txt_path=str(txt_path),
        output_dir=str(out_dir),
        doc_id=args.doc_id,
        category=args.category,
        max_chunk_chars=args.max_chars,
        verbose=not args.quiet,
    )
    print("[DONE] Section-based chunking complete.")


if __name__ == "__main__":
    main()
