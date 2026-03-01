from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from healthcare_rag_llm.chunking.section_semantic_chunking import (
    section_semantic_chunking,
)


def main():
    ap = argparse.ArgumentParser(
        description="Hybrid section + semantic chunking for policy-guidelines TXT files"
    )
    ap.add_argument(
        "--input", "-i",
        default="data/processed/pharmacy/Pharmacy_Policy_Guidelines.txt",
        help="Path to input TXT file",
    )
    ap.add_argument(
        "--output", "-o",
        default="data/chunks/section_semantic_chunking_result",
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
        "--max-chars", "-s", type=int, default=1200,
        help="Max characters per chunk",
    )
    ap.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence-transformer model name",
    )
    ap.add_argument(
        "--unit", default="sentence", choices=["sentence", "paragraph"],
        help="Tokenization unit for semantic splitting",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.35,
        help="Cosine similarity threshold for merging",
    )
    ap.add_argument(
        "--hysteresis", type=float, default=0.02,
        help="Hysteresis band around threshold",
    )
    ap.add_argument("--quiet", action="store_true", help="Suppress logs")
    args = ap.parse_args()

    txt_path = Path(args.input)
    if not txt_path.is_absolute():
        txt_path = ROOT / txt_path

    out_dir = Path(args.output)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir

    print(f"[INFO] Section+Semantic chunking: {txt_path} -> {out_dir}")

    section_semantic_chunking(
        txt_path=str(txt_path),
        output_dir=str(out_dir),
        doc_id=args.doc_id,
        category=args.category,
        max_chunk_chars=args.max_chars,
        model_name=args.model,
        unit=args.unit,
        similarity_threshold=args.threshold,
        hysteresis=args.hysteresis,
        verbose=not args.quiet,
    )
    print("[DONE] Section+Semantic chunking complete.")


if __name__ == "__main__":
    main()
