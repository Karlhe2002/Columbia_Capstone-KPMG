"""
Convert section-chunked manual JSONL into the standard Neo4j ingest format.

Usage:
  python scripts/convert_manual_chunks.py \
    --input data/chunks/section_chunking_result/Pharmacy_Policy_Guidelines.chunks.jsonl \
    --output data/chunks/converted/Pharmacy_Policy_Guidelines.neo4j.jsonl \
    --category provider_manual
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DOC_ID_KEYS = [
    "doc_id",
    "document_id",
    "doc",
    "file_name",
    "filename",
    "source",
    "source_file",
    "pdf",
]

TEXT_KEYS = [
    "text",
    "chunk",
    "content",
    "section_text",
    "body",
]

CHUNK_ID_KEYS = [
    "chunk_id",
    "id",
    "chunkId",
    "chunk_id_str",
]


def _first_str_value(record: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _parse_pages(record: Dict[str, Any]) -> List[int]:
    pages = record.get("pages")
    if isinstance(pages, list) and pages:
        return [int(p) for p in pages if p is not None]
    if isinstance(pages, (int, float)):
        return [int(pages)]

    for key in ["page", "page_no", "page_num", "page_number"]:
        value = record.get(key)
        if isinstance(value, (int, float)):
            return [int(value)]

    start = record.get("page_start")
    end = record.get("page_end")
    if isinstance(start, (int, float)) and isinstance(end, (int, float)) and end >= start:
        return list(range(int(start), int(end) + 1))

    page_range = record.get("page_range")
    if isinstance(page_range, str) and page_range.strip():
        match = re.match(r"^\s*(\d+)\s*[-–]\s*(\d+)\s*$", page_range.strip())
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            if end >= start:
                return list(range(start, end + 1))

    return [1]


def _normalize_doc_id(record: Dict[str, Any], fallback_doc_id: Optional[str]) -> str:
    doc_id = _first_str_value(record, DOC_ID_KEYS)
    if doc_id:
        return doc_id
    if fallback_doc_id:
        return fallback_doc_id
    return "unknown.pdf"


def _normalize_chunk_id(
    record: Dict[str, Any],
    doc_id: str,
    idx: int,
) -> str:
    chunk_id = _first_str_value(record, CHUNK_ID_KEYS)
    if chunk_id:
        return chunk_id
    return f"{doc_id}::{str(idx).zfill(4)}"


def _normalize_text(record: Dict[str, Any]) -> str:
    text = _first_str_value(record, TEXT_KEYS)
    return text or ""


def convert_jsonl(
    input_path: Path,
    output_path: Path,
    category: str,
    doc_id_override: Optional[str] = None,
    chunk_type_default: str = "text",
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for idx, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            doc_id = _normalize_doc_id(record, doc_id_override)
            chunk_id = _normalize_chunk_id(record, doc_id, idx)
            text = _normalize_text(record)
            pages = _parse_pages(record)

            chunk_type = record.get("chunk_type") or chunk_type_default
            if isinstance(chunk_id, str):
                if "::table" in chunk_id:
                    chunk_type = "table"
                elif "::ocr" in chunk_id:
                    chunk_type = "ocr"

            out = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "pages": pages,
                "text": text,
                "chunk_type": chunk_type,
                "category": record.get("category") or category,
            }

            # Preserve optional fields if present
            if "char_start" in record:
                out["char_start"] = record["char_start"]
            if "char_end" in record:
                out["char_end"] = record["char_end"]
            if "effective_date" in record:
                out["effective_date"] = record["effective_date"]

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            count += 1

    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert manual JSONL chunks to Neo4j ingest format.")
    parser.add_argument("--input", required=True, help="Input .chunks.jsonl file (section chunking output)")
    parser.add_argument("--output", required=True, help="Output .jsonl path for Neo4j ingestion")
    parser.add_argument("--category", default="provider_manual", help="Category to assign when missing")
    parser.add_argument("--doc-id", default=None, help="Override doc_id for all records (optional)")
    parser.add_argument("--chunk-type", default="text", help="Default chunk_type when missing")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_path = Path(args.output)
    count = convert_jsonl(
        input_path=input_path,
        output_path=output_path,
        category=args.category,
        doc_id_override=args.doc_id,
        chunk_type_default=args.chunk_type,
    )
    print(f"[OK] Wrote {count} chunks -> {output_path}")


if __name__ == "__main__":
    main()
