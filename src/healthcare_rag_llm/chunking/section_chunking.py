from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict, Tuple


def section_chunking(
    txt_path: str,
    output_dir: str,
    doc_id: str = "",
    category: str = "pharmacy",
    max_chunk_chars: int = 0,
    verbose: bool = True,
) -> None:
    """
    Section-based chunking that splits a policy-guidelines TXT file at
    subsection boundaries identified from its Table of Contents.

    Each ToC entry (primary section like "1.0 General Pharmacy Policy" and
    subsections like "Required Prescribing Information") becomes one chunk.
    If *max_chunk_chars* > 0, oversized sections are further split by size.

    Output:
      <output_dir>/<stem>.chunks.jsonl
    Record schema:
      {
        "doc_id", "chunk_id", "char_start", "char_end",
        "pages", "text", "chunk_type", "category", "section_title"
      }
    """
    txt_path = Path(txt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not doc_id:
        doc_id = txt_path.stem + ".pdf"

    raw_text = txt_path.read_text(encoding="utf-8")

    # ── 1. Parse ToC to get ordered section/subsection titles ──
    titles = _parse_toc(raw_text)
    if not titles:
        if verbose:
            print(f"[SKIP] {txt_path.name} (no ToC entries found)")
        return

    # ── 2. Build page-marker map (position → page number) before cleaning ──
    page_markers = _build_page_marker_map(raw_text)

    # ── 3. Locate the body start (first occurrence of "1.0 " after the ToC) ──
    toc_end_match = re.search(
        r"\n6\.0 Definitions\b[^\n]*\.\.*\s*\d+",  # last ToC line
        raw_text,
    )
    if toc_end_match:
        body_search_start = toc_end_match.end()
    else:
        body_search_start = 0

    first_section_match = re.search(r"^1\.0\s", raw_text[body_search_start:], re.MULTILINE)
    if first_section_match:
        body_start = body_search_start + first_section_match.start()
    else:
        body_start = 0

    body_raw = raw_text[body_start:]

    # ── 4. Clean page headers / footers from body text ──
    body_clean = _clean_page_markers(body_raw)

    # ── 5. Find each title position in the cleaned body and split ──
    split_points: List[Tuple[str, int]] = []  # (title, char_offset_in_clean)
    search_from = 0
    for title in titles:
        escaped = re.escape(title)
        m = re.search(rf"^{escaped}\s*$", body_clean[search_from:], re.MULTILINE)
        if m:
            pos = search_from + m.start()
            split_points.append((title, pos))
            search_from = pos + len(title)

    if not split_points:
        if verbose:
            print(f"[SKIP] {txt_path.name} (no section titles matched in body)")
        return

    # ── 6. Build chunks from consecutive split points ──
    chunks: List[Dict] = []
    chunk_idx = 0

    for i, (title, start) in enumerate(split_points):
        end = split_points[i + 1][1] if i + 1 < len(split_points) else len(body_clean)
        chunk_text = body_clean[start:end].strip()

        if not chunk_text:
            continue

        # Estimate pages from the raw text markers
        raw_start = body_start + start
        raw_end = body_start + end
        pages = _estimate_pages(page_markers, raw_start, raw_end, len(raw_text))

        if max_chunk_chars > 0 and len(chunk_text) > max_chunk_chars:
            # Secondary split for oversized sections
            offset = 0
            while offset < len(chunk_text):
                sub_text = chunk_text[offset : offset + max_chunk_chars]
                sub_pages = _estimate_pages(
                    page_markers,
                    raw_start + offset,
                    raw_start + offset + len(sub_text),
                    len(raw_text),
                )
                rec = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                    "char_start": start + offset,
                    "char_end": start + offset + len(sub_text),
                    "pages": sub_pages,
                    "text": sub_text,
                    "chunk_type": "text",
                    "category": category,
                    "section_title": title,
                }
                chunks.append(rec)
                chunk_idx += 1
                offset += max_chunk_chars
        else:
            rec = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                "char_start": start,
                "char_end": end,
                "pages": pages,
                "text": chunk_text,
                "chunk_type": "text",
                "category": category,
                "section_title": title,
            }
            chunks.append(rec)
            chunk_idx += 1

    # ── 7. Write JSONL output ──
    stem = txt_path.stem
    out_file = output_dir / f"{stem}.chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[OK]  {txt_path.name} -> {len(chunks)} chunks -> {out_file}")


# ── helpers ──────────────────────────────────────────────────────────────


def _parse_toc(text: str) -> List[str]:
    """Extract ordered section/subsection titles from the Table of Contents.

    ToC entries look like:
        1.0 General Pharmacy Policy ................ 4
        Required Prescribing Information .......... 4
    """
    # Find the ToC region
    toc_start = text.find("Table of Contents")
    if toc_start == -1:
        return []

    # ToC ends roughly where the body begins (first "1.0" that is NOT in ToC)
    # We search for the pattern "1.0 " at line start after a page marker
    toc_region_end = text.find("\n1.0 General Pharmacy Policy\n", toc_start + 200)
    if toc_region_end == -1:
        toc_region_end = toc_start + 5000  # fallback

    toc_text = text[toc_start:toc_region_end]

    # Match lines: "Title ..... page_number"
    toc_re = re.compile(r"^(.+?)\s*\.{3,}\s*(\d+)\s*$", re.MULTILINE)
    titles = []
    for m in toc_re.finditer(toc_text):
        title = m.group(1).strip()
        if title:
            titles.append(title)

    return titles


def _clean_page_markers(text: str) -> str:
    """Remove recurring page footers and headers from the body text.

    Patterns removed:
        - Page footer:  "2025-3 October 2025 <page_num>"
        - Page header:  "Policy Guidelines\\nNYRx"
    """
    # Footer pattern: "2025-3 October 2025 NN" (possibly at line start)
    text = re.sub(r"\n?\d{4}-\d+\s+\w+\s+\d{4}\s+\d+\s*", "\n", text)
    # Header pattern: "Policy Guidelines\nNYRx" (repeated at top of each page)
    text = re.sub(r"\nPolicy Guidelines\nNYRx\n?", "\n", text)
    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _build_page_marker_map(text: str) -> List[Tuple[int, int]]:
    """Return a sorted list of (char_position, page_number) extracted from
    page footer markers like '2025-3 October 2025 19'."""
    markers: List[Tuple[int, int]] = []
    for m in re.finditer(r"\d{4}-\d+\s+\w+\s+\d{4}\s+(\d+)", text):
        page_no = int(m.group(1))
        markers.append((m.start(), page_no))
    return markers


def _estimate_pages(
    markers: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
    text_len: int,
) -> List[int]:
    """Estimate which pages a character span covers based on page footer markers."""
    if not markers:
        return []

    pages: List[int] = []

    # Each marker at position P with page N means: content *before* P is on page N.
    # Content after the last marker on page N and before the next marker is on page N+1.
    for i, (pos, page_no) in enumerate(markers):
        # Determine the range this page covers
        if i == 0:
            page_start = 0
        else:
            page_start = markers[i - 1][0]

        page_end = pos

        # Check overlap with the chunk span
        if page_end > char_start and page_start < char_end:
            pages.append(page_no)

    # Also consider content after the last marker
    if markers:
        last_pos, last_page = markers[-1]
        if last_pos < char_end and text_len > last_pos:
            if last_page not in pages:
                pages.append(last_page)

    return sorted(set(pages))
