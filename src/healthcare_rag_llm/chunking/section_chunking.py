from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict, Tuple, Optional


# ── Document-specific presets ────────────────────────────────────────────

PHARMACY_POLICY_PRESET = {
    "footer_patterns": [
        r"\n?\d{4}-\d+\s+\w+\s+\d{4}\s+\d+\s*",
    ],
    "header_patterns": [
        r"\nPolicy Guidelines\nNYRx\n?",
    ],
    "page_marker_regex": r"\d{4}-\d+\s+\w+\s+\d{4}\s+(\d+)",
}

PHARMACY_BILLING_PRESET = {
    "footer_patterns": [
        r"\nPHARMACY\nVersion \d{4} - \d+ \d+/\d+/\d{4}\nPage \d+ of \d+\n?",
    ],
    "header_patterns": [
        r"^EMEDNY INFORMATION\s*$",
        r"^TABLE OF CONTENTS\s*$",
        r"^PURPOSE STATEMENT\s*$",
        r"^CLAIMS SUBMISSION\s*$",
        r"^REMITTANCE ADVICE\s*$",
        r"^APPENDIX [AB] [A-Z ]+\s*$",
    ],
    "page_marker_regex": r"Page (\d+) of \d+",
    "artifacts": [
        r"\[Type text\]\s*",
        r"PPENDIX\nA [AB]\n[A-Z ]+ [A-Z]+\nC S\n?",
    ],
}

PRESETS = {
    "pharmacy_policy": PHARMACY_POLICY_PRESET,
    "pharmacy_billing": PHARMACY_BILLING_PRESET,
}


def section_chunking(
    txt_path: str,
    output_dir: str,
    doc_id: str = "",
    category: str = "pharmacy",
    max_chunk_chars: int = 0,
    preset: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Section-based chunking that splits a TXT file at subsection boundaries
    identified from its Table of Contents.

    Each ToC entry becomes one chunk. If *max_chunk_chars* > 0, oversized
    sections are further split by size.

    Args:
        preset: Name of a document preset (e.g. "pharmacy_policy",
                "pharmacy_billing") for page header/footer patterns.
                If None, auto-detects based on file content.
    """
    txt_path = Path(txt_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not doc_id:
        doc_id = txt_path.stem + ".pdf"

    raw_text = txt_path.read_text(encoding="utf-8")

    # Auto-detect preset if not specified
    config = _get_config(raw_text, preset)

    # ── 1. Parse ToC to get ordered section/subsection titles ──
    titles = _parse_toc(raw_text)
    if not titles:
        if verbose:
            print(f"[SKIP] {txt_path.name} (no ToC entries found)")
        return

    # ── 2. Build page-marker map (position → page number) before cleaning ──
    page_markers = _build_page_marker_map(raw_text, config["page_marker_regex"])

    # ── 3. Locate the body start ──
    body_start = _find_body_start(raw_text, titles)
    body_raw = raw_text[body_start:]

    # ── 4. Clean page headers / footers from body text ──
    body_clean = _clean_page_markers(body_raw, config)

    # ── 5. Find each title position in the cleaned body and split ──
    split_points: List[Tuple[str, int]] = []
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

        raw_start = body_start + start
        raw_end = body_start + end
        pages = _estimate_pages(page_markers, raw_start, raw_end, len(raw_text))

        if max_chunk_chars > 0 and len(chunk_text) > max_chunk_chars:
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


def _get_config(text: str, preset: Optional[str] = None) -> dict:
    """Return document-specific config, auto-detecting if preset is None."""
    if preset and preset in PRESETS:
        return PRESETS[preset]

    # Auto-detect based on content
    if "eMedNY Billing Guidelines" in text or "Page 1 of" in text:
        return PRESETS["pharmacy_billing"]
    if "Policy Guidelines" in text and "NYRx" in text:
        return PRESETS["pharmacy_policy"]

    # Fallback: empty patterns (no cleaning)
    return {
        "footer_patterns": [],
        "header_patterns": [],
        "page_marker_regex": r"Page (\d+) of \d+",
    }


# Hardcoded sections for specific documents that have no ToC but clear sections
_HARDCODED_SECTIONS = {
    "Regarding Medicaid Eligibility and Transfer Process to Health Homes": [
        "A. Children/Youth Referred to the Children\u2019s Waiver without Active Medicaid",
        "B. Children/Youth Referred for Health Home Care Management through C-YES Transfer",
    ],
    "Frequently Asked Questions\nfor Providers and Stakeholders": [
        "1) Q: How can I contact Children and Youth Evaluation Service (C-YES) and/or make a referral",
        "2) Q: Does C-YES assist with Home-Care and/or Private Duty Nursing coordination?",
        "3) Q: Does C-YES provide 24-7 emergency phone coverage and response for families",
        "4) Q: Who can make a referral to C-YES for a child/youth who might need HCBS?",
        "5) Q: How does a referent and or other involved providers know where in the HCBS",
        "6) Q: Does C-YES talk with involved providers to help with the HCBS determination",
        "7) Q: Does C-YES make referrals to the lead Health Home or directly to the Care",
        "8) Q: Can providers refer consumers to C-YES for Medicaid eligibility or if/when",
        "9) Q: How do referents and or involved providers know if the child/youth has been",
        "10) Q: Can a child receive care coordination from both C-YES and a Health Home at",
        "11) Q: Does C-YES confirm availability of the HCBS provider prior to referring the",
        "12) Q: How do involved providers know that the family was transferred to Health",
        "13) Q: How does C-YES co-manage cases for consumers enrolled in Medicaid managed",
    ],
}


def _parse_toc(text: str) -> List[str]:
    """Extract ordered section/subsection titles from the Table of Contents.

    Works generically for any document with a ToC section containing lines like:
        1.0 General Pharmacy Policy ................ 4
        2.1 Electronic Claims ...................... 5
        Appendix A Claim Samples .................. 22
    """
    # Check hardcoded sections first (match by unique text in document)
    for key, titles in _HARDCODED_SECTIONS.items():
        if key in text:
            return titles

    # Find the ToC region (supports multiple heading formats)
    toc_start = -1
    for heading in ["Table of Contents", "TABLE OF CONTENTS", "In This Issue", "Inside this issue", "Contents"]:
        toc_start = text.find(heading)
        if toc_start != -1:
            break
    if toc_start == -1:
        return []

    # Patterns for ToC lines:
    # 1. "Title ............. 4"  (continuous dots)
    # 2. "Title . . . . . pg. 3" (spaced dots)
    # 3. "Title 9"               (just title + page number at end, for sub-entries)
    dotted_re = re.compile(
        r"^(.+?)\s*[.\s]{6,}\s*(?:pg\.?\s*)?(\d+(?:-\d+)?|Cover)\s*$"
    )
    # For lines like "Overview 9", "Vision and Goals 10" (no dots, just trailing number)
    bare_page_re = re.compile(
        r"^([A-Z].{2,}?)\s+(\d+(?:-\d+)?)\s*$"
    )

    skip_headings = {"TABLE OF CONTENTS", "CONTENTS", "IN THIS ISSUE", "IN THIS ISSUE…"}

    toc_region = text[toc_start:]
    toc_lines = toc_region.split("\n")

    # Detect end of ToC region: stop when we hit body text.
    # Limit search to first 100 lines max (no ToC is longer than that).
    max_scan = min(len(toc_lines), 100)
    non_toc_streak = 0
    toc_end_idx = max_scan
    for idx in range(max_scan):
        ls = toc_lines[idx].strip()
        if not ls:
            continue
        if dotted_re.match(ls) or bare_page_re.match(ls):
            non_toc_streak = 0
        else:
            non_toc_streak += 1
            if non_toc_streak >= 3 and idx > 5:
                toc_end_idx = idx - 2
                break

    toc_lines = toc_lines[:toc_end_idx]

    # Collect all ToC entries, merging continuation lines.
    all_entries = []
    pending_prefix = ""

    for line in toc_lines:
        line_stripped = line.strip()
        if not line_stripped:
            pending_prefix = ""
            continue

        # Try dotted match first (most common)
        m = dotted_re.match(line_stripped)
        if m:
            title_part = m.group(1).strip()
            if pending_prefix:
                full_title = pending_prefix + " " + title_part
                pending_prefix = ""
            else:
                full_title = title_part
            if full_title.upper() not in skip_headings:
                all_entries.append(full_title)
            continue

        # Try bare page number match (sub-entries like "Overview 9")
        # Only used as fallback — collected separately and merged later
        # if too few dotted entries were found.
        m2 = bare_page_re.match(line_stripped)
        if m2:
            # Skip bare matches for now; handled after main loop
            pending_prefix = ""
            continue

        # No match — could be continuation line or category header
        if "…" not in line_stripped and "..." not in line_stripped:
            is_category = len(line_stripped) <= 40 and not re.search(r"\d", line_stripped)
            if is_category:
                pending_prefix = ""
            elif pending_prefix:
                pending_prefix += " " + line_stripped
            else:
                pending_prefix = line_stripped
        else:
            pending_prefix = ""

    if not all_entries:
        return []

    titles = all_entries
    return titles


def _find_body_start(text: str, titles: List[str]) -> int:
    """Find where the document body starts by locating the first ToC title
    in the text AFTER the ToC region itself."""
    if not titles:
        return 0

    first_title = titles[0]

    # Find all occurrences of the first title
    pattern = re.compile(rf"^{re.escape(first_title)}\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if len(matches) >= 2:
        # Second occurrence is the body start (first is in the ToC)
        return matches[1].start()
    elif len(matches) == 1:
        return matches[0].start()

    return 0


def _clean_page_markers(text: str, config: dict) -> str:
    """Remove page footers and headers using patterns from config."""
    for pattern in config.get("footer_patterns", []):
        text = re.sub(pattern, "\n", text)

    for pattern in config.get("header_patterns", []):
        text = re.sub(pattern, "\n", text, flags=re.MULTILINE)

    # Remove any extra artifacts
    for pattern in config.get("artifacts", []):
        text = re.sub(pattern, "", text)

    # Collapse runs of 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _build_page_marker_map(
    text: str,
    page_marker_regex: str = r"\d{4}-\d+\s+\w+\s+\d{4}\s+(\d+)",
) -> List[Tuple[int, int]]:
    """Return a sorted list of (char_position, page_number) extracted from
    page markers matching the given regex. Group 1 must capture the page number."""
    markers: List[Tuple[int, int]] = []
    for m in re.finditer(page_marker_regex, text):
        page_no = int(m.group(1))
        markers.append((m.start(), page_no))
    return markers


def _estimate_pages(
    markers: List[Tuple[int, int]],
    char_start: int,
    char_end: int,
    text_len: int,
) -> List[int]:
    """Estimate which pages a character span covers based on page markers."""
    if not markers:
        return []

    pages: List[int] = []

    for i, (pos, page_no) in enumerate(markers):
        if i == 0:
            page_start = 0
        else:
            page_start = markers[i - 1][0]

        page_end = pos

        if page_end > char_start and page_start < char_end:
            pages.append(page_no)

    if markers:
        last_pos, last_page = markers[-1]
        if last_pos < char_end and text_len > last_pos:
            if last_page not in pages:
                pages.append(last_page)

    return sorted(set(pages))
