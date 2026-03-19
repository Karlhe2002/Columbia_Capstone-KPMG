from __future__ import annotations

from pathlib import Path
import json
import re
from typing import List, Dict, Tuple, Optional

import numpy as np
import nltk
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from healthcare_rag_llm.chunking.section_chunking import (
    _parse_toc,
    _find_body_start,
    _build_page_marker_map,
    _estimate_pages,
    _get_config,
)

# Ensure NLTK data is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def section_semantic_chunking(
    txt_path: str = "",
    json_path: str = "",
    output_dir: str = "",
    doc_id: str = "",
    category: str = "",
    max_chunk_chars: int = 1200,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    unit: str = "sentence",
    similarity_threshold: float = 0.35,
    hysteresis: float = 0.02,
    preset: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Unified two-level hybrid chunking for all document types.

      Level 1 — Structural hard splits, auto-detected:
        1. ToC section boundaries (pharmacy manuals, some waiver docs)
        2. Asterisk separator runs (Medicaid Updates)
        3. No structure → skip Level 1, pure semantic

      Level 2 — Within each section/segment, apply semantic chunking
                (hysteresis-based embedding similarity).

    Input:
        - txt_path: Path to a plain text file (pharmacy docs)
        - json_path: Path to a parsed JSON file with full_text + pages
                     (Medicaid Updates, Children Waiver)
        Exactly one of txt_path or json_path must be provided.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load input ──
    if json_path:
        jp = Path(json_path)
        with open(jp, "r", encoding="utf-8") as f:
            meta = json.load(f)
        raw_text = meta.get("full_text", "")
        json_pages = meta.get("pages") or []
        file_name = meta.get("file_name") or jp.name
        if not doc_id:
            doc_id = file_name
        if not category:
            category = meta.get("category") or "unknown"
        if not raw_text:
            if verbose:
                print(f"[SKIP] {jp.name} (empty full_text)")
            return
        input_name = jp.name
        input_mode = "json"
    elif txt_path:
        tp = Path(txt_path)
        raw_text = tp.read_text(encoding="utf-8")
        json_pages = None
        if not doc_id:
            doc_id = tp.stem + ".pdf"
        if not category:
            category = "unknown"
        input_name = tp.name
        input_mode = "txt"
    else:
        raise ValueError("Either txt_path or json_path must be provided")

    # ── Detect boundary type ──
    boundary_mode, titles = _detect_boundary_mode(raw_text)
    if verbose:
        print(f"  [{input_name}] boundary_mode={boundary_mode}, "
              f"{'ToC entries=' + str(len(titles)) if titles else 'no ToC'}")

    # ── Load sentence-transformer model ──
    if verbose:
        print(f"  Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    # ── Dispatch by boundary mode ──
    if boundary_mode == "toc":
        chunks = _chunk_toc_mode(
            raw_text, titles, doc_id, category,
            model, unit, similarity_threshold, hysteresis, max_chunk_chars,
            json_pages, input_mode, preset, verbose,
        )
    elif boundary_mode == "asterisk":
        chunks = _chunk_asterisk_mode(
            raw_text, doc_id, category,
            model, unit, similarity_threshold, hysteresis, max_chunk_chars,
            json_pages, verbose, toc_titles=titles,
        )
    else:  # fallback: pure semantic
        chunks = _chunk_fallback_mode(
            raw_text, doc_id, category,
            model, unit, similarity_threshold, hysteresis, max_chunk_chars,
            json_pages, verbose,
        )

    if not chunks:
        if verbose:
            print(f"[SKIP] {input_name} (no chunks produced)")
        return

    # ── Filter out junk chunks (page headers, single chars, etc.) ──
    chunks = [c for c in chunks if len(c["text"].strip()) >= MIN_CHUNK_CHARS]

    # ── Filter out ToC chunks (entire "In This Issue" / ToC pages) ──
    chunks = [c for c in chunks if not _is_toc_chunk(c["text"])]

    # ── Hard-split oversized chunks as fallback ──
    final_chunks: List[Dict] = []
    for c in chunks:
        if len(c["text"]) > max_chunk_chars + 500:
            # Split into smaller pieces
            text = c["text"]
            offset = 0
            sub_idx = 0
            while offset < len(text):
                end = min(offset + max_chunk_chars, len(text))
                # Try to break at a sentence boundary
                if end < len(text):
                    last_period = text.rfind(". ", offset, end)
                    if last_period > offset + max_chunk_chars // 2:
                        end = last_period + 1
                sub_text = text[offset:end].strip()
                if len(sub_text) >= MIN_CHUNK_CHARS:
                    sub_rec = dict(c)
                    sub_rec["text"] = sub_text
                    sub_rec["chunk_id"] = f"{c['chunk_id']}_{sub_idx}"
                    final_chunks.append(sub_rec)
                    sub_idx += 1
                offset = end
        else:
            final_chunks.append(c)
    chunks = final_chunks

    # ── Re-number chunk IDs ──
    for i, c in enumerate(chunks):
        c["chunk_id"] = f"{doc_id}::{str(i).zfill(4)}"

    # ── Write JSONL ──
    stem = Path(doc_id).stem
    out_file = output_dir / f"{stem}.chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[OK]  {input_name} -> {len(chunks)} chunks -> {out_file}")


# ── Boundary detection ────────────────────────────────────────────────


def _detect_boundary_mode(text: str) -> Tuple[str, List[str]]:
    """Auto-detect the best Level 1 boundary strategy.

    Returns:
        (mode, titles)
        mode: "toc", "asterisk", or "fallback"
        titles: list of ToC titles if mode is "toc", else []
    """
    titles = _parse_toc(text)
    asterisk_count = len(re.findall(r"\*{10,}", text))

    # If doc has BOTH asterisks and a ToC ("In This Issue"), use asterisk mode
    # but pass ToC titles for better section_title extraction.
    # Medicaid Updates have "In This Issue" as ToC but titles don't repeat
    # as standalone lines in the body — they're only in the ToC region.
    if asterisk_count >= 2:
        return "asterisk", titles

    # Pure ToC mode (Pharmacy manuals, some waiver docs):
    # titles appear both in ToC AND as standalone lines in the body.
    if titles:
        return "toc", titles

    # No structure
    return "fallback", []


# ── ToC mode (existing logic, works for TXT and JSON) ────────────────


def _chunk_toc_mode(
    raw_text: str,
    titles: List[str],
    doc_id: str,
    category: str,
    model: SentenceTransformer,
    unit: str,
    similarity_threshold: float,
    hysteresis: float,
    max_chunk_chars: int,
    json_pages: Optional[List[Dict]],
    input_mode: str,
    preset: Optional[str],
    verbose: bool,
) -> List[Dict]:
    """Level 1 = ToC sections, Level 2 = semantic sub-chunking."""

    # Page estimation setup
    if input_mode == "txt":
        config = _get_config(raw_text, preset)
        page_markers = _build_page_marker_map(raw_text, config["page_marker_regex"])
        page_estimator = lambda start, end: _estimate_pages(
            page_markers, start, end, len(raw_text)
        )
    else:
        page_spans = _build_json_page_spans(json_pages, raw_text)
        page_estimator = lambda start, end: _pages_from_spans(page_spans, start, end)
        config = {"footer_patterns": [], "header_patterns": []}

    # Locate body start
    body_start = _find_body_start(raw_text, titles)
    body_raw = raw_text[body_start:]

    # Clean page markers (only for TXT with known patterns)
    if input_mode == "txt" and (config.get("footer_patterns") or config.get("header_patterns")):
        body_clean, clean_to_raw = _clean_page_markers_preserve_breaks(body_raw, config)
    else:
        body_clean = body_raw
        clean_to_raw = list(range(len(body_raw)))

    def _map_to_raw(clean_pos: int) -> int:
        if clean_pos < len(clean_to_raw):
            return body_start + clean_to_raw[clean_pos]
        return body_start + (clean_to_raw[-1] if clean_to_raw else 0)

    # Find section split points
    split_points: List[Tuple[str, int]] = []
    search_from = 0
    for title in titles:
        # Try strict match first (exact line)
        escaped = re.escape(title)
        m = re.search(rf"^{escaped}\s*$", body_clean[search_from:], re.MULTILINE)
        if m:
            pos = search_from + m.start()
            split_points.append((title, pos))
            search_from = pos + len(title)
            continue

        # Fuzzy match: normalize whitespace and search as substring
        # Strip leading numbering like "I. ", "II. ", "1.0 " for matching
        title_core = re.sub(r"^(?:[IVXLC]+\.\s*|\d+\.\d*\s*)", "", title).strip()
        if len(title_core) < 10:
            continue
        title_norm = re.sub(r"\s+", " ", title_core)
        body_norm = re.sub(r"\s+", " ", body_clean[search_from:])
        idx = body_norm.find(title_norm)
        if idx != -1:
            # Map back to body_clean position (approximate)
            pos = search_from + idx
            split_points.append((title, pos))
            search_from = pos + len(title_core)

    if not split_points:
        if verbose:
            print(f"  [WARN] No section titles matched in body, falling back to semantic")
        return _chunk_fallback_mode(
            raw_text, doc_id, category, model, unit,
            similarity_threshold, hysteresis, max_chunk_chars,
            json_pages if input_mode == "json" else None, verbose,
        )

    # Build chunks
    chunks: List[Dict] = []
    chunk_idx = 0

    for i, (title, start) in enumerate(split_points):
        end = split_points[i + 1][1] if i + 1 < len(split_points) else len(body_clean)
        section_text = body_clean[start:end].strip()
        if not section_text:
            continue

        raw_start = _map_to_raw(start)
        raw_end = _map_to_raw(end)

        if len(section_text) <= max_chunk_chars:
            pages = page_estimator(raw_start, raw_end)
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                "char_start": start,
                "char_end": end,
                "pages": pages,
                "text": section_text,
                "chunk_type": "text",
                "category": category,
                "section_title": title,
            })
            chunk_idx += 1
        else:
            sub_chunks = _semantic_split(
                section_text, model, unit,
                similarity_threshold, hysteresis, max_chunk_chars,
            )
            for sc in sub_chunks:
                sc_raw_start = _map_to_raw(start + sc["offset"])
                sc_raw_end = _map_to_raw(start + sc["offset"] + len(sc["text"]))
                pages = page_estimator(sc_raw_start, sc_raw_end)
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                    "char_start": start + sc["offset"],
                    "char_end": start + sc["offset"] + len(sc["text"]),
                    "pages": pages,
                    "text": sc["text"],
                    "chunk_type": "text",
                    "category": category,
                    "section_title": title,
                })
                chunk_idx += 1

    return chunks


# ── Asterisk mode (Medicaid Updates) ─────────────────────────────────


def _chunk_asterisk_mode(
    raw_text: str,
    doc_id: str,
    category: str,
    model: SentenceTransformer,
    unit: str,
    similarity_threshold: float,
    hysteresis: float,
    max_chunk_chars: int,
    json_pages: Optional[List[Dict]],
    verbose: bool,
    toc_titles: Optional[List[str]] = None,
) -> List[Dict]:
    """Level 1 = asterisk separators, Level 2 = semantic sub-chunking."""

    sep_re = re.compile(r"\*{10,}")

    # Page estimation from JSON pages
    if json_pages:
        page_spans = _build_json_page_spans(json_pages, raw_text)
        page_estimator = lambda start, end: _pages_from_spans(page_spans, start, end)
    else:
        page_estimator = lambda start, end: []

    # Split on asterisk runs
    segments: List[Tuple[int, int]] = []
    last_end = 0
    for m in sep_re.finditer(raw_text):
        start, end = m.span()
        if start > last_end:
            segments.append((last_end, start))
        last_end = end
    if last_end < len(raw_text):
        segments.append((last_end, len(raw_text)))

    # Pre-process: if first segment contains "In This Issue", split it
    # into cover article (before ToC) and discard the ToC/boilerplate region
    segments = _split_cover_segment(segments, raw_text)

    chunks: List[Dict] = []
    chunk_idx = 0

    for seg_start, seg_end in segments:
        seg_text = raw_text[seg_start:seg_end].strip()
        if not seg_text:
            continue

        # Try to match segment to a ToC title first, fallback to first-line extraction
        section_title = _match_toc_title(seg_text, toc_titles) if toc_titles else ""
        if not section_title:
            section_title = _extract_article_title(seg_text)

        if len(seg_text) <= max_chunk_chars:
            pages = page_estimator(seg_start, seg_end)
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                "char_start": seg_start,
                "char_end": seg_end,
                "pages": pages,
                "text": seg_text,
                "chunk_type": "text",
                "category": category,
                "section_title": section_title,
            })
            chunk_idx += 1
        else:
            sub_chunks = _semantic_split(
                seg_text, model, unit,
                similarity_threshold, hysteresis, max_chunk_chars,
            )
            for sc in sub_chunks:
                sc_abs_start = seg_start + sc["offset"]
                sc_abs_end = sc_abs_start + len(sc["text"])
                pages = page_estimator(sc_abs_start, sc_abs_end)
                chunks.append({
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                    "char_start": sc_abs_start,
                    "char_end": sc_abs_end,
                    "pages": pages,
                    "text": sc["text"],
                    "chunk_type": "text",
                    "category": category,
                    "section_title": section_title,
                })
                chunk_idx += 1

    return chunks


# ── Fallback mode (pure semantic, no Level 1) ────────────────────────


def _chunk_fallback_mode(
    raw_text: str,
    doc_id: str,
    category: str,
    model: SentenceTransformer,
    unit: str,
    similarity_threshold: float,
    hysteresis: float,
    max_chunk_chars: int,
    json_pages: Optional[List[Dict]],
    verbose: bool,
) -> List[Dict]:
    """No structural boundaries — pure semantic chunking on full text."""

    if json_pages:
        page_spans = _build_json_page_spans(json_pages, raw_text)
        page_estimator = lambda start, end: _pages_from_spans(page_spans, start, end)
    else:
        page_estimator = lambda start, end: []

    sub_chunks = _semantic_split(
        raw_text, model, unit,
        similarity_threshold, hysteresis, max_chunk_chars,
    )

    chunks: List[Dict] = []
    for idx, sc in enumerate(sub_chunks):
        sc_start = sc["offset"]
        sc_end = sc_start + len(sc["text"])
        # Fallback mode has no structural sections, leave title empty
        section_title = ""

        pages = page_estimator(sc_start, sc_end)
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}::{str(idx).zfill(4)}",
            "char_start": sc_start,
            "char_end": sc_end,
            "pages": pages,
            "text": sc["text"],
            "chunk_type": "text",
            "category": category,
            "section_title": section_title,
        })

    return chunks


# ── ToC chunk detection ──────────────────────────────────────────────


def _is_toc_chunk(text: str) -> bool:
    """Check if a chunk is mostly a Table of Contents page (lots of dotted lines)."""
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return False
    dotted_lines = sum(1 for l in lines if re.search(r"[.\s]{6,}\d+", l))
    # If more than 40% of lines are dotted ToC entries, it's a ToC chunk
    return dotted_lines / len(lines) > 0.4


# ── Cover segment splitting ──────────────────────────────────────────


def _split_cover_segment(
    segments: List[Tuple[int, int]], raw_text: str
) -> List[Tuple[int, int]]:
    """If the first segment contains 'In This Issue', split it:
    - Keep cover article text (before the boilerplate/ToC block)
    - Discard the Governor info + ToC region
    The boilerplate block typically starts with a person's name followed by
    Governor/Commissioner lines, then 'In This Issue'.
    """
    if not segments:
        return segments

    seg_start, seg_end = segments[0]
    seg_text = raw_text[seg_start:seg_end]

    iti_pos = seg_text.find("In This Issue")
    if iti_pos == -1:
        return segments

    # Find where the boilerplate block starts (before "In This Issue").
    # Walk backwards from "In This Issue" to find the start of the
    # Governor/Commissioner block. Look for a blank line gap or known
    # boilerplate markers.
    # The block usually has lines like:
    #   "Andrew M. Cuomo\nGovernor\nState of New York\n..."
    # We find the blank-line boundary before this block.
    pre_iti = seg_text[:iti_pos]

    # Find the last substantive paragraph break before the boilerplate.
    # The boilerplate block usually starts after a blank line following
    # the last article paragraph.
    # Strategy: search backwards for "Governor" or "Commissioner" to find
    # where the boilerplate starts, then find the paragraph break before it.
    boilerplate_start = iti_pos  # default: cut right at "In This Issue"

    gov_match = re.search(r"\n(?:Governor|Commissioner)\n", pre_iti)
    if gov_match:
        # Find the paragraph break (double newline or just before the name line)
        # by searching backwards from the Governor line for a blank line
        before_gov = pre_iti[:gov_match.start()]
        last_break = before_gov.rfind("\n\n")
        if last_break != -1:
            boilerplate_start = last_break
        else:
            boilerplate_start = gov_match.start()

    # Find end of ToC region (last dotted line after "In This Issue")
    toc_end = iti_pos
    toc_line_re = re.compile(r"^.+?\.{3,}\s*(?:\d+|Cover)\s*$", re.MULTILINE)
    for m in toc_line_re.finditer(seg_text[iti_pos:]):
        toc_end = iti_pos + m.end()

    # Build new segments:
    # 1. Cover article: seg_start to boilerplate_start (the real content)
    # 2. Skip boilerplate + ToC
    # 3. If there's content after ToC but before next asterisk, include it
    new_segments = []

    cover_end = seg_start + boilerplate_start
    if cover_end > seg_start + 50:  # only if there's meaningful cover content
        new_segments.append((seg_start, cover_end))

    after_toc_start = seg_start + toc_end
    if after_toc_start < seg_end:
        remaining = raw_text[after_toc_start:seg_end].strip()
        if len(remaining) > 50:
            new_segments.append((after_toc_start, seg_end))

    # Add the rest of the segments unchanged
    new_segments.extend(segments[1:])
    return new_segments


# ── Title extraction helper ───────────────────────────────────────────

# Patterns that look like page headers rather than real article titles
def _is_boilerplate_line(line: str) -> bool:
    """Check if a line is boilerplate (page header, metadata) rather than a real title."""
    s = line.strip()
    if not s:
        return True

    # Too short to be a title (single word like "Governor", "Commissioner")
    if len(s.split()) <= 1 and len(s) < 20:
        return True

    # Date-like: "January 2025", "February 2019"
    if re.match(r"^(?:January|February|March|April|May|June|July|August|"
                r"September|October|November|December)\s+\d{4}$", s, re.IGNORECASE):
        return True

    # Volume/Number: "Volume 41 | Number 1"
    if re.match(r"Volume\s+\d+", s, re.IGNORECASE):
        return True

    # Page references: "...pg. 4", "Page 3 of 25", "Continued on Page 3"
    if re.search(r"(?:pg\.?\s*\d+|Page\s+\d+\s+of\s+\d+|Continued on Page)", s, re.IGNORECASE):
        return True

    # Medicaid Update header
    if re.search(r"Medicaid Update", s, re.IGNORECASE) and len(s.split()) <= 10:
        return True

    # Looks like a person's name: 2-4 words, possibly with titles/suffixes like M.D., M.P.H.
    # e.g. "Andrew M. Cuomo", "James McDonald, M.D., M.P.H.", "Kathy Hochul"
    if re.match(
        r"^[A-Z][a-z]+(?:\s+[A-Z]\.?\s*)*(?:\s+[A-Z][a-z]+)*"
        r"(?:,?\s*(?:M\.?D\.?|M\.?P\.?H\.?|Ph\.?D\.?|J\.?D\.?|Jr\.?|Sr\.?))*\s*$",
        s
    ) and len(s.split()) <= 6:
        return True

    # "Special Edition", "In This Issue"
    if re.match(r"^(?:Special Edition|In This Issue).*$", s, re.IGNORECASE):
        return True

    return False


def _match_toc_title(seg_text: str, toc_titles: List[str]) -> str:
    """Find the best matching ToC title for this segment.

    Checks if any ToC title appears in the first ~500 chars of the segment.
    Handles line-break differences by normalizing whitespace before matching.
    Returns the longest matching title (most specific match).
    """
    # Normalize whitespace in segment head for matching
    head = seg_text[:500]
    head_normalized = re.sub(r"\s+", " ", head)

    matches = []
    for title in toc_titles:
        title_normalized = re.sub(r"\s+", " ", title)
        if title_normalized in head_normalized:
            matches.append(title)
    if matches:
        return max(matches, key=len)
    return ""


def _extract_article_title(seg_text: str) -> str:
    """Extract the real article title from a segment, skipping boilerplate lines."""
    lines = seg_text.split("\n")
    for line in lines:
        line_stripped = line.strip()
        if _is_boilerplate_line(line_stripped):
            continue
        if len(line_stripped) < 200:
            return line_stripped
    return ""


# ── JSON page helpers ────────────────────────────────────────────────


def _build_json_page_spans(
    pages: List[Dict], full_text: str
) -> List[Tuple[int, int, int]]:
    """Build (page_no, start, end) spans from JSON pages array.

    Tries double-newline join first, falls back to text search.
    """
    # Try double-newline assumption
    spans = _try_double_newline_spans(pages, full_text)
    if spans:
        return spans

    # Fallback: search for each page's text
    spans = []
    cursor = 0
    for i, p in enumerate(pages):
        page_no = int(p.get("page", i + 1))
        page_text = (p.get("text") or "").strip()
        if not page_text:
            spans.append((page_no, cursor, cursor))
            continue
        pos = full_text.find(page_text, cursor)
        if pos == -1:
            # Can't find page text, skip
            spans.append((page_no, cursor, cursor))
            continue
        spans.append((page_no, pos, pos + len(page_text)))
        cursor = pos + len(page_text)
    return spans


def _try_double_newline_spans(
    pages: List[Dict], full_text: str
) -> List[Tuple[int, int, int]]:
    """Attempt to map pages assuming they are joined by double newlines."""
    spans = []
    cursor = 0
    total_len = len(full_text)
    for i, p in enumerate(pages):
        page_no = int(p.get("page", i + 1))
        page_text = p.get("text", "") or ""
        start = cursor
        end = start + len(page_text)
        if end > total_len:
            return []
        spans.append((page_no, start, end))
        cursor = end
        if i < len(pages) - 1:
            if cursor + 1 < total_len and full_text[cursor:cursor + 2] == "\n\n":
                cursor += 2
            else:
                return []
    if cursor != total_len:
        return []
    return spans


def _pages_from_spans(
    page_spans: List[Tuple[int, int, int]], c_start: int, c_end: int
) -> List[int]:
    """Find which pages a character span overlaps."""
    out = []
    for page_no, p_start, p_end in page_spans:
        if p_end <= c_start:
            continue
        if p_start >= c_end:
            break
        out.append(page_no)
    return out


# ── Shared helpers ───────────────────────────────────────────────────


def _clean_page_markers_preserve_breaks(text: str, config: dict) -> Tuple[str, List[int]]:
    """Remove page footers/headers but replace each page break with ``\\n\\n``
    so that paragraph boundaries survive for sentence/paragraph tokenisers.

    Returns:
        (cleaned_text, clean_to_raw_map)
        clean_to_raw_map[i] gives the position in the original text that
        corresponds to position i in the cleaned text.
    """
    raw_indices = list(range(len(text)))

    def _sub_with_map(pattern, replacement, current_text, indices, flags=0):
        result_chars = []
        result_indices = []
        last_end = 0
        for m in re.finditer(pattern, current_text, flags=flags):
            result_chars.append(current_text[last_end:m.start()])
            result_indices.extend(indices[last_end:m.start()])
            result_chars.append(replacement)
            result_indices.extend([indices[m.start()]] * len(replacement))
            last_end = m.end()
        result_chars.append(current_text[last_end:])
        result_indices.extend(indices[last_end:])
        return "".join(result_chars), result_indices

    current = text
    indices = raw_indices

    for pattern in config.get("footer_patterns", []):
        current, indices = _sub_with_map(pattern, "\n\n", current, indices)

    for pattern in config.get("header_patterns", []):
        current, indices = _sub_with_map(pattern, "\n\n", current, indices, flags=re.MULTILINE)

    for pattern in config.get("artifacts", []):
        current, indices = _sub_with_map(pattern, "", current, indices)

    # Collapse 3+ newlines into 2
    current, indices = _sub_with_map(r"\n{3,}", "\n\n", current, indices)

    return current, indices


MIN_CHUNK_CHARS = 80


def _semantic_split(
    text: str,
    model: SentenceTransformer,
    unit: str,
    similarity_threshold: float,
    hysteresis: float,
    max_chunk_chars: int,
) -> List[Dict]:
    """Apply hysteresis-based semantic merging within a single section.

    Returns a list of dicts: [{"text": ..., "offset": <char offset in text>}, ...]
    """
    # Tokenize into units
    if unit == "sentence":
        units = []
        unit_spans = []
        for s, e in nltk.tokenize.punkt.PunktSentenceTokenizer().span_tokenize(text):
            units.append(text[s:e])
            unit_spans.append((s, e))
    else:  # paragraph
        units = []
        unit_spans = []
        cursor = 0
        for para in text.split("\n\n"):
            para_stripped = para.strip()
            if para_stripped:
                start = text.find(para_stripped, cursor)
                if start != -1:
                    units.append(para_stripped)
                    unit_spans.append((start, start + len(para_stripped)))
                    cursor = start + len(para_stripped)

    if not units:
        return [{"text": text, "offset": 0}]
    if len(units) == 1:
        return [{"text": units[0], "offset": unit_spans[0][0]}]

    # Generate embeddings
    unit_embeddings = model.encode(units, show_progress_bar=False, convert_to_numpy=True)

    # Hysteresis-based merging
    result = []
    cur_units = [units[0]]
    cur_indices = [0]
    cur_embeddings = [unit_embeddings[0]]

    upper = similarity_threshold + hysteresis / 2
    lower = similarity_threshold - hysteresis / 2

    for i in range(1, len(units)):
        avg_emb = np.mean(cur_embeddings, axis=0).reshape(1, -1)
        next_emb = unit_embeddings[i].reshape(1, -1)
        sim = cosine_similarity(avg_emb, next_emb)[0][0]

        potential = " ".join(cur_units + [units[i]])
        would_exceed = len(potential) > max_chunk_chars

        if sim < lower or would_exceed:
            chunk_start = unit_spans[cur_indices[0]][0]
            chunk_end = unit_spans[cur_indices[-1]][1]
            result.append({
                "text": text[chunk_start:chunk_end].strip(),
                "offset": chunk_start,
            })
            cur_units = [units[i]]
            cur_indices = [i]
            cur_embeddings = [unit_embeddings[i]]
        else:
            cur_units.append(units[i])
            cur_indices.append(i)
            cur_embeddings.append(unit_embeddings[i])

    # Last chunk
    if cur_units:
        chunk_start = unit_spans[cur_indices[0]][0]
        chunk_end = unit_spans[cur_indices[-1]][1]
        result.append({
            "text": text[chunk_start:chunk_end].strip(),
            "offset": chunk_start,
        })

    # Post-process: merge tiny chunks with their next neighbour
    merged = []
    i = 0
    while i < len(result):
        cur = result[i]
        while (
            len(cur["text"]) < MIN_CHUNK_CHARS
            and i + 1 < len(result)
            and len(cur["text"]) + len(result[i + 1]["text"]) + 1 <= max_chunk_chars
        ):
            nxt = result[i + 1]
            merged_start = min(cur["offset"], nxt["offset"])
            merged_end = max(
                cur["offset"] + len(cur["text"]),
                nxt["offset"] + len(nxt["text"]),
            )
            cur = {
                "text": text[merged_start:merged_end].strip(),
                "offset": merged_start,
            }
            i += 1
        merged.append(cur)
        i += 1

    return merged
