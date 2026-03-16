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
    txt_path: str,
    output_dir: str,
    doc_id: str = "",
    category: str = "pharmacy",
    max_chunk_chars: int = 1200,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    unit: str = "sentence",
    similarity_threshold: float = 0.35,
    hysteresis: float = 0.02,
    preset: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """
    Two-level hybrid chunking:

      Level 1 — Split at ToC section/subsection boundaries (hard splits).
      Level 2 — Within each section, apply semantic chunking (hysteresis-
                based embedding similarity) to further split large sections.

    Sections shorter than *max_chunk_chars* are kept as a single chunk.

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

    # Auto-detect or use specified preset
    config = _get_config(raw_text, preset)

    # ── 1. Parse ToC ──
    titles = _parse_toc(raw_text)
    if not titles:
        if verbose:
            print(f"[SKIP] {txt_path.name} (no ToC entries found)")
        return

    # ── 2. Page marker map (before cleaning) ──
    page_markers = _build_page_marker_map(raw_text, config["page_marker_regex"])

    # ── 3. Locate body start ──
    body_start = _find_body_start(raw_text, titles)
    body_raw = raw_text[body_start:]

    # ── 4. Clean page markers (replace with \n\n to preserve paragraph breaks) ──
    body_clean = _clean_page_markers_preserve_breaks(body_raw, config)

    # ── 5. Find section split points ──
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

    # ── 6. Load sentence-transformer model ──
    if verbose:
        print(f"  Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    # ── 7. For each section, apply semantic sub-chunking ──
    chunks: List[Dict] = []
    chunk_idx = 0

    for i, (title, start) in enumerate(split_points):
        end = split_points[i + 1][1] if i + 1 < len(split_points) else len(body_clean)
        section_text = body_clean[start:end].strip()

        if not section_text:
            continue

        raw_start = body_start + start
        raw_end = body_start + end

        if len(section_text) <= max_chunk_chars:
            # Short section → single chunk, no semantic split
            pages = _estimate_pages(page_markers, raw_start, raw_end, len(raw_text))
            rec = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                "char_start": start,
                "char_end": end,
                "pages": pages,
                "text": section_text,
                "chunk_type": "text",
                "category": category,
                "section_title": title,
            }
            chunks.append(rec)
            chunk_idx += 1
        else:
            # Large section → semantic sub-chunking
            sub_chunks = _semantic_split(
                section_text, model, unit,
                similarity_threshold, hysteresis, max_chunk_chars,
            )
            for sc in sub_chunks:
                sc_raw_start = raw_start + sc["offset"]
                sc_raw_end = raw_start + sc["offset"] + len(sc["text"])
                pages = _estimate_pages(
                    page_markers, sc_raw_start, sc_raw_end, len(raw_text)
                )
                rec = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}::{str(chunk_idx).zfill(4)}",
                    "char_start": start + sc["offset"],
                    "char_end": start + sc["offset"] + len(sc["text"]),
                    "pages": pages,
                    "text": sc["text"],
                    "chunk_type": "text",
                    "category": category,
                    "section_title": title,
                }
                chunks.append(rec)
                chunk_idx += 1

    # ── 8. Write JSONL ──
    stem = txt_path.stem
    out_file = output_dir / f"{stem}.chunks.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for rec in chunks:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"[OK]  {txt_path.name} -> {len(chunks)} chunks -> {out_file}")


# ── helpers ──────────────────────────────────────────────────────────────


def _clean_page_markers_preserve_breaks(text: str, config: dict) -> str:
    """Remove page footers/headers but replace each page break with ``\\n\\n``
    so that paragraph boundaries survive for sentence/paragraph tokenisers."""
    for pattern in config.get("footer_patterns", []):
        text = re.sub(pattern, "\n\n", text)

    for pattern in config.get("header_patterns", []):
        text = re.sub(pattern, "\n\n", text, flags=re.MULTILINE)

    for pattern in config.get("artifacts", []):
        text = re.sub(pattern, "", text)

    # Collapse 3+ newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


MIN_CHUNK_CHARS = 80  # chunks smaller than this get merged with their neighbour


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
            # Split: save current chunk
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
            # Merge
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
        # If current chunk is too small and there is a next chunk, merge forward
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
