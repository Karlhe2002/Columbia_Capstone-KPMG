from __future__ import annotations

from pathlib import Path
import re
import csv
import json
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

# =============================================================================
# Purpose
# =============================================================================
# This script scans parsed provider-manual text files (parse_raw) and extracts
# hyperlink occurrences into a single CSV for downstream RAG workflows.
#
# What it produces:
#   - One CSV row per hyperlink occurrence found in the text
#   - Line-level context (the full line containing the URL)
#   - Basic URL metadata (domain, whether it looks like a document link)
#   - Optional page number (best-effort) using companion parsed JSON/JSONL files
#
# Input folders (relative to PROJECT_ROOT):
#   data/raw/pm/parse_raw         -> text files to scan
#   data/raw/pm/parse_raw_json    -> optional companion parsed output with pages
#
# Output folder:
#   data/raw/pm/parse_hyperlink/hyperlinks.csv
# =============================================================================


# =============================================================================
# Paths
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[3]

SRC_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pm" / "parse_raw"
SRC_JSON_DIR = PROJECT_ROOT / "data" / "raw" / "pm" / "parse_raw_json"

OUT_DIR = PROJECT_ROOT / "data" / "raw" / "pm" / "parse_hyperlink"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_FILE = OUT_DIR / "hyperlinks.csv"


# =============================================================================
# Regex patterns
# =============================================================================
# Markdown link format: [anchor](https://example.com)
MD_LINK_RE = re.compile(r'\[([^\]]+)\]\((https?://[^\s)]+)\)')

# HTML anchor format: <a href="https://example.com">anchor</a>
HTML_A_RE = re.compile(
    r'<a\s+(?:[^>]*?)href=[\'"](https?://[^\'"]+)[\'"][^>]*>(.*?)</a>',
    re.IGNORECASE | re.DOTALL
)

# Plain URL format: https://example.com/whatever
URL_RE = re.compile(r'https?://[^\s)<>\]\}\'",]+')

# Whitespace normalizer for cleaned context fields
WHITESPACE_RE = re.compile(r"\s+")

# ASCII control characters (often appear in PDF->text extraction)
CTRL_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

# Unicode blocks that commonly show up as extraction artifacts (CJK/fullwidth/etc.)
CJK_AND_FULLWIDTH_RE = re.compile(
    r"["
    r"\u3000-\u303F"            # CJK Symbols and Punctuation
    r"\u3400-\u4DBF"            # CJK Unified Ideographs Extension A
    r"\u4E00-\u9FFF"            # CJK Unified Ideographs
    r"\uF900-\uFAFF"            # CJK Compatibility Ideographs
    r"\uFF00-\uFFEF"            # Halfwidth and Fullwidth Forms
    r"\u3040-\u309F"            # Hiragana
    r"\u30A0-\u30FF"            # Katakana
    r"\uAC00-\uD7AF"            # Hangul Syllables
    r"\U00020000-\U0002EBEF"    # CJK Extensions B–F (supplementary planes)
    r"]+"
)

# Common tracking parameters to remove during normalization
TRACKING_PARAM_PREFIXES = ("utm_",)
TRACKING_PARAMS_EXACT = {"gclid", "fbclid", "mc_cid", "mc_eid"}

# File extensions that indicate the URL likely points to a document-like artifact
DOC_LIKE_EXT = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt", ".html", ".htm"}


# =============================================================================
# Text cleaning utilities
# =============================================================================
def clean_text(s: str) -> str:
    """Collapse whitespace into single spaces and trim."""
    return WHITESPACE_RE.sub(" ", s).strip()


def strip_cjk_and_controls(s: str, ascii_only: bool = True) -> str:
    """
    Clean line-level context to remove common extraction artifacts.

    Steps:
      1) remove ASCII control chars
      2) remove CJK/fullwidth/compatibility unicode blocks that show up as noise
      3) optionally force ASCII-only to drop remaining non-ASCII glyphs
      4) normalize whitespace

    Note:
      - ascii_only=True is a strong cleaning choice. It removes all non-ASCII
        characters (e.g., “–” becomes removed). If you want to keep punctuation
        like en-dash, set ascii_only=False.
    """
    s = CTRL_RE.sub("", s)
    s = CJK_AND_FULLWIDTH_RE.sub("", s)
    if ascii_only:
        s = s.encode("ascii", "ignore").decode("ascii", errors="ignore")
    return clean_text(s)


# =============================================================================
# URL utilities
# =============================================================================
def normalize_url(url: str) -> str:
    """
    Normalize a URL for consistent metadata extraction / de-duplication.

    Normalization choices:
      - lowercase scheme and host
      - remove common tracking query params (utm_*, gclid, etc.)
      - drop fragment (#...)
    """
    try:
        p = urlparse(url.strip())
        scheme = (p.scheme or "http").lower()
        netloc = (p.netloc or "").lower()

        q = []
        for k, v in parse_qsl(p.query, keep_blank_values=True):
            kl = k.lower()
            if kl in TRACKING_PARAMS_EXACT or any(kl.startswith(pref) for pref in TRACKING_PARAM_PREFIXES):
                continue
            q.append((k, v))
        query = urlencode(q, doseq=True)

        return urlunparse((scheme, netloc, p.path, p.params, query, ""))  # drop fragment
    except Exception:
        return url.strip()


def url_domain(url: str) -> str:
    """Return the URL hostname/domain, lowercased (empty string if parsing fails)."""
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def url_path(url: str) -> str:
    """Return the URL path component (empty string if parsing fails)."""
    try:
        return urlparse(url).path or ""
    except Exception:
        return ""


def is_pdf(url: str) -> bool:
    """Heuristic: treat URL as PDF if the path ends in '.pdf'."""
    return Path(url_path(url)).suffix.lower() == ".pdf"


def is_doc_like(url: str) -> bool:
    """Heuristic: treat URL as document-like if the path ends in a known extension."""
    return Path(url_path(url)).suffix.lower() in DOC_LIKE_EXT


def clean_extracted_url(url: str) -> str:
    """
    Improve URL clickability by trimming punctuation that often gets captured
    adjacent to a URL in text.
    """
    u = (url or "").strip()

    # Remove common trailing punctuation (e.g., "https://x.com/." or "https://x.com/)")
    while u and u[-1] in ".,;:)]}\"'":
        u = u[:-1]

    # Remove common leading punctuation (e.g., "(https://x.com/)")
    while u and u[0] in "([{\"'":
        u = u[1:]

    return u.strip()


# =============================================================================
# Context extraction utility
# =============================================================================
def get_line_text(raw: str, pos: int) -> str:
    """
    Return the full line containing the match position `pos`.
    This provides a clean, human-readable context for the URL in the output CSV.
    """
    start = raw.rfind("\n", 0, pos)
    start = 0 if start == -1 else start + 1
    end = raw.find("\n", pos)
    end = len(raw) if end == -1 else end
    return raw[start:end]


# =============================================================================
# Companion parsed JSON/JSONL support (page number lookup)
# =============================================================================
def find_companion_json(raw_path: Path) -> Path | None:
    """
    Given a raw text file path, try to find a companion parsed file
    in SRC_JSON_DIR with the same stem name.

    Example:
      parse_raw/Pharmacy_Policy_Guidelines.txt
        -> parse_raw_json/Pharmacy_Policy_Guidelines.json or .jsonl
    """
    if not SRC_JSON_DIR.exists():
        return None
    stem = raw_path.stem
    for ext in (".json", ".jsonl"):
        p = SRC_JSON_DIR / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_parsed_pages(json_path: Path) -> list[dict]:
    """
    Load page-level parsed content used to guess page_number for each URL.

    Supported formats:
      - .json:  {"pages":[{"page": 1, "text": "..."} , ...]}
      - .jsonl: each line is {"page": 1, "text": "..."}
    """
    if not json_path or not json_path.exists():
        return []

    try:
        if json_path.suffix.lower() == ".jsonl":
            pages: list[dict] = []
            with json_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "page" in obj and "text" in obj:
                        pages.append({"page": obj["page"], "text": obj["text"]})
            return pages

        obj = json.loads(json_path.read_text(encoding="utf-8", errors="replace"))
        pages = obj.get("pages", [])
        out: list[dict] = []
        for p in pages:
            if isinstance(p, dict) and "page" in p and "text" in p:
                out.append({"page": p["page"], "text": p["text"]})
        return out

    except Exception:
        return []


def guess_page_number(pages: list[dict], url: str) -> int | None:
    """
    Page number guess:
      1) find exact URL in any page text
      2) fallback: find URL domain in any page text

    If no match is found, return None (written as empty in CSV).
    """
    if not pages:
        return None

    u = (url or "").strip()
    if u:
        for p in pages:
            t = p.get("text", "") or ""
            if u in t:
                return p.get("page")

    dom = url_domain(u)
    if dom:
        for p in pages:
            t = p.get("text", "") or ""
            if dom in t:
                return p.get("page")

    return None


# =============================================================================
# Hyperlink extraction
# =============================================================================
def extract_from_text(text: str) -> list[dict]:
    """
    Extract hyperlinks from text in three formats:
      - markdown links
      - html anchor links
      - plain URLs

    Returns a list of dictionaries:
      {pos, end, url, link_type}
    where pos/end are character offsets in the file.
    """
    results: list[dict] = []

    for m in MD_LINK_RE.finditer(text):
        results.append({
            "pos": m.start(0),
            "end": m.end(0),
            "url": m.group(2).strip(),
            "link_type": "markdown"
        })

    for m in HTML_A_RE.finditer(text):
        results.append({
            "pos": m.start(0),
            "end": m.end(0),
            "url": m.group(1).strip(),
            "link_type": "html"
        })

    for m in URL_RE.finditer(text):
        url = m.group(0).strip()
        # De-duplicate when markdown/html match already captured the same URL at ~same position
        if any(abs(m.start() - r["pos"]) < 3 and r["url"] == url for r in results):
            continue
        results.append({
            "pos": m.start(),
            "end": m.end(),
            "url": url,
            "link_type": "plain_url"
        })

    results.sort(key=lambda x: x["pos"])
    return results


# =============================================================================
# File processing
# =============================================================================
def process_file(path: Path, writer: csv.DictWriter, relbase: Path) -> int:
    """
    Extract URLs from a single file, then write one CSV row per URL occurrence.
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return 0

    # Load companion parsed pages (optional) for page_number guesses
    pages: list[dict] = []
    json_path = find_companion_json(path)
    if json_path:
        pages = load_parsed_pages(json_path)

    hits = extract_from_text(raw)

    for idx, h in enumerate(hits, start=1):
        pos = h["pos"]
        end = h["end"]
        url_raw = h["url"]
        link_type = h["link_type"]

        # Convert character offset -> line number (1-based)
        line_no = raw.count("\n", 0, pos) + 1

        # Extract and clean the line containing this URL
        line_text_raw = get_line_text(raw, pos)
        line_text = strip_cjk_and_controls(line_text_raw, ascii_only=True)

        # Clean URL for clickability and normalize for metadata checks
        url_clean = clean_extracted_url(url_raw)
        url_norm = normalize_url(url_clean)

        dom = url_domain(url_norm)
        page_no = guess_page_number(pages, url=url_clean)

        writer.writerow({
            "source": str(path.relative_to(relbase)),
            "file_ext": path.suffix.lower(),

            "occurrence_index": idx,
            "char_start": pos,
            "char_end": end,
            "line_number": line_no,
            "page_number": "" if page_no is None else page_no,

            "link_type": link_type,
            "url": url_clean,
            "url_domain": dom,

            "is_pdf": "1" if is_pdf(url_norm) else "0",
            "is_doc_like": "1" if is_doc_like(url_norm) else "0",

            "line_text": line_text,
        })

    return len(hits)


# =============================================================================
# Main
# =============================================================================
def main():
    """Scan all files under SRC_RAW_DIR and write hyperlinks.csv."""
    if not SRC_RAW_DIR.exists():
        print(f"Source directory not found: {SRC_RAW_DIR}")
        return

    relbase = SRC_RAW_DIR

    fieldnames = [
        "source", "file_ext",
        "occurrence_index", "char_start", "char_end", "line_number", "page_number",
        "link_type", "url", "url_domain",
        "is_pdf", "is_doc_like",
        "line_text",
    ]

    total = 0
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for p in sorted(SRC_RAW_DIR.rglob("*")):
            if p.is_file():
                total += process_file(p, writer, relbase)

    print(f"Wrote {total} hyperlink occurrences to {OUT_FILE}")


if __name__ == "__main__":
    main()