"""
Convert **compare-style** CSV test results to JSON format expected by
scripts/llm_evaluation_compare.py.

Expected CSV columns:
- Test Question
- answers (preferred, new format)
- Test Result (fallback)

Output JSON schema per entry:
{
  "query_id": "csv_test_001",
  "question": "...",
  "concept": "",
  "compare_sections": {
        "headline_summary": "...",
    "policy_definition": "...",
    "provider_manual_definition": "...",
        "comparison": ["..."],
        "evidence_quoted": ["..."],
    "caveats": "..." | null
  },
  "retrieved_docs": {
    "policy": [],
    "provider_manual": []
  },
    "answer": "raw markdown answer text"
}

Note:
- The parser is title-based for the new markdown format:
    **Headline Summary of the 2 sources**,
    **Policy Definition**,
    **Provider Manual Definition**,
    **Comparison (similarities and differences)**,
    **Evidence (quoted)**,
    **Caveats (if any)**.
- `comparison` is kept as-is from bullets under the comparison section.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _normalize_answer(text: str) -> str:
    s = text.strip()
    if s.startswith("Answer:\n"):
        return s[len("Answer:\n") :].strip()
    if s == "Answer:":
        return ""
    if s.startswith("Answer:"):
        return s[len("Answer:") :].strip()
    # New format from generation step may return a JSON string field with
    # heading-rich markdown directly, starting from the first heading.
    return s


def _find_first_marker(text: str, markers: List[str], start_at: int = 0) -> Optional[Tuple[int, str]]:
    matches: List[Tuple[int, str]] = []
    for marker in markers:
        idx = text.find(marker, start_at)
        if idx != -1:
            matches.append((idx, marker))
    if not matches:
        return None
    return min(matches, key=lambda x: x[0])


def _section(text: str, start_markers: List[str], end_markers: List[str]) -> str:
    start_match = _find_first_marker(text, start_markers)
    if not start_match:
        return ""

    start_idx, start_marker = start_match
    start_idx += len(start_marker)

    end_match = _find_first_marker(text, end_markers, start_idx)
    end_idx = end_match[0] if end_match else len(text)

    return text[start_idx:end_idx].strip()


def _flatten_block(block: str) -> str:
    lines: List[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("* "):
            s = s[2:].strip()
        lines.append(s)
    return "\n".join(lines).strip()


def _clean_inline_item(text: str) -> str:
    s = text.strip()
    s = re.sub(r"^[*\-]\s*", "", s)
    return s.strip()


def _extract_bullets(block: str) -> List[str]:
    items: List[str] = []
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("*") or s.startswith("-"):
            item = _clean_inline_item(s)
            if item:
                items.append(item)
    return items


def _parse_compare_sections(answer: str) -> Dict[str, object]:
    answer = _normalize_answer(answer)

    headline_block = _section(
        answer,
        ["**Headline Summary of the 2 sources**"],
        ["**Policy Definition**"],
    )

    policy_block = _section(
        answer,
        ["**Policy Definition**"],
        ["**Provider Manual Definition**"],
    )
    provider_block = _section(
        answer,
        ["**Provider Manual Definition**"],
        ["**Comparison (similarities and differences)**"],
    )
    comparison_block = _section(
        answer,
        ["**Comparison (similarities and differences)**"],
        [
            "**Evidence (quoted)**",
            "**Caveats (if any)**",
            "📚 Retrieved Sources",
            "Retrieved Sources",
        ],
    )

    evidence_block = _section(
        answer,
        ["**Evidence (quoted)**"],
        ["**Caveats (if any)**", "📚 Retrieved Sources", "Retrieved Sources"],
    )

    comparison_items = _extract_bullets(comparison_block)

    caveats = _section(
        answer,
        ["**Caveats (if any)**"],
        ["📚 Retrieved Sources", "Retrieved Sources"],
    ) or None

    return {
        "headline_summary": _flatten_block(headline_block),
        "policy_definition": _flatten_block(policy_block),
        "provider_manual_definition": _flatten_block(provider_block),
        "comparison": comparison_items,
        "evidence_quoted": _extract_bullets(evidence_block),
        "caveats": caveats,
    }


def convert_csv_to_compare_json(input_csv: Path, output_json: Path) -> int:
    output_json.parent.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict[str, object]] = {}

    with input_csv.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for idx, row in enumerate(reader, 1):
            question = (row.get("Test Question") or "").strip()
            # New format uses `answers` as the main answer field.
            # Keep fallback to `Test Result` to avoid breaking mixed CSVs.
            test_result = (row.get("answers") or row.get("Test Result") or "").strip()

            if not question and not test_result:
                continue

            test_id = f"csv_test_{idx:03d}"
            compare_sections = _parse_compare_sections(test_result)

            results[test_id] = {
                "query_id": test_id,
                "question": question,
                "concept": "",
                "compare_sections": compare_sections,
                "retrieved_docs": {
                    "policy": [],
                    "provider_manual": [],
                },
                "answer": test_result,
            }

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    return len(results)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert compare CSV results to llm_evaluation_compare JSON input",
    )
    parser.add_argument(
        "-i",
        "--input_csv",
        required=True,
        help="Path to CSV (e.g. data/rag system test - Sheet1.csv)",
    )
    parser.add_argument(
        "-o",
        "--output_json",
        required=True,
        help="Path to output JSON (e.g. data/compare_results/sheet1_compare_results.json)",
    )

    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_json = Path(args.output_json)

    if not input_csv.exists():
        print(f"Error: Input CSV not found: {input_csv}")
        return 1

    count = convert_csv_to_compare_json(input_csv, output_json)
    print(f"Converted {count} rows -> {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
