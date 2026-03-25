# JSON generated from src/healthcare_rag_llm/testing/generate_test_result.py
"""
Columbia_Capstone-KPMG/.venv/bin/python scripts/convert_json_to_json_compare.py \
  -i your_input_file.json \
  -o data/comparison_llm_eval_cases/your_output_file.json
"""

"""
Convert compare test-results JSON to compare-evaluation JSON.

Mapping:
- query_content -> question (Q)
- answers -> answer (A)

Input example item:
{
  "test_id_1": {
    "query_id": "Test_query_compare_1",
    "query_content": "...",
    "concept": "...",
    "retrieved_docs": {"policy": [...], "provider_manual": [...]},
    "answers": "**Headline Summary ..."
  }
}

Output example item:
{
  "test_id_1": {
    "query_id": "Test_query_compare_1",
    "question": "...",
    "concept": "...",
    "compare_sections": {
      "headline_summary": "...",
      "policy_definition": "...",
      "provider_manual_definition": "...",
      "comparison": ["..."],
      "evidence_quoted": ["..."],
      "caveats": "..."
    },
    "retrieved_docs": {"policy": [...], "provider_manual": [...]},
    "answer": "..."
  }
}
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _normalize_answer(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("Answer:\n"):
        return s[len("Answer:\n") :].strip()
    if s == "Answer:":
        return ""
    if s.startswith("Answer:"):
        return s[len("Answer:") :].strip()
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


def _parse_compare_sections(answer: str) -> Dict[str, Any]:
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
    caveats = _section(
        answer,
        ["**Caveats (if any)**"],
        ["📚 Retrieved Sources", "Retrieved Sources"],
    ) or None

    return {
        "headline_summary": _flatten_block(headline_block),
        "policy_definition": _flatten_block(policy_block),
        "provider_manual_definition": _flatten_block(provider_block),
        "comparison": _extract_bullets(comparison_block),
        "evidence_quoted": _extract_bullets(evidence_block),
        "caveats": caveats,
    }


def convert_json_to_json(input_json: Path, output_json: Path) -> int:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object keyed by test_id")

    out: Dict[str, Dict[str, Any]] = {}

    for test_id, item in data.items():
        if not isinstance(item, dict):
            continue

        query_id = item.get("query_id", test_id)
        question = item.get("query_content") or item.get("question") or ""
        answer_text = item.get("answers") or item.get("answer") or ""

        retrieved_docs = item.get("retrieved_docs")
        if not isinstance(retrieved_docs, dict):
            retrieved_docs = {"policy": [], "provider_manual": []}
        else:
            retrieved_docs = {
                "policy": retrieved_docs.get("policy", []),
                "provider_manual": retrieved_docs.get("provider_manual", []),
            }

        out[test_id] = {
            "query_id": query_id,
            "question": question,
            "concept": item.get("concept", ""),
            "compare_sections": _parse_compare_sections(str(answer_text)),
            "retrieved_docs": retrieved_docs,
            "answer": str(answer_text),
        }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert compare test-results JSON to compare-evaluation JSON")
    parser.add_argument("-i", "--input_json", required=True, help="Path to source JSON")
    parser.add_argument("-o", "--output_json", required=True, help="Path to converted JSON")
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)

    if not input_json.exists():
        print(f"Error: Input JSON not found: {input_json}")
        return 1

    count = convert_json_to_json(input_json, output_json)
    print(f"Converted {count} items -> {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
