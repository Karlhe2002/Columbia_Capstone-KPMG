# input generated from src/healthcare_rag_llm/testing/generate_test_result.py
"""
.venv/bin/python scripts/convert_raw_to_processed_compare.py \
  -i data/llm_eval_results_compare/comparison_raw/input_file.json \
    -o data/llm_eval_results_compare/comparison_processed/input_file_processed.json \
    -g "data/llm_eval_results_compare/Q&A_ground_truth.csv"
"""

"""
Convert compare test-results JSON to compare-evaluation JSON.

Mapping:
- query_content -> question (Q)
- answers -> answers (structured A)

Input example item:
{
  "test_id_1": {
    "query_id": "Test_query_compare_1",
    "query_content": "...",
    "concept": "...",
    "retrieved_docs": {"policy": [...], "provider_manual": [...]},
    "answers": "..."
  }
}

Output example item:
{
  "test_id_1": {
    "query_id": "Test_query_compare_1",
    "question": "...",
    "concept": "...",
        "answers": {
            "headline_summary": "...",
            "policy_definition": "...",
            "provider_manual_definition": "...",
            "similarities": ["..."],
            "differences": ["..."],
            "caveats": "..."
        },
    "retrieved_docs": {"policy": [...], "provider_manual": [...]},
    "answer": "...",
    "ground_truth": "..."
  }
}
"""

import argparse
import json
import re
import csv
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


def _extract_comparison_items(block: str) -> List[str]:
    """Extract comparison items, supporting both bullet and paragraph formats."""
    bullet_items = _extract_bullets(block)
    if bullet_items:
        return bullet_items

    paragraph = _flatten_block(block)
    return [paragraph] if paragraph else []


def _extract_evidence(block: str) -> List[str]:
    """Extract evidence items (lines starting with [ or contain :)."""
    items: List[str] = []
    for line in block.splitlines():
        s = line.strip()
        if not s:
            continue
        # Keep lines that look like evidence: start with [ or contain citation pattern
        if s.startswith("[") or ("]: " in s and "[" in s):
            items.append(s)
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


def _sanitize_chunk(chunk: Any) -> Any:
    """Remove retrieval scoring/ranking metadata from a chunk dict."""
    if not isinstance(chunk, dict):
        return chunk

    cleaned = dict(chunk)
    for key in ["score", "rerank_score", "final_score", "rank"]:
        cleaned.pop(key, None)
    return cleaned


def _normalize_query(text: str) -> str:
    """Normalize query text for matching across JSON and CSV sources."""
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _normalize_ground_truth_answer(text: str) -> str:
    """Normalize CSV answer text while preserving multiline content."""
    return (text or "").strip()


def _to_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    return [str(value).strip()] if str(value).strip() else []


def _extract_sentence_by_keyword(text: str, keyword: str) -> str:
    """Return first sentence containing keyword (case-insensitive)."""
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    key = keyword.lower()
    for part in parts:
        sentence = part.strip()
        if sentence and key in sentence.lower():
            return sentence
    return ""


def _get_with_alias(d: Dict[str, Any], primary_key: str, alias_keys: List[str]) -> str:
    """Get value from dict trying primary key first, then alias keys. Return stripped string."""
    result = d.get(primary_key)
    if result:
        return str(result).strip()
    for alias_key in alias_keys:
        result = d.get(alias_key)
        if result:
            return str(result).strip()
    return ""


def _parse_answer_payload(answer: Any) -> Optional[Dict[str, Any]]:
    if isinstance(answer, dict):
        return answer
    if not isinstance(answer, str):
        return None

    text = _normalize_answer(answer)
    if not text:
        return None

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None

    return parsed if isinstance(parsed, dict) else None


def _load_ground_truth_csv(csv_path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    """Load query->ground-truth mappings from the ground-truth CSV."""
    if csv_path is None or not csv_path.exists():
        return {}

    ground_truth: Dict[str, Dict[str, str]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = _normalize_query(row.get("Query", ""))
            answer = _normalize_ground_truth_answer(row.get("Answer", ""))
            pm_ground_truth = _normalize_ground_truth_answer(row.get("provider_manual_definition", ""))
            policy_ground_truth = _normalize_ground_truth_answer(row.get("policy_definition", ""))
            if not query:
                continue
            if not answer and not pm_ground_truth and not policy_ground_truth:
                continue
            # Keep the first non-empty answer for duplicate queries.
            ground_truth.setdefault(
                query,
                {
                    "ground_truth": answer,
                    "pm_ground_truth": pm_ground_truth,
                    "policy_ground_truth": policy_ground_truth,
                },
            )

    return ground_truth


def _load_ground_truth_rows(csv_path: Optional[Path]) -> List[Tuple[str, Dict[str, str]]]:
    """Load ordered (query, ground_truth_fields) rows from the ground-truth CSV."""
    if csv_path is None or not csv_path.exists():
        return []

    rows: List[Tuple[str, Dict[str, str]]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = _normalize_query(row.get("Query", ""))
            answer = _normalize_ground_truth_answer(row.get("Answer", ""))
            pm_ground_truth = _normalize_ground_truth_answer(row.get("provider_manual_definition", ""))
            policy_ground_truth = _normalize_ground_truth_answer(row.get("policy_definition", ""))
            if not query:
                continue
            if not answer and not pm_ground_truth and not policy_ground_truth:
                continue
            rows.append(
                (
                    query,
                    {
                        "ground_truth": answer,
                        "pm_ground_truth": pm_ground_truth,
                        "policy_ground_truth": policy_ground_truth,
                    },
                )
            )

    return rows


def _normalize_answers_payload(answer: Any) -> Dict[str, Any]:
    payload = _parse_answer_payload(answer)

    if payload:
        similarities = _to_string_list(payload.get("similarities", []))
        differences = _to_string_list(payload.get("differences", []))
        comparison = _to_string_list(payload.get("comparison", []))
        if not similarities and not differences and comparison:
            if len(comparison) >= 1:
                similarities = [comparison[0]]
            if len(comparison) >= 2:
                differences = [comparison[1]]
            if len(comparison) > 2:
                differences.extend(comparison[2:])

        if not comparison:
            comparison = similarities + differences

        # Try to extract policy & provider_manual from aligned_pairs if available
        policy_def = _get_with_alias(payload, "policy_definition", ["policy_update"])
        provider_manual_def = _get_with_alias(payload, "provider_manual_definition", ["provider_manual"])
        
        # If not found at top level, try to extract from aligned_pairs
        if not policy_def and not provider_manual_def:
            aligned_pairs = payload.get("aligned_pairs", [])
            if aligned_pairs and isinstance(aligned_pairs, list) and len(aligned_pairs) > 0:
                first_pair = aligned_pairs[0]
                if isinstance(first_pair, dict):
                    policy_def = first_pair.get("policy_update", "") or first_pair.get("policy_definition", "")
                    provider_manual_def = first_pair.get("provider_manual", "") or first_pair.get("provider_manual_definition", "")

        sections = {
            "headline_summary": str(payload.get("headline_summary", "")).strip(),
            "policy_definition": policy_def,
            "provider_manual_definition": provider_manual_def,
            "similarities": similarities,
            "differences": differences,
            "comparison": comparison,
            "caveats": str(payload.get("caveats", "")).strip() or None,
        }
        return sections

    return {}


def convert_json_to_json(
    input_json: Path,
    output_json: Path,
    ground_truth_csv: Optional[Path] = None,
) -> int:
    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("Input JSON must be an object keyed by test_id")

    if ground_truth_csv is None:
        ground_truth_csv = input_json.parent.parent / "Q&A_ground_truth.csv"
    ground_truth_map = _load_ground_truth_csv(ground_truth_csv)
    ground_truth_rows = _load_ground_truth_rows(ground_truth_csv)

    out: Dict[str, Dict[str, Any]] = {}

    fallback_group_size = 5

    for index, (test_id, item) in enumerate(data.items()):
        if not isinstance(item, dict):
            continue

        query_id = item.get("query_id", test_id)
        question = item.get("query_content") or item.get("question") or ""
        answer_text = item.get("answers") or item.get("answer") or ""
        normalized_question = _normalize_query(question)

        retrieved_docs = item.get("retrieved_docs")
        if not isinstance(retrieved_docs, dict):
            retrieved_docs = {"policy": [], "provider_manual": []}
        else:
            retrieved_docs = {
                "policy": [_sanitize_chunk(c) for c in retrieved_docs.get("policy", [])],
                "provider_manual": [_sanitize_chunk(c) for c in retrieved_docs.get("provider_manual", [])],
            }

        output_item: Dict[str, Any] = {
            "query_id": query_id,
            "question": question,
            "concept": item.get("concept", ""),
            "answers": _normalize_answers_payload(answer_text),
            "retrieved_docs": retrieved_docs,
            "answer": json.dumps(answer_text, ensure_ascii=False) if isinstance(answer_text, dict) else str(answer_text),
        }

        ground_truth_fields = ground_truth_map.get(normalized_question)
        if not ground_truth_fields and ground_truth_rows:
            fallback_index = index // fallback_group_size
            if fallback_index < len(ground_truth_rows):
                ground_truth_fields = ground_truth_rows[fallback_index][1]
        if ground_truth_fields:
            ground_truth_answer = ground_truth_fields.get("ground_truth", "")
            pm_ground_truth = ground_truth_fields.get("pm_ground_truth", "")
            policy_ground_truth = ground_truth_fields.get("policy_ground_truth", "")

            if ground_truth_answer:
                output_item["ground_truth"] = ground_truth_answer
            if pm_ground_truth:
                output_item["pm_ground_truth"] = pm_ground_truth
            if policy_ground_truth:
                output_item["policy_ground_truth"] = policy_ground_truth

        out[test_id] = output_item

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    return len(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert compare test-results JSON to compare-evaluation JSON")
    parser.add_argument("-i", "--input_json", required=True, help="Path to source JSON")
    parser.add_argument("-o", "--output_json", required=True, help="Path to converted JSON")
    parser.add_argument(
        "-g",
        "--ground_truth_csv",
        default=None,
        help="Optional path to a CSV with Query,Answer columns for adding ground_truth.",
    )
    args = parser.parse_args()

    input_json = Path(args.input_json)
    output_json = Path(args.output_json)
    ground_truth_csv = Path(args.ground_truth_csv) if args.ground_truth_csv else None

    if not input_json.exists():
        print(f"Error: Input JSON not found: {input_json}")
        return 1

    count = convert_json_to_json(input_json, output_json, ground_truth_csv=ground_truth_csv)
    print(f"Converted {count} items -> {output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
