"""
LLM-based evaluation for answer_compare_definitions outputs.

1. Policy side: evaluate policy_definition against policy chunks
2. Provider side: evaluate provider_manual_definition against provider chunks
3. Query alignment: evaluate comparison similarities/differences vs query
4. Compare quality: evaluate comparison similarities/differences vs chunks (accuracy_sim, accuracy_diff, correctness, completeness, balance)
"""

import json
from typing import Dict, List, Any, Optional
from pathlib import Path

from healthcare_rag_llm.evaluate.llm_evaluate import LLMEvaluator
from healthcare_rag_llm.llm.llm_client import LLMClient

def _clamp_score(value: Any) -> float:
    """Clamp any numeric-like score to [0.0, 1.0]."""
    return max(0.0, min(1.0, float(value)))


def _derive_score_from_fields(data: Dict[str, Any], fields: List[str]) -> Optional[float]:
    """Compute mean score from available numeric fields in [0,1]."""
    vals: List[float] = []
    for key in fields:
        if key not in data:
            continue
        try:
            vals.append(_clamp_score(data[key]))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _parse_json_response(
    response: str,
    metric_name: str,
    require_score: bool = False,
    fallback: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Parse a JSON object from LLM output with optional score validation."""
    try:
        cleaned = str(response).replace("\t", " ").replace("\r", "")
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx:end_idx]
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                result = json.loads(json_str, strict=False)
            if require_score:
                if "score" not in result:
                    raise ValueError("Missing 'score' field")
                result["score"] = _clamp_score(result["score"])
            return result
        raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"  Warning: Failed to parse {metric_name} response: {e}")
        if fallback is not None:
            return fallback
        return {
            "score": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}",
            "error": str(e),
        }


def _to_string_list(value: Any) -> List[str]:
    """Normalize a value into a list[str]."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, list):
        out: List[str] = []
        for item in value:
            if item is None:
                continue
            item_s = str(item).strip()
            if item_s:
                out.append(item_s)
        return out
    return [str(value)]


def _normalize_retrieved_chunks(raw_chunks: Any) -> List[Dict[str, Any]]:
    """Normalize retrieved chunks into dicts centered on each item's text field."""
    if not isinstance(raw_chunks, list):
        return []

    normalized: List[Dict[str, Any]] = []
    for item in raw_chunks:
        text = ""
        doc_id = "unknown"
        pages: Any = "?"

        if isinstance(item, dict):
            text = str(item.get("text", "") or "").strip()
            doc_id = str(item.get("doc_id", "unknown") or "unknown")
            pages = item.get("pages", "?")
        elif isinstance(item, str):
            text = item.strip()

        if text:
            normalized.append(
                {
                    "text": text,
                    "doc_id": doc_id,
                    "pages": pages,
                }
            )

    return normalized


def _evaluate_query_alignment(
    llm_client: LLMClient,
    query: str,
    concept: str,
    similarities: List[str],
    differences: List[str],
) -> Dict[str, Any]:
    """
    Evaluate whether comparison similarities/differences align with the query.
    """
    target = concept or "the requested concept"
    sim_text = "\n".join(f"  - {s}" for s in similarities) if similarities else "(none)"
    diff_text = "\n".join(f"  - {d}" for d in differences) if differences else "(none)"

    prompt = f"""You are an expert evaluator assessing whether a comparison summary is relevant to a user query.

QUERY:
{query}

TARGET CONCEPT:
{target}

COMPARISON SUMMARY (similarities / differences):
SIMILARITIES:
{sim_text}

DIFFERENCES:
{diff_text}

TASK:
Evaluate whether the listed similarities/differences address the asked concept and user intent.

Rate comparison alignment on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly answers the query
- 0.8-0.9: Mostly relevant, minor gaps
- 0.5-0.7: Partially relevant, some off-topic content
- 0.3-0.4: Low relevance, mostly off-topic
- 0.0-0.2: Not relevant, unrelated

Respond in JSON format:
{{
    "comparison_query_relevance": <float between 0 and 1>,
    "reasoning": "<brief explanation>",
    "off_topic_points": ["<optional off-topic or mismatched points>"]
}}"""
    response = llm_client.chat(prompt)
    result = _parse_json_response(response, "query_alignment", require_score=False)
    if "comparison_query_relevance" in result:
        result["comparison_query_relevance"] = _clamp_score(result["comparison_query_relevance"])
    result.pop("score", None)
    return result


def _evaluate_compare_quality(
    llm_client: LLMClient,
    query: str,
    concept: str,
    similarities: List[str],
    differences: List[str],
    policy_chunks: List[Dict[str, Any]],
    provider_chunks: List[Dict[str, Any]],
    include_correctness: bool,
) -> Dict[str, Any]:
    """
    Single-function evaluation for comparison quality against chunks,
    using separated similarity/difference accuracy scores.
    """
    target = concept or "the requested concept"

    sim_text = "\n".join(f"  - {s}" for s in similarities) if similarities else "(none)"
    diff_text = "\n".join(f"  - {d}" for d in differences) if differences else "(none)"
    policy_ctx = "\n\n".join(
        f"Chunk {i+1}: {chunk.get('text', '')[:500]}..."
        for i, chunk in enumerate(policy_chunks[:5])
    ) if policy_chunks else "(no policy chunks)"
    provider_ctx = "\n\n".join(
        f"Chunk {i+1}: {chunk.get('text', '')[:500]}..."
        for i, chunk in enumerate(provider_chunks[:5])
    ) if provider_chunks else "(no provider chunks)"

    correctness_task = "3. CORRECTNESS: Are claims factually correct given query and source chunks?" if include_correctness else ""
    completeness_idx = 4 if include_correctness else 3
    balance_idx = 5 if include_correctness else 4
    correctness_json = '    "correctness": <float between 0 and 1>,\n' if include_correctness else ""

    prompt = f"""You are an expert evaluator assessing comparison quality in ONE pass.

QUERY:
{query}

TARGET CONCEPT:
{target}

LLM-IDENTIFIED SIMILARITIES:
{sim_text}

LLM-IDENTIFIED DIFFERENCES:
{diff_text}

POLICY SOURCE CHUNKS (excerpts):
{policy_ctx}

PROVIDER MANUAL SOURCE CHUNKS (excerpts):
{provider_ctx}

TASK:
Evaluate the quality of the similarities and differences in the summary in a single step. Consider:
1. ACCURACY_SIM: Are the similarities supported by source chunks?
2. ACCURACY_DIFF: Are the differences supported by source chunks?
{correctness_task}
{completeness_idx}. COMPLETENESS: Are major similarities or differences missing?
{balance_idx}. BALANCE: Is the comparison balanced across sources?

Rate the compare quality on a scale of 0.0 to 1.0:
- 1.0: Similarities and differences are accurate, well-grounded, and complete
- 0.8-0.9: Minor inaccuracies or omissions
- 0.5-0.7: Some similarities/differences unsupported or incorrect
- 0.3-0.4: Significant errors or omissions
- 0.0-0.2: Largely inaccurate or ungrounded

Respond in JSON format:
{{
    "accuracy_sim": <float between 0 and 1>,
    "accuracy_diff": <float between 0 and 1>,
{correctness_json}    "completeness": <float between 0 and 1>,
    "balance": <float between 0 and 1>,
    "reasoning": "<explanation of compare quality assessment>",
    "inaccurate_points": ["<unsupported or wrong points>"],
    "missing_points": ["<important missing points>"]
}}
"""
    response = llm_client.chat(prompt)
    result = _parse_json_response(response, "compare_quality", require_score=False)
    result.pop("score", None)
    score_keys = ["accuracy_sim", "accuracy_diff", "completeness", "balance"]
    if include_correctness:
        score_keys.append("correctness")
    for key in score_keys:
        if key in result:
            result[key] = _clamp_score(result[key])
    if not include_correctness:
        result.pop("correctness", None)
    result["inaccurate_points"] = _to_string_list(result.get("inaccurate_points", []))
    result["missing_points"] = _to_string_list(result.get("missing_points", []))
    return result

def _extract_compare_parts(test_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract policy_definition, provider_manual_definition, policy_chunks, provider_chunks
    from compare_definitions output. Falls back to parsing raw answer if compare_sections
    not present.
    """
    compare_sections = test_entry.get("compare_sections")
    if not compare_sections:
        raw = test_entry.get("answer") or test_entry.get("answers", "")
        if isinstance(raw, str):
            try:
                compare_sections = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        if not compare_sections or not isinstance(compare_sections, dict):
            return None
    policy_def = compare_sections.get("policy_definition", "")
    provider_def = compare_sections.get("provider_manual_definition", "")
    sim = _to_string_list(compare_sections.get("similarities", []))
    diff = _to_string_list(compare_sections.get("differences", []))
    comparison = _to_string_list(compare_sections.get("comparison", []))
    # Current format: comparison[0] is similarity, comparison[1] is difference.
    # Keep backward compatibility when explicit similarities/differences are absent.
    if not sim and not diff and comparison:
        if len(comparison) >= 1:
            sim = [comparison[0]]
        if len(comparison) >= 2:
            diff = [comparison[1]]
        if len(comparison) > 2:
            diff.extend(comparison[2:])
    retrieved = test_entry.get("retrieved_docs", {})
    policy_chunks = _normalize_retrieved_chunks(retrieved.get("policy", []))
    provider_chunks = _normalize_retrieved_chunks(retrieved.get("provider_manual", []))
    return {
        "policy_definition": str(policy_def) if policy_def else "",
        "provider_manual_definition": str(provider_def) if provider_def else "",
        "similarities": list(sim),
        "differences": list(diff),
        "comparison": list(comparison),
        "policy_chunks": list(policy_chunks),
        "provider_chunks": list(provider_chunks),
    }


def evaluate_compare_test_results(
    compare_results_path: str,
    output_path: str,
    llm_client: LLMClient,
    ground_truth_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate answer_compare_definitions results:
    1. Policy side: faithfulness, answer_relevance, correctness (reuse LLMEvaluator)
    2. Provider side: same metrics
    3. Query alignment: policy def, provider def, and comparison vs query
    4. Compare quality: similarities/differences accuracy_sim, accuracy_diff, correctness, completeness, balance.
    """
    print(f"Loading compare results from: {compare_results_path}")
    with open(compare_results_path, "r", encoding="utf-8") as f:
        compare_results = json.load(f)

    ground_truth: Dict[str, Any] = {}
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"Loading ground truth from: {ground_truth_path}")
        with open(ground_truth_path, "r", encoding="utf-8") as f:
            gt_data = json.load(f)
            ground_truth = gt_data if isinstance(gt_data, dict) else {}
        print(f"Loaded {len(ground_truth)} ground truth entries")

    evaluator = LLMEvaluator(llm_client)
    evaluation_results = {
        "metadata": {
            "compare_results_file": compare_results_path,
            "ground_truth_file": ground_truth_path,
            "evaluator_model": llm_client.model,
            "total_tests": len(compare_results),
            "mode": "compare_definitions",
        },
        "results": {},
        "summary": {},
    }

    test_items = [(k, v) for k, v in compare_results.items() if isinstance(v, dict)]
    if limit:
        test_items = test_items[:limit]
        print(f"Limiting evaluation to {limit} tests")

    scores_by_metric: Dict[str, List[float]] = {
        "policy_faithfulness": [],
        "policy_answer_relevance": [],
        "policy_correctness": [],
        "provider_faithfulness": [],
        "provider_answer_relevance": [],
        "provider_correctness": [],
        "query_alignment": [],
        "compare_quality": [],
        "overall": [],
    }

    for i, (test_id, test_data) in enumerate(test_items, 1):
        query_id = test_data.get("query_id", test_id)
        print(f"\n[{i}/{len(test_items)}] Evaluating {test_id} (query: {query_id})")

        query = test_data.get("question") or test_data.get("query_content") or query_id
        concept = test_data.get("concept", "") or ""
        if concept:
            query_full = f"{query}\n\nTarget concept: {concept}"
        else:
            query_full = query

        gt_answer = None
        if query_id in ground_truth:
            gt_answer = ground_truth[query_id].get("Answer") or ground_truth[query_id].get("answer")

        parts = _extract_compare_parts(test_data)
        if not parts:
            print(f"  Skipping: no compare_sections/retrieved_docs (need policy/provider definitions)")
            continue

        policy_def = parts["policy_definition"]
        provider_def = parts["provider_manual_definition"]
        policy_chunks = parts["policy_chunks"]
        provider_chunks = parts["provider_chunks"]

        similarities = parts.get("similarities", [])
        differences = parts.get("differences", [])

        result: Dict[str, Any] = {
            "query": query_full,
            "policy": {},
            "provider": {},
            "query_alignment": {},
            "compare_quality": {},
            "overall_score": 0.0,
        }

        # --- 1. Policy side ---
        if policy_def:
            print("  Evaluating policy side...")
            result["policy"]["faithfulness"] = evaluator.evaluate_faithfulness(
                query_full, policy_def, policy_chunks
            )
            result["policy"]["answer_relevance"] = evaluator.evaluate_answer_relevance(
                query_full, policy_def
            )
            if gt_answer:
                result["policy"]["correctness"] = evaluator.evaluate_correctness(
                    query_full, policy_def, gt_answer
                )
            for k, v in result["policy"].items():
                if isinstance(v, dict) and "score" in v:
                    scores_by_metric[f"policy_{k}"].append(v["score"])

        # --- 2. Provider side ---
        if provider_def:
            print("  Evaluating provider side...")
            result["provider"]["faithfulness"] = evaluator.evaluate_faithfulness(
                query_full, provider_def, provider_chunks
            )
            result["provider"]["answer_relevance"] = evaluator.evaluate_answer_relevance(
                query_full, provider_def
            )
            if gt_answer:
                result["provider"]["correctness"] = evaluator.evaluate_correctness(
                    query_full, provider_def, gt_answer
                )
            for k, v in result["provider"].items():
                if isinstance(v, dict) and "score" in v:
                    scores_by_metric[f"provider_{k}"].append(v["score"])

        # --- 3. Query alignment ---
        if similarities or differences:
            print("  Evaluating query alignment...")
            qa = _evaluate_query_alignment(
                llm_client,
                query_full,
                concept,
                similarities,
                differences,
            )
            qa_metric_score = _derive_score_from_fields(qa, ["comparison_query_relevance"])
            if qa_metric_score is not None:
                qa["metric_score"] = qa_metric_score
            result["query_alignment"] = qa
            if qa_metric_score is not None:
                scores_by_metric["query_alignment"].append(qa_metric_score)

        # --- 4. Compare quality (similarities / differences) ---
        if similarities or differences:
            print("  Evaluating compare quality (similarities/differences)...")
            cq = _evaluate_compare_quality(
                llm_client,
                query_full,
                concept,
                similarities,
                differences,
                policy_chunks,
                provider_chunks,
                include_correctness=bool(gt_answer),
            )
            cq_fields = ["accuracy_sim", "accuracy_diff", "completeness", "balance"]
            if gt_answer:
                cq_fields.append("correctness")
            cq_metric_score = _derive_score_from_fields(
                cq,
                cq_fields,
            )
            if cq_metric_score is not None:
                cq["metric_score"] = cq_metric_score
            result["compare_quality"] = cq
            if cq_metric_score is not None:
                scores_by_metric["compare_quality"].append(cq_metric_score)

        # --- Overall (average of policy, provider, query_alignment, compare_quality) ---
        all_scores = []
        for section in (result.get("policy", {}), result.get("provider", {})):
            for v in section.values():
                if isinstance(v, dict) and "score" in v:
                    all_scores.append(v["score"])
        if result.get("query_alignment", {}).get("metric_score") is not None:
            all_scores.append(result["query_alignment"]["metric_score"])
        if result.get("compare_quality", {}).get("metric_score") is not None:
            all_scores.append(result["compare_quality"]["metric_score"])
        result["overall_score"] = round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0
        scores_by_metric["overall"].append(result["overall_score"])

        evaluation_results["results"][test_id] = result
        print(f"  Overall score: {result['overall_score']:.3f}")

    evaluation_results["summary"] = {
        metric: {
            "mean": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "min": round(min(scores), 3) if scores else 0.0,
            "max": round(max(scores), 3) if scores else 0.0,
            "count": len(scores),
        }
        for metric, scores in scores_by_metric.items()
        if scores
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving evaluation results to: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("COMPARE DEFINITIONS EVALUATION SUMMARY")
    print("=" * 60)
    for metric, stats in evaluation_results["summary"].items():
        print(f"{metric:25s}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
    print("=" * 60)

    return evaluation_results
