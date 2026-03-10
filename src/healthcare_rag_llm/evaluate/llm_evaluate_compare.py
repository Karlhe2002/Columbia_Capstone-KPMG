"""
LLM-based evaluation for answer_compare_definitions outputs.

1. Policy side: evaluate policy_definition against policy chunks
2. Provider side: evaluate provider_manual_definition against provider chunks
3. Cross-source correlation: policy vs provider definitions
4. Compare quality: similarities/differences accuracy
5. Format compliance: raw LLM output (before parse) matches expected JSON schema
   (headline_summary, policy_definition, provider_manual_definition, similarities, differences, caveats)
"""

import inspect
import json
import re
from typing import Dict, List, Any, Optional
from pathlib import Path

from healthcare_rag_llm.evaluate.llm_evaluate import LLMEvaluator
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.response_generator import ResponseGenerator

_compare_keys_cache: Optional[List[str]] = None


def _get_compare_output_required_keys() -> List[str]:
    """Extract required JSON keys from response_generator prompt (no modification of source)."""
    global _compare_keys_cache
    if _compare_keys_cache is not None:
        return _compare_keys_cache
    src = inspect.getsource(ResponseGenerator.answer_compare_definitions)
    schema_start = src.find("exactly these keys")
    schema_end = src.find("}}", schema_start) + 2
    block = src[schema_start:schema_end]
    keys = re.findall(r'"([a-z_]+)"\s*:', block)
    seen = []
    for k in keys:
        if k not in seen:
            seen.append(k)
    _compare_keys_cache = seen
    return _compare_keys_cache


def _parse_json_response(response: str, metric_name: str) -> Dict[str, Any]:
    """Parse LLM JSON response, same logic as LLMEvaluator._parse_json_response."""
    try:
        cleaned = response.replace("\t", " ").replace("\r", "")
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}") + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx:end_idx]
            try:
                result = json.loads(json_str)
            except json.JSONDecodeError:
                result = json.loads(json_str, strict=False)
            
            # Validate required fields
            required_keys = ["headline_summary", "policy_definition", "provider_manual_definition", "similarities", "differences", "caveats"]
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing required field: {key}")
            
            # Validate field types
            if not isinstance(result.get("headline_summary"), str):
                raise ValueError("'headline_summary' must be a string")
            if not isinstance(result.get("policy_definition"), str):
                raise ValueError("'policy_definition' must be a string")
            if not isinstance(result.get("provider_manual_definition"), str):
                raise ValueError("'provider_manual_definition' must be a string")
            if not isinstance(result.get("similarities"), list):
                raise ValueError("'similarities' must be a list")
            if not isinstance(result.get("differences"), list):
                raise ValueError("'differences' must be a list")
            if result.get("caveats") is not None and not isinstance(result.get("caveats"), str):
                raise ValueError("'caveats' must be a string or null")
            
            # Validate score field
            if "score" not in result:
                raise ValueError("Missing 'score' field")
            result["score"] = max(0.0, min(1.0, float(result["score"])))
            return result
        raise ValueError("No JSON found in response")
    except Exception as e:
        print(f"  Warning: Failed to parse {metric_name} response: {e}")
        return {
            "score": 0.0,
            "reasoning": f"Failed to parse LLM response: {str(e)}",
            "error": str(e),
        }


def _evaluate_format_compliance(raw_answer: str) -> Dict[str, Any]:
    """
    Check if raw LLM output (before parse) conforms to the expected JSON schema.
    No LLM call - programmatic validation only.
    """
    issues: List[str] = []
    score = 1.0

    if not raw_answer or not isinstance(raw_answer, str):
        return {
            "score": 0.0,
            "parse_ok": False,
            "issues": ["No raw answer provided"],
            "reasoning": "Raw LLM output (answer field) is required for format compliance check.",
        }

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_answer.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return {
            "score": 0.0,
            "parse_ok": False,
            "issues": [f"JSON parse error: {e}"],
            "reasoning": "Raw output is not valid JSON.",
        }

    if not isinstance(parsed, dict):
        return {
            "score": 0.0,
            "parse_ok": False,
            "issues": ["Root must be a JSON object"],
            "reasoning": "Expected a JSON object, not array or primitive.",
        }

    for key in _get_compare_output_required_keys():
        if key not in parsed:
            issues.append(f"Missing required key: {key}")
            score -= 0.2
        else:
            val = parsed[key]
            if key in ("headline_summary", "policy_definition", "provider_manual_definition"):
                if not isinstance(val, str):
                    issues.append(f"'{key}' must be string, got {type(val).__name__}")
                    score -= 0.1
            elif key in ("similarities", "differences"):
                if not isinstance(val, list):
                    issues.append(f"'{key}' must be list, got {type(val).__name__}")
                    score -= 0.15
                else:
                    for i, item in enumerate(val):
                        if not isinstance(item, str):
                            issues.append(f"'{key}[{i}]' must be string")
                            score -= 0.05
            elif key == "caveats":
                if val is not None and not isinstance(val, str):
                    issues.append(f"'{key}' must be string or null, got {type(val).__name__}")
                    score -= 0.1

    extra = set(parsed.keys()) - set(_get_compare_output_required_keys())
    if extra:
        issues.append(f"Extra keys (not in schema): {sorted(extra)}")
        score -= 0.05 * len(extra)

    score = max(0.0, min(1.0, score))
    return {
        "score": round(score, 3),
        "parse_ok": True,
        "issues": issues if issues else [],
        "reasoning": "Schema valid" if not issues else "; ".join(issues),
    }


def _evaluate_cross_source_correlation(
    llm_client: LLMClient,
    query: str,
    concept: str,
    policy_definition: str,
    provider_manual_definition: str,
) -> Dict[str, Any]:
    """
    LLM-as-judge: Evaluate whether policy and provider definitions
    correlate with each other (consistent, complementary, same concept).
    """
    target = concept if concept else "the requested concept"
    prompt = f"""You are an expert evaluator assessing whether two definitions of the same concept (from different sources) correlate with each other.

QUERY:
{query}

TARGET CONCEPT:
{target}

POLICY DEFINITION:
{policy_definition}

PROVIDER MANUAL DEFINITION:
{provider_manual_definition}

TASK:
Evaluate whether these two definitions CORRELATE with each other. Consider:
1. Do they address the same concept?
2. Are they consistent (no contradictions)?
3. Do they complement each other or reinforce key points?
4. Are differences explained and reasonable (e.g., different perspectives)?
5. Overall coherence between the two sources

Rate the correlation on a scale of 0.0 to 1.0:
- 1.0: Strongly correlated, consistent, complementary, clearly same concept
- 0.8-0.9: Well correlated, minor inconsistencies
- 0.5-0.7: Partially correlated, some contradictions or divergent focus
- 0.3-0.4: Weakly correlated, significant inconsistencies
- 0.0-0.2: Not correlated, contradictory, or different concepts

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation of correlation assessment>",
    "contradictions": ["<list any contradictions if present>"],
    "consistencies": ["<list key consistencies>"]
}}"""

    response = llm_client.chat(prompt)
    return _parse_json_response(response, "cross_source_correlation")


def _evaluate_compare_quality(
    llm_client: LLMClient,
    query: str,
    concept: str,
    policy_definition: str,
    provider_manual_definition: str,
    similarities: List[str],
    differences: List[str],
    policy_chunks: List[Dict[str, Any]],
    provider_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    LLM-as-judge: Evaluate whether similarities and differences are accurate,
    grounded in sources, and complete.
    """
    target = concept if concept else "the requested concept"
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

    prompt = f"""You are an expert evaluator assessing the QUALITY of a comparison summary between policy and provider manual definitions.

QUERY:
{query}

TARGET CONCEPT:
{target}

POLICY DEFINITION (LLM output):
{policy_definition}

PROVIDER MANUAL DEFINITION (LLM output):
{provider_manual_definition}

LLM-IDENTIFIED SIMILARITIES:
{sim_text}

LLM-IDENTIFIED DIFFERENCES:
{diff_text}

POLICY SOURCE CHUNKS (excerpts):
{policy_ctx}

PROVIDER MANUAL SOURCE CHUNKS (excerpts):
{provider_ctx}

TASK:
Evaluate the quality of the similarities and differences. Consider:
1. ACCURACY: Are the similarities actually supported by both policy and provider sources?
2. ACCURACY: Are the differences truly distinguishing policy from provider (not mixing them up)?
3. GROUNDING: Can each similarity/difference point be traced to the source chunks?
4. COMPLETENESS: Are major similarities or differences missed?
5. BALANCE: Is the comparison balanced (not overemphasizing one source)?

Rate the compare quality on a scale of 0.0 to 1.0:
- 1.0: Similarities and differences are accurate, well-grounded, and complete
- 0.8-0.9: Minor inaccuracies or omissions
- 0.5-0.7: Some similarities/differences unsupported or incorrect
- 0.3-0.4: Significant errors or omissions
- 0.0-0.2: Largely inaccurate or ungrounded

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "reasoning": "<explanation of compare quality assessment>",
    "inaccurate_points": ["<list any similarity/difference points not supported by sources>"],
    "missing_points": ["<list important similarities or differences that were omitted>"]
}}"""

    response = llm_client.chat(prompt)
    return _parse_json_response(response, "compare_quality")


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
    sim = compare_sections.get("similarities", [])
    diff = compare_sections.get("differences", [])
    if isinstance(sim, str):
        sim = [sim] if sim.strip() else []
    if isinstance(diff, str):
        diff = [diff] if diff.strip() else []
    retrieved = test_entry.get("retrieved_docs", {})
    policy_chunks = retrieved.get("policy", [])
    provider_chunks = retrieved.get("provider_manual", [])
    return {
        "policy_definition": str(policy_def) if policy_def else "",
        "provider_manual_definition": str(provider_def) if provider_def else "",
        "similarities": list(sim),
        "differences": list(diff),
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
    3. Cross-source correlation: policy vs provider definitions
    4. Compare quality: similarities/differences accuracy
    5. Format compliance: raw LLM output (answer field) vs expected JSON schema
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
        "cross_source_correlation": [],
        "compare_quality": [],
        "format_compliance": [],
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

        raw_answer = test_data.get("answer") or test_data.get("answers", "")
        if isinstance(raw_answer, dict):
            raw_answer = json.dumps(raw_answer, ensure_ascii=False)

        result: Dict[str, Any] = {
            "query": query_full,
            "policy": {},
            "provider": {},
            "cross_source_correlation": {},
            "compare_quality": {},
            "format_compliance": {},
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

        # --- 3. Cross-source correlation ---
        if policy_def and provider_def:
            print("  Evaluating cross-source correlation...")
            corr = _evaluate_cross_source_correlation(
                llm_client, query_full, concept, policy_def, provider_def
            )
            result["cross_source_correlation"] = corr
            scores_by_metric["cross_source_correlation"].append(corr["score"])

        # --- 4. Compare quality (similarities / differences) ---
        if policy_def and provider_def:
            print("  Evaluating compare quality (similarities/differences)...")
            cq = _evaluate_compare_quality(
                llm_client,
                query_full,
                concept,
                policy_def,
                provider_def,
                similarities,
                differences,
                policy_chunks,
                provider_chunks,
            )
            result["compare_quality"] = cq
            scores_by_metric["compare_quality"].append(cq["score"])

        # --- 5. Format compliance (raw LLM output schema) ---
        if raw_answer:
            print("  Evaluating format compliance (raw output schema)...")
            fc = _evaluate_format_compliance(raw_answer)
            result["format_compliance"] = fc
            scores_by_metric["format_compliance"].append(fc["score"])
        else:
            result["format_compliance"] = {
                "score": None,
                "parse_ok": None,
                "issues": ["Raw answer not provided - cannot check format"],
                "reasoning": "Need 'answer' field with raw LLM output (before parse).",
            }

        # --- Overall (average of policy, provider, correlation, compare_quality, format_compliance) ---
        all_scores = []
        for section in (result.get("policy", {}), result.get("provider", {})):
            for v in section.values():
                if isinstance(v, dict) and "score" in v:
                    all_scores.append(v["score"])
        if result.get("cross_source_correlation", {}).get("score") is not None:
            all_scores.append(result["cross_source_correlation"]["score"])
        if result.get("compare_quality", {}).get("score") is not None:
            all_scores.append(result["compare_quality"]["score"])
        fc_score = result.get("format_compliance", {}).get("score")
        if fc_score is not None:
            all_scores.append(fc_score)
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
