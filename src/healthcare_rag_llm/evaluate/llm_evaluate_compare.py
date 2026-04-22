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
            value = data[key]
            # Support both flat numeric values and nested metric objects: {"score": ...}.
            if isinstance(value, dict) and "score" in value:
                vals.append(_clamp_score(value["score"]))
            else:
                vals.append(_clamp_score(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)


def _derive_section_metric_score(section: Dict[str, Any], fields: List[str]) -> Optional[float]:
    """Compute mean score from section metrics like {metric: {score: ...}}."""
    vals: List[float] = []
    for key in fields:
        metric_obj = section.get(key)
        if not isinstance(metric_obj, dict) or "score" not in metric_obj:
            continue
        try:
            vals.append(_clamp_score(metric_obj["score"]))
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


def _remove_schema_fields(value: Any) -> Any:
    """Recursively remove all `schema` keys from nested dict/list structures."""
    if isinstance(value, dict):
        return {
            k: _remove_schema_fields(v)
            for k, v in value.items()
            if k != "schema"
        }
    if isinstance(value, list):
        return [_remove_schema_fields(item) for item in value]
    return value


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


def _non_empty_text(value: Any) -> Optional[str]:
    """Return stripped text when non-empty, otherwise None."""
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


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
    comparison_items: List[str],
    policy_chunks: List[Dict[str, Any]],
    provider_chunks: List[Dict[str, Any]],
    compare_ground_truth: Optional[str],
    include_correctness: bool,
) -> Dict[str, Any]:
    """
    Multi-pass evaluation for comparison quality against chunks,
    where each metric is scored in a separate LLM call.
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

    combined_accuracy_mode = len(comparison_items) == 1 and bool(comparison_items[0].strip())
    comparison_text = comparison_items[0].strip() if combined_accuracy_mode else ""

    combined_accuracy_result: Optional[Dict[str, Any]] = None
    if combined_accuracy_mode:
        prompt = f"""You are an expert evaluator assessing TWO related metrics for comparison quality.

QUERY:
{query}

TARGET CONCEPT:
{target}

COMPARISON PARAGRAPH:
{comparison_text}

POLICY SOURCE CHUNKS (excerpts):
{policy_ctx}

PROVIDER MANUAL SOURCE CHUNKS (excerpts):
{provider_ctx}

TASK:
Evaluate the paragraph-style comparison and score BOTH metrics separately:
- accuracy_sim: whether similarity claims are directly supported by evidence.
- accuracy_diff: whether difference claims are directly supported by evidence.

Score band guidance:
- 0.9-1.0: Fully supported and well grounded.
- 0.7-0.8: Mostly supported with minor issues.
- 0.4-0.6: Partially supported with notable gaps.
- 0.1-0.3: Largely unsupported.
- 0.0: No valid support.

Respond in JSON format:
{{
    "accuracy_sim": {{
        "score": <float between 0 and 1>,
        "schema": "similarity_grounding",
        "reason": "<brief reason for similarity grounding>"
    }},
    "accuracy_diff": {{
        "score": <float between 0 and 1>,
        "schema": "difference_grounding",
        "reason": "<brief reason for difference grounding>"
    }},
    "reason": "<brief combined reason>"
}}"""
        combined_resp = llm_client.chat(prompt)
        combined_accuracy_result = _parse_json_response(
            combined_resp,
            "compare_quality_accuracy",
            require_score=False,
            fallback={
                "accuracy_sim": {"score": 0.0, "schema": "similarity_grounding", "reason": "Failed to evaluate accuracy_sim"},
                "accuracy_diff": {"score": 0.0, "schema": "difference_grounding", "reason": "Failed to evaluate accuracy_diff"},
                "reason": "Failed to evaluate combined accuracy metrics",
            },
        )

    metric_specs: List[Dict[str, str]] = [
        {
            "key": "accuracy_sim",
            "schema": "similarity_grounding",
            "task": "Score only whether similarity claims are directly supported by evidence.",
        },
        {
            "key": "accuracy_diff",
            "schema": "difference_grounding",
            "task": "Score only whether difference claims are directly supported by evidence.",
        },
        {
            "key": "completeness",
            "schema": "coverage_completeness",
            "task": "Score whether major expected similarities/differences are missing.",
        },
        {
            "key": "balance",
            "schema": "cross_source_balance",
            "task": "Score whether BOTH sources are represented and grounded without one-sided emphasis or one-sided hallucination.",
        },
    ]
    if include_correctness:
        metric_specs.append(
            {
                "key": "correctness",
                "schema": "factual_correctness",
                "task": "Score factual correctness against query and source evidence; ignore writing style.",
            }
        )

    result: Dict[str, Any] = {
        "reasoning": "",
    }
    metric_reasons: List[str] = []

    if combined_accuracy_result is not None:
        for metric_key, metric_schema in [("accuracy_sim", "similarity_grounding"), ("accuracy_diff", "difference_grounding")]:
            metric_obj = combined_accuracy_result.get(metric_key, {})
            if not isinstance(metric_obj, dict):
                metric_obj = {}
            score_val = metric_obj.get("score", 0.0)
            try:
                score_val = _clamp_score(score_val)
            except (TypeError, ValueError):
                score_val = 0.0
            normalized_metric = {
                "score": score_val,
                "schema": str(metric_obj.get("schema", metric_schema)),
                "reason": str(metric_obj.get("reason", "")),
            }
            result[metric_key] = normalized_metric
            if normalized_metric["reason"]:
                metric_reasons.append(f"{metric_key}: {normalized_metric['reason']}")

    for spec in metric_specs:
        metric_key = spec["key"]
        metric_schema = spec["schema"]
        metric_task = spec["task"]

        # Single-paragraph comparisons already produced both accuracy scores above.
        # Skip the per-metric calls in that case so we do not duplicate work.
        if combined_accuracy_mode and metric_key in {"accuracy_sim", "accuracy_diff"}:
            continue

        correctness_reference = ""
        if metric_key == "correctness" and compare_ground_truth:
            correctness_reference = f"""
    GROUND TRUTH ANSWER FOR COMPARISON CORRECTNESS:
    {compare_ground_truth}
    """

        prompt = f"""You are an expert evaluator assessing ONE metric for comparison quality.

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

{correctness_reference}

METRIC TO EVALUATE:
- metric_name: {metric_key}
- schema: {metric_schema}
- instruction: {metric_task}

Score band guidance:
- 0.9-1.0: Fully satisfies this metric schema with strong grounding.
- 0.7-0.8: Mostly satisfies schema with minor issues.
- 0.4-0.6: Partially satisfies schema with notable gaps.
- 0.1-0.3: Largely fails schema.
- 0.0: No valid support for this metric.

Respond in JSON format:
{{
    "score": <float between 0 and 1>,
    "schema": "{metric_schema}",
    "reason": "<brief reason for this metric>"
}}
"""
        metric_resp = llm_client.chat(prompt)
        metric_result = _parse_json_response(
            metric_resp,
            f"compare_quality_{metric_key}",
            require_score=False,
            fallback={
                "score": 0.0,
                "schema": metric_schema,
                "reason": f"Failed to evaluate {metric_key}",
            },
        )

        score_val = metric_result.get("score", 0.0)
        try:
            score_val = _clamp_score(score_val)
        except (TypeError, ValueError):
            score_val = 0.0

        normalized_metric = {
            "score": score_val,
            "schema": str(metric_result.get("schema", metric_schema)),
            "reason": str(metric_result.get("reason", "")),
        }
        result[metric_key] = normalized_metric

        if normalized_metric["reason"]:
            metric_reasons.append(f"{metric_key}: {normalized_metric['reason']}")

    if combined_accuracy_result is not None and str(combined_accuracy_result.get("reason", "")):
        metric_reasons.insert(0, f"accuracy_pair: {combined_accuracy_result.get('reason', '')}")

    result["reasoning"] = " | ".join(metric_reasons)
    return result

def _extract_compare_parts(test_entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extract policy_definition, provider_manual_definition, similarities, differences,
    policy_chunks, and provider_chunks from the current compare output shape.
    """
    answers = test_entry.get("answers")
    if not isinstance(answers, dict):
        return None

    policy_def = answers.get("policy_definition", "")
    provider_def = answers.get("provider_manual_definition", "")
    sim = _to_string_list(answers.get("similarities", []))
    diff = _to_string_list(answers.get("differences", []))
    retrieved = test_entry.get("retrieved_docs", {})
    policy_chunks = _normalize_retrieved_chunks(retrieved.get("policy", []))
    provider_chunks = _normalize_retrieved_chunks(retrieved.get("provider_manual", []))
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
    3. Query alignment: policy def, provider def, and comparison vs query
    4. Compare quality: similarities/differences accuracy_sim, accuracy_diff, correctness, completeness, balance.
    """
    print(f"Loading compare results from: {compare_results_path}")
    with open(compare_results_path, "r", encoding="utf-8") as f:
        compare_results = json.load(f)

    ground_truth: Dict[str, Any] = {}
    if ground_truth_path and Path(ground_truth_path).exists():
        print(f"Loading ground truth from: {ground_truth_path}")
        try:
            # utf-8-sig handles files that start with BOM.
            with open(ground_truth_path, "r", encoding="utf-8-sig") as f:
                gt_data = json.load(f)
                ground_truth = gt_data if isinstance(gt_data, dict) else {}
            print(f"Loaded {len(ground_truth)} ground truth entries")
        except json.JSONDecodeError:
            print(
                "Warning: ground truth is not a JSON object file. "
                "Skipping external ground truth for eval step."
            )
            ground_truth = {}

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
        "comparison_overall": [],
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

        policy_gt_answer = _non_empty_text(test_data.get("policy_ground_truth"))
        provider_gt_answer = _non_empty_text(test_data.get("pm_ground_truth"))
        compare_gt_answer = _non_empty_text(test_data.get("ground_truth"))

        # Backward compatibility: if per-scope ground truth is absent, fall back to shared value.
        shared_gt_answer = _non_empty_text(gt_answer)
        if not policy_gt_answer:
            policy_gt_answer = shared_gt_answer
        if not provider_gt_answer:
            provider_gt_answer = shared_gt_answer
        if not compare_gt_answer:
            compare_gt_answer = shared_gt_answer

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
        comparison_items = parts.get("comparison", [])

        result: Dict[str, Any] = {
            "query": query,
            "concept": concept,
            "answer": test_data.get("answer", ""),
            "policy": {},
            "provider": {},
            "query_alignment": {},
            "compare_quality": {},
            "comparison_overall_score": 0.0,
            "overall_score": 0.0,
        }

        # --- 1. Policy side ---
        if policy_def:
            print("  Evaluating policy side...")
            result["policy"]["faithfulness"] = evaluator.evaluate_faithfulness(
                query_full, policy_def, policy_chunks
            )
            # Use policy-specific query for relevance evaluation (no comparison intent)
            # Format: "Medicaid for {concept}"
            query_policy = f"Medicaid for {concept}" if concept else query
            result["policy"]["answer_relevance"] = evaluator.evaluate_answer_relevance(
                query_policy, policy_def
            )
            if policy_gt_answer:
                result["policy"]["correctness"] = evaluator.evaluate_correctness(
                    query_full, policy_def, policy_gt_answer
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
            # Use provider-specific query for relevance evaluation (no comparison intent)
            # Format: "provider manual for {concept}"
            query_provider = f"provider manual for {concept}" if concept else query
            result["provider"]["answer_relevance"] = evaluator.evaluate_answer_relevance(
                query_provider, provider_def
            )
            if provider_gt_answer:
                result["provider"]["correctness"] = evaluator.evaluate_correctness(
                    query_full, provider_def, provider_gt_answer
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
                comparison_items,
                policy_chunks,
                provider_chunks,
                compare_ground_truth=compare_gt_answer,
                include_correctness=bool(compare_gt_answer),
            )
            cq_fields = ["accuracy_sim", "accuracy_diff", "completeness", "balance"]
            if compare_gt_answer:
                cq_fields.append("correctness")
            cq_metric_score = _derive_score_from_fields(
                cq,
                cq_fields,
            )
            result["compare_quality"] = cq
            if cq_metric_score is not None:
                scores_by_metric["compare_quality"].append(cq_metric_score)

        # Weighted comparison overall score combines query alignment and compare-quality dimensions.
        # With correctness: query_alignment 30%, correctness 30%, completeness 20%, accuracy 15%, balance 5%.
        # Without correctness: query_alignment 50%, completeness 30%, accuracy 15%, balance 5%.
        qa_score = _derive_score_from_fields(result.get("query_alignment", {}), ["comparison_query_relevance"])
        cq_result = result.get("compare_quality", {})
        accuracy_score = _derive_score_from_fields(cq_result, ["accuracy_sim", "accuracy_diff"])
        completeness_score = _derive_score_from_fields(cq_result, ["completeness"])
        correctness_score = _derive_score_from_fields(cq_result, ["correctness"])
        balance_score = _derive_score_from_fields(cq_result, ["balance"])

        weighted_components: List[tuple[float, Optional[float]]] = []
        if correctness_score is not None:
            weighted_components = [
                (0.30, qa_score),
                (0.30, correctness_score),
                (0.20, completeness_score),
                (0.15, accuracy_score),
                (0.05, balance_score),
            ]
        else:
            weighted_components = [
                (0.50, qa_score),
                (0.30, completeness_score),
                (0.15, accuracy_score),
                (0.05, balance_score),
            ]

        weighted_sum = sum(weight * score for weight, score in weighted_components if score is not None)
        available_weight = sum(weight for weight, score in weighted_components if score is not None)
        if available_weight > 0:
            result["comparison_overall_score"] = round(weighted_sum / available_weight, 3)
            scores_by_metric["comparison_overall"].append(result["comparison_overall_score"])

        # Weighted overall score:
        # - comparison_overall_score: 70%
        # - policy side average (faithfulness/correctness/answer_relevance): 15%
        # - provider side average (faithfulness/correctness/answer_relevance): 15%
        # For policy/provider side, if correctness is missing, average the remaining two metrics.
        policy_section_score = _derive_section_metric_score(
            result.get("policy", {}),
            ["faithfulness", "correctness", "answer_relevance"],
        )
        provider_section_score = _derive_section_metric_score(
            result.get("provider", {}),
            ["faithfulness", "correctness", "answer_relevance"],
        )

        comparison_overall_score = None
        if (result.get("query_alignment") or result.get("compare_quality")) and (
            "comparison_overall_score" in result
        ):
            comparison_overall_score = _clamp_score(result["comparison_overall_score"])

        weighted_components: List[tuple[float, Optional[float]]] = [
            (0.70, comparison_overall_score),
            (0.15, policy_section_score),
            (0.15, provider_section_score),
        ]
        weighted_sum = sum(weight * score for weight, score in weighted_components if score is not None)
        available_weight = sum(weight for weight, score in weighted_components if score is not None)
        result["overall_score"] = round(weighted_sum / available_weight, 3) if available_weight > 0 else 0.0
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
    output_payload = _remove_schema_fields(evaluation_results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("COMPARE DEFINITIONS EVALUATION SUMMARY")
    print("=" * 60)
    for metric, stats in evaluation_results["summary"].items():
        print(f"{metric:25s}: mean={stats['mean']:.3f}, min={stats['min']:.3f}, max={stats['max']:.3f}")
    print("=" * 60)

    return evaluation_results
