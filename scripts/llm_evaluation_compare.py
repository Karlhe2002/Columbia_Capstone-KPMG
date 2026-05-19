"""
Script to run LLM-based evaluation on answer_compare_definitions test results.

Accepts compare_definitions output format. Metrics: policy/provider faithfulness,
answer_relevance, correctness; cross_source_correlation; compare_quality; format_compliance.

Input fields: question, concept, compare_sections, retrieved_docs. For format_compliance
(LLM raw output schema check), include "answer" with the raw LLM generation before parse.

Usage:
.venv/bin/python scripts/llm_evaluation_compare.py \
-t data/llm_eval_results_compare/comparison_processed/input.json \
-o data/llm_eval_results_compare/comparison_llm_eval_results/output.json \
--provider openai_official --model gpt-4o-mini \
--limit 1
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from healthcare_rag_llm.evaluate.llm_evaluate_compare import evaluate_compare_test_results
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.utils.api_config import load_api_config


def main():
    parser = argparse.ArgumentParser(
        description="Run LLM-based evaluation on answer_compare_definitions test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        """,
    )

    parser.add_argument("-t", "--test_results", required=True, help="Path to compare_definitions test results JSON")
    parser.add_argument("-o", "--output", required=True, help="Path to save LLM evaluation results")
    parser.add_argument("-g", "--ground_truth", default=None, help="Optional path to ground truth for correctness evaluation")
    parser.add_argument("--model", default="gpt-5", help="Model for evaluation (default: gpt-5)")
    parser.add_argument("--provider", default=None, help="API provider (default: from config)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tests")
    parser.add_argument("--config", default="configs/api_config.yaml", help="Path to API config")

    args = parser.parse_args()

    if not Path(args.test_results).exists():
        print(f"Error: Test results file not found: {args.test_results}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading API configuration from: {args.config}")
    try:
        config = load_api_config(args.config)
    except Exception as e:
        print(f"Error loading API config: {e}")
        sys.exit(1)

    provider_name = args.provider or config.get("default_provider", "bltcy")
    if provider_name not in config["api_providers"]:
        print(f"Error: Provider '{provider_name}' not found in config")
        print(f"Available: {list(config['api_providers'].keys())}")
        sys.exit(1)

    provider_config = config["api_providers"][provider_name]

    print(f"Initializing LLM client: model={args.model}, provider={provider_name}")
    try:
        llm_client = LLMClient(
            api_key=provider_config["api_key"],
            base_url=provider_config.get("base_url"),
            model=args.model,
            provider=provider_config.get("provider", "openai"),
        )
    except Exception as e:
        print(f"Error initializing LLM client: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STARTING LLM EVALUATION (compare_definitions)")
    print("=" * 60)
    print(f"Compare results: {args.test_results}")
    print(f"Output: {args.output}")
    print(f"Ground truth: {args.ground_truth or 'Not provided'}")
    print(f"Evaluator model: {args.model}")
    print(f"Limit: {args.limit or 'All tests'}")
    print("=" * 60 + "\n")

    try:
        evaluate_compare_test_results(
            compare_results_path=args.test_results,
            output_path=args.output,
            llm_client=llm_client,
            ground_truth_path=args.ground_truth,
            limit=args.limit,
        )
        print(f"\nEvaluation complete! Results saved to: {args.output}")
        return 0
    except Exception as e:
        print(f"\nError during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
