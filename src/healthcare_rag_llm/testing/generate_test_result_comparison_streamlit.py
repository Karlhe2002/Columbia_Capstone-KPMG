"""
Batch runner that mirrors the Streamlit Compare page (frontend/pages/2_Compare.py)
exactly, so the JSON output matches what a user would see on the Streamlit UI.

Key differences from generate_test_result_comparison.py (intentional, to match
what the frontend actually does):

  1. LLM client is built from APIConfigManager().get_default_config() — i.e.
     whatever is set as `default_provider` / `default_model` in
     `configs/api_config.yaml` (currently openai_official + gpt-5.4-mini).
  2. ResponseGenerator is constructed with only (llm_client, filter_extractor),
     so it keeps the library-default `use_reranker=True`.
  3. answer_compare_definitions is called with `concept=question` (i.e. the
     question itself is passed as the concept, exactly like the frontend's
     `rag_pipeline.answer_compare_definitions(pending_query, concept=pending_query)`).
     The `concept` field from the input JSON is NOT used for retrieval — it
     is only preserved in the output as metadata.
  4. top_k_per_source / rerank_top_k are NOT passed — the library defaults
     (4 and 30) are used, matching the frontend.
  5. chat_history is cleared before each new query, mirroring how the
     frontend starts a fresh compare session for every new (non-follow-up)
     submission.

Output JSON schema per row (matching the schema agreed upon):
    {
        "query_id": <input test key>,
        "query_content": <question>,
        "concept": <concept from input JSON, or null>,
        "task_type": "compare_definition",
        "long_version_id": "<tester-name>-<LLMClient>-<provider>-<model>-<repeats>",
        "short_version_id": <provided version id>,
        "retrieved_docs": {
            "policy": [...],
            "provider_manual": [...]
        },
        "answers": <parsed structured comparison sections>
    }

The `answers` field holds the parsed compare_sections dict (headline_summary,
policy_definition, provider_manual_definition, similarities, differences,
caveats) — i.e. the structured content the Streamlit Compare page actually
renders on screen. The raw LLM string, follow-up questions and query
understanding are intentionally dropped because the raw string is just the
JSON-encoded form of compare_sections, and the other fields are UI-only.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from tqdm import tqdm

from healthcare_rag_llm.filters.load_metadata import build_filter_extractor
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.response_generator import ResponseGenerator
from healthcare_rag_llm.utils.api_config import APIConfigManager


class RAGComparisonStreamlitBatchTester:
    """Run the compare-definitions pipeline for every test query, producing the
    same outputs the Streamlit Compare page would produce, and save as JSON."""

    def __init__(
        self,
        testing_queries_path: str = "data/testing_queries/test_query_compare.json",
        output_dir: str = "data/test_results",
        version_id: str = "comparison_streamlit_openai",
        repeats: int = 1,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self.testing_queries_path = testing_queries_path
        self.output_dir = output_dir
        self.version_id = version_id
        self.repeats = int(repeats)

        self.llm_client = llm_client if llm_client is not None else self._build_frontend_llm_client()

        self.filter_extractor = build_filter_extractor(llm_client=self.llm_client)
        self.response_generator = ResponseGenerator(
            self.llm_client,
            filter_extractor=self.filter_extractor,
        )

        llm_name = self.llm_client.__class__.__name__
        provider = getattr(self.llm_client, "provider", "unknown")
        model = getattr(self.llm_client, "model", "unknown")
        self.long_version_id = (
            f"RAGComparisonStreamlitBatchTester-{llm_name}-{provider}-{model}-r{self.repeats}"
        )
        self.short_version_id = self.version_id

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"{self.version_id}.json")

    @staticmethod
    def _build_frontend_llm_client() -> LLMClient:
        """Build the LLMClient the same way frontend/pages/2_Compare.py does,
        but prefer the explicit `default_model` declared in
        `configs/api_config.yaml` so the batch run is locked to whatever model
        the project is currently standardised on."""
        api_config_manager = APIConfigManager()
        api_config_default = api_config_manager.get_default_config()
        provider = (api_config_default.provider or "").lower()

        model = api_config_manager._config.get("default_model")
        if not model:
            provider_to_model = {
                "gemini": "gemini-2.5-flash",
                "openai": "gpt-5.4-mini-2026-03-17",
            }
            model = provider_to_model.get(provider, "gpt-5.4-mini-2026-03-17")

        return LLMClient(
            api_key=api_config_default.api_key,
            model=model,
            provider=api_config_default.provider,
            base_url=api_config_default.base_url,
        )

    def run(self) -> Dict[str, Any]:
        tests = self._read_json(self.testing_queries_path)

        results: Dict[str, Any] = {}
        row_counter = 0
        total_iterations = len(tests) * self.repeats

        print(f"\n{'=' * 60}")
        print("Starting RAG Comparison (Streamlit-parity) Batch Testing")
        print(f"Provider: {getattr(self.llm_client, 'provider', 'unknown')}")
        print(f"Model:    {getattr(self.llm_client, 'model', 'unknown')}")
        print(f"Total queries: {len(tests)} | Repeats per query: {self.repeats}")
        print(f"Total iterations: {total_iterations}")
        print(f"Output: {self.output_path}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        with tqdm(total=total_iterations, desc="Processing comparisons", unit="query") as pbar:
            for query_id, payload in tests.items():
                question = self._extract_question(payload, query_id)
                # concept is preserved in output for reference, but NOT used in
                # the compare call (the frontend passes the question itself as
                # the concept).
                original_concept = self._extract_optional_str(payload, "concept")
                task_type = self._extract_optional_str(payload, "task_type") or "compare_definition"

                for _ in range(self.repeats):
                    row_counter += 1
                    row_name = f"test_id_{row_counter}"
                    pbar.set_description(f"Processing {query_id}")

                    # Mirror the frontend: a brand-new compare query starts a
                    # fresh conversation.
                    try:
                        self.response_generator.chat_history.clear()
                    except Exception:
                        pass

                    llm_start = time.time()
                    try:
                        compare_result = self.response_generator.answer_compare_definitions(
                            question,
                            concept=question,
                        )
                    except Exception as e:
                        compare_result = {
                            "compare_sections": {"_parse_failed": True, "error": f"{type(e).__name__}: {e}"},
                            "retrieved_docs": {"policy": [], "provider_manual": []},
                        }
                    llm_elapsed = time.time() - llm_start
                    pbar.set_postfix({"LLM_time": f"{llm_elapsed:.1f}s"})

                    retrieved_docs = compare_result.get("retrieved_docs", {}) or {}
                    if not isinstance(retrieved_docs, dict):
                        retrieved_docs = {"policy": [], "provider_manual": []}

                    results[row_name] = {
                        "query_id": query_id,
                        "query_content": question,
                        "concept": original_concept,
                        "task_type": task_type,
                        "long_version_id": self.long_version_id,
                        "short_version_id": self.short_version_id,
                        "retrieved_docs": {
                            "policy": retrieved_docs.get("policy", []) or [],
                            "provider_manual": retrieved_docs.get("provider_manual", []) or [],
                        },
                        "answers": compare_result.get("compare_sections", {}),
                    }

                    # Write incrementally so a crash mid-run does not lose
                    # everything that has finished so far.
                    self._write_json(self.output_path, results)

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / total_iterations if total_iterations > 0 else 0

        print(f"\n{'=' * 60}")
        print("Comparison testing completed!")
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time / 60:.1f} min)")
        print(f"Average time per query: {avg_time:.2f}s")
        print(f"Output saved to: {self.output_path}")
        print(f"{'=' * 60}\n")

        return results

    @staticmethod
    def _read_json(path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)

    @staticmethod
    def _write_json(path: str, data: Dict[str, Any]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def _extract_question(payload: Any, query_id: str) -> str:
        if isinstance(payload, dict) and "question" in payload and isinstance(payload["question"], str):
            return payload["question"].strip()
        raise ValueError(f"Input JSON missing 'question' for test key: {query_id}")

    @staticmethod
    def _extract_optional_str(payload: Any, key: str) -> Optional[str]:
        if isinstance(payload, dict):
            value = payload.get(key)
            if isinstance(value, str):
                cleaned = value.strip()
                return cleaned or None
        return None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch-run the Compare Definitions pipeline using the same "
                    "configuration as the Streamlit frontend, and save results as JSON.",
    )
    parser.add_argument(
        "--input",
        default="data/testing_queries/test_query_compare.json",
        help="Path to the input test queries JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/test_results",
        help="Directory to write the result JSON into.",
    )
    parser.add_argument(
        "--version-id",
        default="comparison_streamlit_openai",
        help="Short version id (also used as output filename stem).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to run each query (default 1, matching UI behavior).",
    )
    args = parser.parse_args()

    tester = RAGComparisonStreamlitBatchTester(
        testing_queries_path=args.input,
        output_dir=args.output_dir,
        version_id=args.version_id,
        repeats=args.repeats,
    )
    tester.run()


if __name__ == "__main__":
    main()
