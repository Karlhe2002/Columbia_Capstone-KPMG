from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

from tqdm import tqdm

from healthcare_rag_llm.filters.load_metadata import build_filter_extractor
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.response_generator import ResponseGenerator


class RAGComparisonBatchTester:
    """
    Batch tester for definition comparison across policy vs provider manual documents.

    Default behavior mirrors generate_test_result.py, but each test case runs the
    comparison pipeline backed by configs/system_prompt_compare.txt.

    Expected input JSON example:
    {
        "test1": {
            "question": "Compare Medicaid versus provider manual guidance for ...",
            "concept": "target concept",
            "task_type": "compare_definition"
        }
    }

    Output JSON schema per row:
      {
        "query_id": <input test key>,
        "query_content": <question>,
        "concept": <concept or null>,
        "task_type": "compare_definition",
        "long_version_id": "<tester-name>-<LLMClient>-k-<top_k_per_source>-<repeats>",
        "short_version_id": <provided version id>,
        "retrieved_docs": {
            "policy": [...],
            "provider_manual": [...]
        },
        "answers": <LLM comparison output>
      }
    """

    def __init__(
        self,
        testing_queries_path: str = "data/testing_queries/testing_query_compare.json",
        output_dir: str = "data/test_results",
        version_id: str = "comparison_version_ollama_theme_aware",
        llm_client: Optional[LLMClient] = None,
        top_k_per_source: int = 3,
        repeats: int = 5,
        rerank_top_k: int = 20,
        use_reranker: bool = True,
    ) -> None:
        self.testing_queries_path = testing_queries_path
        self.output_dir = output_dir
        self.version_id = version_id
        self.top_k_per_source = int(top_k_per_source)
        self.repeats = int(repeats)
        self.rerank_top_k = int(rerank_top_k)
        self.use_reranker = use_reranker

        self.llm_client = (
            llm_client
            if llm_client is not None
            else LLMClient(
                api_key="",
                provider="ollama",
                model="llama3.2:3b",
            )
        )
        
        self.filter_extractor = build_filter_extractor(llm_client=self.llm_client)
        self.response_generator = ResponseGenerator(
            llm_client=self.llm_client,
            use_reranker=self.use_reranker,
            filter_extractor=self.filter_extractor,
        )

        llm_name = self.llm_client.__class__.__name__
        self.long_version_id = (
            f"RAGComparisonBatchTester-{llm_name}-k-{self.top_k_per_source}-{self.repeats}"
        )
        self.short_version_id = self.version_id

        os.makedirs(self.output_dir, exist_ok=True)
        self.output_path = os.path.join(self.output_dir, f"{self.version_id}.json")

    def run(self) -> Dict[str, Any]:
        tests = self._read_json(self.testing_queries_path)

        results: Dict[str, Any] = {}
        row_counter = 0
        total_iterations = len(tests) * self.repeats

        print(f"\n{'=' * 60}")
        print("Starting RAG Comparison Batch Testing")
        print(f"Total queries: {len(tests)} | Repeats per query: {self.repeats}")
        print(f"Total iterations: {total_iterations}")
        print(f"{'=' * 60}\n")

        start_time = time.time()

        with tqdm(total=total_iterations, desc="Processing comparisons", unit="query") as pbar:
            for query_id, payload in tests.items():
                question = self._extract_question(payload, query_id)
                concept = self._extract_optional_str(payload, "concept")
                task_type = self._extract_optional_str(payload, "task_type") or "compare_definition"

                for _ in range(self.repeats):
                    row_counter += 1
                    row_name = f"test_id_{row_counter}"
                    pbar.set_description(f"Processing {query_id}")

                    llm_start = time.time()
                    comparison_result = self.response_generator.answer_compare_definitions(
                        question=question,
                        concept=concept,
                        top_k_per_source=self.top_k_per_source,
                        rerank_top_k=self.rerank_top_k,
                    )
                    llm_elapsed = time.time() - llm_start
                    pbar.set_postfix({"LLM_time": f"{llm_elapsed:.1f}s"})

                    results[row_name] = {
                        "query_id": query_id,
                        "query_content": question,
                        "concept": concept,
                        "task_type": task_type,
                        "long_version_id": self.long_version_id,
                        "short_version_id": self.short_version_id,
                        "retrieved_docs": comparison_result.get("retrieved_docs", {}),
                        "answers": comparison_result.get("answer"),
                    }

                    pbar.update(1)

        elapsed_time = time.time() - start_time
        avg_time = elapsed_time / total_iterations if total_iterations > 0 else 0

        print(f"\n{'=' * 60}")
        print("Comparison testing completed!")
        print(f"Total time: {elapsed_time:.2f}s ({elapsed_time / 60:.1f} min)")
        print(f"Average time per query: {avg_time:.2f}s")
        print(f"Output saved to: {self.output_path}")
        print(f"{'=' * 60}\n")

        self._write_json(self.output_path, results)
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


if __name__ == "__main__":
    tester = RAGComparisonBatchTester()
    tester.run()


