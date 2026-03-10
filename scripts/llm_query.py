import argparse
from typing import Iterable, List, Optional, Tuple

from healthcare_rag_llm.filters.load_metadata import build_filter_extractor
from healthcare_rag_llm.llm.llm_client import LLMClient
from healthcare_rag_llm.llm.response_generator import ResponseGenerator


QA_TEST_CASES: List[str] = [
    "When did redetrmination begin for the COVID-19 Public Health Emergency unwind in New York State",
    "When did the public health emergency end?",
    "When submitting a claim for Brixandi, how many units should be indicated on the claim?",
    "What rate codes should FQHCs use to bill for audio only telehealth?",
    "Give me a chronological list of the commissioners and what year they first appeared in the medicaid updates.",
    "What are the requirements for appointment scheduling in the medicaid model contract for urgent care?",
    "When did the pharmacy carve out occur?",
    "What are the key components of the SCN program in the NYHER Waiver?",
    "What constitutes RRP referral requirements?",
    "what are the requirements for a referral for enrollment in the childrens waiver?",
    "What are REC services offered to NYS providers?",
]


COMPARE_TEST_CASES: List[Tuple[str, str]] = [
    (
        "Compare Medicare versus provider manual guidance for dual eligible pharmacy claims.",
        "dual eligible pharmacy claims",
    ),
    (
        "Compare policy documents versus provider manual guidance for Brixadi billing units.",
        "Brixadi billing units",
    ),
    (
        "Compare policy documents versus provider manual guidance for telehealth audio-only billing.",
        "telehealth audio-only billing",
    ),
]


def build_response_generator() -> ResponseGenerator:
    llm_client = LLMClient(
        api_key="",
        model="gpt-5",
        provider="openai",
        base_url="https://api.bltcy.ai/v1",
    )
    filter_extractor = build_filter_extractor()
    return ResponseGenerator(llm_client, filter_extractor=filter_extractor)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ad hoc LLM tests for either standard QA or compare-definition prompts."
    )
    parser.add_argument(
        "--mode",
        choices=["qa", "compare"],
        default="qa",
        help="Select the standard QA pipeline or the compare-definition pipeline.",
    )
    parser.add_argument(
        "--question",
        action="append",
        help=(
            "Custom question to run. Repeat the flag to run multiple questions. "
            "If omitted, built-in sample questions are used."
        ),
    )
    parser.add_argument(
        "--concept",
        action="append",
        help=(
            "Optional concept for compare mode. Repeat in the same order as --question. "
            "If omitted, the question text is used as the concept."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k chunks for QA mode.",
    )
    parser.add_argument(
        "--top-k-per-source",
        type=int,
        default=5,
        help="Top-k chunks per source for compare mode.",
    )
    parser.add_argument(
        "--rerank-top-k",
        type=int,
        default=20,
        help="Initial retrieval depth before reranking.",
    )
    return parser.parse_args()


def get_compare_cases(
    questions: Optional[Iterable[str]],
    concepts: Optional[Iterable[str]],
) -> List[Tuple[str, str]]:
    if not questions:
        return COMPARE_TEST_CASES

    question_list = list(questions)
    concept_list = list(concepts or [])

    if concept_list and len(concept_list) != len(question_list):
        raise ValueError("--concept must be provided the same number of times as --question in compare mode.")

    if not concept_list:
        concept_list = question_list

    return list(zip(question_list, concept_list))


def run_qa_tests(
    response_generator: ResponseGenerator,
    questions: List[str],
    top_k: int,
    rerank_top_k: int,
) -> None:
    for index, question in enumerate(questions, start=1):
        response = response_generator.answer_question(
            question,
            top_k=top_k,
            rerank_top_k=rerank_top_k,
        )
        print(f"Question {index}: {question}\n")
        print(response["answer"])
        print("-" * 100)


def run_compare_tests(
    response_generator: ResponseGenerator,
    compare_cases: List[Tuple[str, str]],
    top_k_per_source: int,
    rerank_top_k: int,
) -> None:
    for index, (question, concept) in enumerate(compare_cases, start=1):
        response = response_generator.answer_compare_definitions(
            question,
            concept=concept,
            top_k_per_source=top_k_per_source,
            rerank_top_k=rerank_top_k,
        )
        print(f"Compare Question {index}: {question}")
        print(f"Target Concept: {concept}\n")
        print(response["answer"])
        print("-" * 100)


def main() -> None:
    args = parse_args()
    response_generator = build_response_generator()

    if args.mode == "qa":
        questions = list(args.question) if args.question else QA_TEST_CASES
        run_qa_tests(
            response_generator=response_generator,
            questions=questions,
            top_k=args.top_k,
            rerank_top_k=args.rerank_top_k,
        )
        return

    compare_cases = get_compare_cases(args.question, args.concept)
    run_compare_tests(
        response_generator=response_generator,
        compare_cases=compare_cases,
        top_k_per_source=args.top_k_per_source,
        rerank_top_k=args.rerank_top_k,
    )


if __name__ == "__main__":
    main()
