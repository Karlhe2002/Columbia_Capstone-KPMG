# How to Set Up Testing Queries

This document explains how to create and configure testing queries for evaluating the RAG system.

## 1. Location

Place your testing query files in the following directory:
```
./data/testing_queries
```

## 2. File Format

Testing query files must be in **JSON format**.

### JSON Structure

Each file should contain a dictionary with test queries as keys. Each test query includes:

```json
{
  "test_query_1": {
    "question": "Your question for the LLM",
    "document": ["ground_truth_doc_1.pdf", "ground_truth_doc_2.pdf"],
    "Answer": "The reference ground truth answer"
  },
  "test_query_2": {
    "question": "Another question for the LLM",
    "document": ["relevant_doc.pdf"],
    "Answer": "Expected answer for evaluation"
  }
}
```

### Field Descriptions

- **Key** (e.g., `"test_query_1"`): Unique identifier for each test query
- **`question`**: The query/question to ask the LLM
- **`document`**: Array of ground truth document names that should be retrieved
- **`Answer`**: The reference answer used for evaluation

## 3. Example

```json
{
  "diabetes_treatment": {
    "question": "What are the recommended treatments for type 2 diabetes?",
    "document": ["diabetes_guidelines_2024.pdf", "endocrinology_handbook.pdf"],
    "Answer": "First-line treatment for type 2 diabetes includes lifestyle modifications (diet and exercise) and metformin as initial pharmacotherapy."
  },
  "hypertension_diagnosis": {
    "question": "What are the diagnostic criteria for hypertension?",
    "document": ["cardiovascular_guidelines.pdf"],
    "Answer": "Hypertension is diagnosed when blood pressure readings are consistently ≥130/80 mmHg."
  }
}
```

## Notes

- Ensure all document names in the `document` field match actual files in your knowledge base
- Use descriptive names for test query keys to easily identify them in evaluation results
- The `Answer` field should contain accurate, reference-standard responses for meaningful evaluation

## 4. Running Prompt Tests with `llm_query.py`

Use [`scripts/llm_query.py`](/c:/Users/22840/OneDrive/Desktop/KPMG/Columbia_Capstone-KPMG/scripts/llm_query.py) when you want to quickly test prompt behavior without building a full evaluation JSON file.

### Run the original QA prompt

```powershell
python scripts/llm_query.py --mode qa
```

This runs the standard `answer_question(...)` flow.

### Run the comparison prompt

```powershell
python scripts/llm_query.py --mode compare
```

This runs the `answer_compare_definitions(...)` flow, which uses `configs/system_prompt_compare.txt` and retrieves from both:

- `policy` documents
- `provider_manual` documents

### Run one custom QA question

```powershell
python scripts/llm_query.py --mode qa --question "When did the pharmacy carve out occur?"
```

### Run one custom compare question

```powershell
python scripts/llm_query.py --mode compare --question "Compare Medicare versus provider manual guidance for dual eligible pharmacy claims." --concept "dual eligible pharmacy claims"
```

### Optional retrieval settings

```powershell
python scripts/llm_query.py --mode compare --top-k-per-source 5 --rerank-top-k 20
```

Notes:

- Use `--mode qa` for normal single-answer testing.
- Use `--mode compare` for side-by-side policy vs provider manual testing.
- In compare mode, `--concept` is optional. If omitted, the script uses the question text as the concept.
- The JSON format described above is still the correct format for stored evaluation query files such as `data/testing_queries/testing_query.json`.
