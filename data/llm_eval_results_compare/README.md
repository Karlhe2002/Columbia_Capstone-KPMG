# Comparison Eval Pipeline

## 1) Prepare raw data
- Copy compare outputs from the testing pipeline (usually `data/test_results/comparison_version_update.json`) into:
- `data/llm_eval_results_compare/comparison_raw/`

## 2) raw -> processed
- Parse answers into structured sections, remove retrieval scoring metadata, and attach ground truth.

```bash
.venv/bin/python scripts/convert_raw_to_process_compare.py \
    -i data/llm_eval_results_compare/comparison_raw/input_file.json \
    -o data/llm_eval_results_compare/comparison_processed/input_file_processed.json
```

- Optional: specify the ground truth CSV explicitly

```bash
.venv/bin/python scripts/convert_raw_to_processed_compare.py \
    -i data/llm_eval_results_compare/comparison_raw/input_file.json \
    -o data/llm_eval_results_compare/comparison_processed/input_file_processed.json \
    -g "data/llm_eval_results_compare/Q&A_ground_truth.csv"
```

## 3) processed -> llm eval
- Evaluation entry script: `scripts/llm_evaluation_compare.py`
- Evaluation logic: `src/healthcare_rag_llm/evaluate/llm_evaluate_compare.py`

```bash
.venv/bin/python scripts/llm_evaluation_compare.py \
    -t data/llm_eval_results_compare/comparison_processed/input_file_processed.json \
    -o data/llm_eval_results_compare/comparison_llm_eval_results/input_file_llm_eval.json \
    --provider openai_official \
    --model gpt-4o-mini
```

- For a quick smoke test, add: `--limit 1`