# Comparison Eval Pipeline

## 1) Prepare raw data
- Because different experiment adjustments were run on different branches, copy the compare outputs from the corresponding branch's testing pipeline at the same output location in that branch (usually `data/test_results/comparison_version_update.json`) into:
    `data/llm_eval_results_compare/comparison_raw/input_file.json`
- The `input_file` is a placeholder for the branch-specific feature or experiment name.

## 2) raw -> processed
- Parse answers into structured sections and attach ground truth.

```bash
.venv/bin/python scripts/convert_raw_to_processed_compare.py \
    -i data/llm_eval_results_compare/comparison_raw/input_file.json \
    -o data/llm_eval_results_compare/comparison_processed/input_file_processed.json
```

- Here `input_file_processed.json` means the processed version of the same raw file, with `_processed` appended to the base name.

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

- `input_file_llm_eval.json` means the final evaluation output for that same sample, with `_llm_eval` appended to the base name.
- For a quick smoke test, add: `--limit 1`