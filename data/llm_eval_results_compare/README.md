# Comparison Eval Pipeline

## 1) Prepare raw data
- Because different experiment adjustments were run on different branches, copy the compare outputs from the corresponding branch's testing pipeline at the same output location in that branch (usually `data/test_results/result.json`) into:
    `data/llm_eval_results_compare/comparison_raw/input_file.json`
- The `result.json` and `input_file` is a placeholder for the branch-specific feature or experiment name.
- All input files should be renamed to `branch_name_model.json` (for example `dense_sparse_gemini.json`)

## 2) One-shot pipeline
- If you want to run all files in `data/llm_eval_results_compare/comparison_raw` in one command, pass the directory to the wrapper:

```bash
.venv/bin/python scripts/compare_eval_pipeline.py \
    -i data/llm_eval_results_compare/comparison_raw \
    --provider openai_official \
    --model gpt-4o-mini
```

- This runs every JSON file in `data/llm_eval_results_compare/comparison_raw`, then writes the processed and LLM eval outputs under the matching `comparison_processed/` and `comparison_llm_eval_results/` folders.

- You can run only selected files (not all) by passing multiple `-i` values:

```bash
.venv/bin/python scripts/compare_eval_pipeline.py \
    -i data/llm_eval_results_compare/comparison_raw/exp_a.json \
       data/llm_eval_results_compare/comparison_raw/exp_c.json \
    --provider openai_official \
    --model gpt-4o-mini
```

- You can also run files by suffix under `comparison_raw` (for example all files ending with `_llama.json`):

```bash
.venv/bin/python scripts/compare_eval_pipeline.py \
    --suffix _llama.json \
    --provider openai_official \
    --model gpt-4o-mini
```

# Debug / Step-by-step pipeline (recommended for troubleshooting)

## 1) raw -> processed
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

## 2) processed -> llm eval
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