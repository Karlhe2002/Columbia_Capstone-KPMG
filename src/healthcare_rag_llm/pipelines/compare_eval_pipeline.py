"""End-to-end compare evaluation pipeline.

This wrapper reads a raw compare output that is already in the canonical
compare raw directory, then runs:
1. raw -> processed
2. processed -> llm eval

It reuses the existing CLI scripts so the core conversion/evaluation logic
stays in one place.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_ground_truth_path(repo_root: Path) -> Path:
    return repo_root / "data" / "llm_eval_results_compare" / "Q&A_ground_truth.csv"


def _run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the compare evaluation pipeline end to end",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "-i",
        "--input_raw",
        nargs="+",
        help="One or more raw compare JSON paths (typically under data/llm_eval_results_compare/comparison_raw)",
    )
    input_group.add_argument(
        "--suffix",
        default=None,
        help="Run all raw JSON files in comparison_raw ending with this suffix, for example: _llama.json",
    )
    parser.add_argument(
        "-n",
        "--name",
        default=None,
        help="Experiment name used for output filenames; defaults to the input stem",
    )
    parser.add_argument(
        "-g",
        "--ground_truth",
        default=None,
        help="Optional path to the compare ground-truth CSV",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="API provider passed to the evaluation step",
    )
    parser.add_argument(
        "--model",
        default="gpt-5",
        help="Evaluator model passed to the evaluation step",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for the evaluation step",
    )
    parser.add_argument(
        "--config",
        default="configs/api_config.yaml",
        help="Path to the API config used by the evaluation step",
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    input_raw_files: list[Path] = []
    if args.input_raw:
        for raw_item in args.input_raw:
            input_raw_path = Path(raw_item).expanduser()
            if not input_raw_path.is_absolute():
                input_raw_path = (repo_root / input_raw_path).resolve()
            else:
                input_raw_path = input_raw_path.resolve()
            if not input_raw_path.exists():
                print(f"Error: raw input file not found: {input_raw_path}")
                return 1
            input_raw_files.append(input_raw_path)
    else:
        raw_dir = repo_root / "data" / "llm_eval_results_compare" / "comparison_raw"
        if not raw_dir.exists():
            print(f"Error: comparison_raw directory not found: {raw_dir}")
            return 1

        suffix = str(args.suffix)
        matched_files = sorted(
            [p.resolve() for p in raw_dir.iterdir() if p.is_file() and p.name.endswith(suffix)]
        )
        if not matched_files:
            print(f"Error: no raw files found in {raw_dir} ending with suffix: {suffix}")
            return 1

        input_raw_files.extend(matched_files)

    if len(input_raw_files) > 1 and args.name:
        print("Error: --name only supports single-file mode. Remove --name when passing multiple --input_raw files.")
        return 1

    raw_dir = repo_root / "data" / "llm_eval_results_compare" / "comparison_raw"
    processed_dir = repo_root / "data" / "llm_eval_results_compare" / "comparison_processed"
    eval_dir = repo_root / "data" / "llm_eval_results_compare" / "comparison_llm_eval_results"

    ground_truth = Path(args.ground_truth).expanduser().resolve() if args.ground_truth else _default_ground_truth_path(repo_root)
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = (repo_root / config_path).resolve()
    else:
        config_path = config_path.resolve()

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    convert_script = repo_root / "scripts" / "convert_raw_to_processed_compare.py"
    eval_script = repo_root / "scripts" / "llm_evaluation_compare.py"

    print("=== Compare Eval Pipeline ===")
    print(f"Input count: {len(input_raw_files)}")
    print(f"Ground truth: {ground_truth}")
    print(f"Config: {config_path}")

    eval_ground_truth: Path | None = None
    if ground_truth.suffix.lower() == ".json":
        eval_ground_truth = ground_truth
    else:
        print("Eval step ground truth: skipped (non-JSON ground truth is used only in raw -> processed)")

    for index, input_raw in enumerate(input_raw_files, start=1):
        experiment_name = args.name or input_raw.stem
        raw_target = input_raw
        processed_target = processed_dir / f"{experiment_name}_processed.json"
        eval_target = eval_dir / f"{experiment_name}_llm_eval.json"

        print("\n" + "-" * 60)
        print(f"[{index}/{len(input_raw_files)}] Raw input: {input_raw}")
        print(f"Processed target: {processed_target}")
        print(f"Eval target: {eval_target}")

        convert_command = [
            sys.executable,
            str(convert_script),
            "-i",
            str(raw_target),
            "-o",
            str(processed_target),
            "-g",
            str(ground_truth),
        ]
        print("Running raw -> processed")
        _run_command(convert_command)

        eval_command = [
            sys.executable,
            str(eval_script),
            "-t",
            str(processed_target),
            "-o",
            str(eval_target),
            "--config",
            str(config_path),
        ]
        if eval_ground_truth is not None:
            eval_command.extend(["-g", str(eval_ground_truth)])
        if args.provider:
            eval_command.extend(["--provider", args.provider])
        if args.model:
            eval_command.extend(["--model", args.model])
        if args.limit is not None:
            eval_command.extend(["--limit", str(args.limit)])

        print("Running processed -> llm eval")
        _run_command(eval_command)

    print("\nPipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())