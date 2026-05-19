"""CLI wrapper for the end-to-end compare evaluation pipeline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from healthcare_rag_llm.pipelines.compare_eval_pipeline import main


if __name__ == "__main__":
    raise SystemExit(main())