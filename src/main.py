"""
Entry point — runs the full evaluation pipeline end-to-end.

For step-by-step control, run the numbered scripts in experiments/ directly.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from src.utils.logger import get_logger

logger = get_logger("main")

STEPS = [
    ("01_prepare_data.py", "Prepare and sample PubMedQA"),
    ("02_run_generation.py", "Run baseline and RAG generation"),
    ("03_run_consistency_generation.py", "Run consistency generation (5 runs)"),
    ("04_create_perturbations.py", "Create question perturbations"),
    ("05_run_robustness_generation.py", "Run robustness generation"),
    ("06_run_llm_judge.py", "Run LLM-as-Judge evaluation"),
    ("07_aggregate_results.py", "Aggregate and save final results"),
]


def run_step(script: str) -> int:
    path = Path(__file__).parent.parent / "experiments" / script
    logger.info(f"Running: {script}")
    result = subprocess.run([sys.executable, str(path)], check=False)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="LLM Healthcare Reliability Evaluation — full pipeline runner"
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=range(1, 8),
        help="Which steps to run (1-7). Runs all steps by default.",
    )
    args = parser.parse_args()

    steps_to_run = args.steps or list(range(1, 8))

    logger.info("=" * 60)
    logger.info("LLM Healthcare Reliability Evaluation")
    logger.info("=" * 60)

    for step_num in steps_to_run:
        script, description = STEPS[step_num - 1]
        logger.info(f"\nStep {step_num}: {description}")
        code = run_step(script)
        if code != 0:
            logger.error(f"Step {step_num} failed with exit code {code}. Aborting.")
            sys.exit(code)

    logger.info("\nAll steps completed successfully.")
    logger.info("Results saved to: outputs/final_results/model_pipeline_reliability_scores.csv")


if __name__ == "__main__":
    main()
