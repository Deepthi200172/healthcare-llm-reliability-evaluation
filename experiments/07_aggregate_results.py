"""
Step 7 — Aggregate Results

Reads all judge scores, computes the four metric averages per (model, pipeline),
applies the weighted reliability formula, and saves the final comparison table.

Final Reliability = 0.30 * F + 0.30 * H_safety + 0.20 * C + 0.20 * R

Output: outputs/final_results/model_pipeline_reliability_scores.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yaml

from src.evaluation.aggregate_scores import compute_final_reliability
from src.evaluation.robustness import compute_robustness_scores
from src.utils.file_utils import save_csv
from src.utils.logger import get_logger

logger = get_logger("07_aggregate_results")


def main():
    logger.info("=== Step 7: Aggregate Results ===")

    with open("configs/evaluation.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]
    weights = cfg["final_reliability_weights"]

    # Load all judge scores
    factuality_df = pd.read_csv(paths["factuality_scores"])
    hallucination_df = pd.read_csv(paths["hallucination_scores"])
    consistency_df = pd.read_csv(paths["consistency_scores"])
    robustness_raw_df = pd.read_csv(paths["robustness_scores"])

    logger.info(
        f"Loaded: factuality={len(factuality_df)}, "
        f"hallucination={len(hallucination_df)}, "
        f"consistency={len(consistency_df)}, "
        f"robustness={len(robustness_raw_df)}"
    )

    # Compute robustness score (needs original factuality for drop calculation)
    robustness_df = compute_robustness_scores(factuality_df, robustness_raw_df)

    # Compute final reliability
    final_df = compute_final_reliability(
        factuality_df=factuality_df,
        hallucination_df=hallucination_df,
        consistency_df=consistency_df,
        robustness_df=robustness_df,
        weights=weights,
    )

    output_path = paths["final_results"]
    save_csv(final_df.to_dict("records"), output_path)

    logger.info(f"\nFinal reliability scores saved to {output_path}")
    logger.info("\nPreview:")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
