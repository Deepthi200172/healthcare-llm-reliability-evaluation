"""
Step 1 — Prepare Data

Loads raw PubMedQA from HuggingFace, samples 100 examples,
and saves a clean CSV to data/processed/pubmedqa_sample_100.csv.
"""

import sys
from pathlib import Path

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import yaml
from src.data.preprocess_pubmedqa import preprocess_pubmedqa
from src.utils.logger import get_logger

logger = get_logger("01_prepare_data")


def main():
    logger.info("=== Step 1: Prepare PubMedQA Data ===")

    with open("configs/evaluation.yaml") as f:
        cfg = yaml.safe_load(f)

    sample_size = cfg["dataset_sample_size"]
    output_path = cfg["paths"]["processed_data"]

    df = preprocess_pubmedqa(
        sample_size=sample_size,
        seed=42,
        split="train",
        output_path=output_path,
    )

    logger.info(f"Done. {len(df)} samples saved to {output_path}")
    logger.info(f"Label distribution:\n{df['final_decision'].value_counts().to_string()}")


if __name__ == "__main__":
    main()
