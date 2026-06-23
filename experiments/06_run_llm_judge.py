"""
Step 6 — Run LLM-as-Judge Evaluation

Scores all generated responses across four reliability dimensions:
  - Factuality
  - Hallucination
  - Consistency
  - Robustness

Outputs:
  outputs/judge_scores/factuality_scores.csv
  outputs/judge_scores/hallucination_scores.csv
  outputs/judge_scores/consistency_scores.csv
  outputs/judge_scores/robustness_scores.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yaml
from tqdm import tqdm

from src.evaluation.factuality import evaluate_factuality
from src.evaluation.hallucination import evaluate_hallucination
from src.evaluation.consistency import evaluate_consistency, build_consistency_records
from src.evaluation.robustness import evaluate_perturbed_factuality
from src.evaluation.llm_judge import LLMJudge
from src.utils.config_loader import get_judge_config
from src.utils.file_utils import save_csv
from src.utils.logger import get_logger

logger = get_logger("06_run_llm_judge")


def main():
    logger.info("=== Step 6: Run LLM-as-Judge Evaluation ===")

    with open("configs/evaluation.yaml") as f:
        cfg = yaml.safe_load(f)
    paths = cfg["paths"]

    judge_cfg = get_judge_config()
    judge = LLMJudge(judge_cfg)

    # ------------------------------------------------------------------
    # A) Factuality & Hallucination — from generated_responses.csv
    # ------------------------------------------------------------------
    logger.info("\n[A] Scoring factuality and hallucination...")
    gen_df = pd.read_csv(paths["responses"])

    factuality_records = []
    hallucination_records = []

    for _, row in tqdm(gen_df.iterrows(), total=len(gen_df), desc="Factuality+Hallucination"):
        row_dict = row.to_dict()
        factuality_records.append(evaluate_factuality(judge, row_dict))
        hallucination_records.append(evaluate_hallucination(judge, row_dict))

    save_csv(factuality_records, paths["factuality_scores"])
    save_csv(hallucination_records, paths["hallucination_scores"])
    logger.info(f"Factuality scores: {paths['factuality_scores']}")
    logger.info(f"Hallucination scores: {paths['hallucination_scores']}")

    # ------------------------------------------------------------------
    # B) Consistency — from consistency_responses.csv
    # ------------------------------------------------------------------
    logger.info("\n[B] Scoring consistency...")
    cons_df = pd.read_csv(paths["consistency_responses"])
    cons_groups = build_consistency_records(cons_df)

    consistency_records = []
    for group in tqdm(cons_groups, desc="Consistency"):
        result = evaluate_consistency(
            judge,
            sample_id=group["sample_id"],
            question=group["question"],
            answers=group["answers"],
        )
        result["model_name"] = group["model_name"]
        result["model_type"] = group["model_type"]
        result["pipeline_name"] = group["pipeline_name"]
        consistency_records.append(result)

    save_csv(consistency_records, paths["consistency_scores"])
    logger.info(f"Consistency scores: {paths['consistency_scores']}")

    # ------------------------------------------------------------------
    # C) Robustness — from robustness_responses.csv
    # ------------------------------------------------------------------
    logger.info("\n[C] Scoring robustness (perturbed factuality)...")
    rob_df = pd.read_csv(paths["robustness_responses"])

    robustness_records = []
    for _, row in tqdm(rob_df.iterrows(), total=len(rob_df), desc="Robustness"):
        robustness_records.append(evaluate_perturbed_factuality(judge, row.to_dict()))

    save_csv(robustness_records, paths["robustness_scores"])
    logger.info(f"Robustness scores: {paths['robustness_scores']}")

    logger.info("\nDone. All judge scores saved.")


if __name__ == "__main__":
    main()
