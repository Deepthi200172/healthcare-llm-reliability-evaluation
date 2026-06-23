from typing import Dict, List

import pandas as pd

from src.evaluation.judge_prompts import ROBUSTNESS_FACTUALITY_PROMPT
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_perturbed_factuality(judge: LLMJudge, row: Dict) -> Dict:
    """
    Score factuality for a perturbed-question response.
    The reference is always the original reference answer.

    Returns the perturbed factuality score in [0.0, 1.0].
    """
    prompt = ROBUSTNESS_FACTUALITY_PROMPT.format(
        original_question=row["original_question"],
        perturbed_question=row["question"],
        perturbation_type=row["perturbation_type"],
        reference_answer=row["reference_answer"],
        reference_decision=row["reference_decision"],
        generated_answer=row["generated_answer"],
    )
    result = judge.evaluate_robustness_factuality(prompt)
    return {
        "sample_id": row["sample_id"],
        "model_name": row["model_name"],
        "model_type": row["model_type"],
        "pipeline_name": row["pipeline_name"],
        "perturbation_type": row["perturbation_type"],
        "perturbed_factuality_score": result["score"],
        "reason": result.get("reason", ""),
        "evidence": result.get("evidence", ""),
    }


def compute_robustness_drop(
    original_factuality: float,
    perturbed_factuality: float,
) -> float:
    """
    Robustness Drop = max(0, original_F - perturbed_F)
    (If perturbed is better than original, drop is 0.)
    """
    if original_factuality < 0 or perturbed_factuality < 0:
        return -1.0  # flag as invalid
    return max(0.0, original_factuality - perturbed_factuality)


def compute_robustness_scores(
    factuality_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge original factuality scores with perturbed factuality scores,
    compute per-sample robustness drop, then aggregate per model/pipeline.

    Formula:
        drop_i      = max(0, F_original_i - F_perturbed_i)
        R_perturb   = 1 - mean(drop_i)   for each perturbation type
        R_overall   = mean across perturbation types
    """
    orig = factuality_df[["sample_id", "model_name", "model_type", "pipeline_name", "factuality_score"]].copy()
    orig = orig[orig["factuality_score"] >= 0]

    merged = robustness_df.merge(
        orig,
        on=["sample_id", "model_name", "model_type", "pipeline_name"],
        how="left",
    )
    merged["drop"] = merged.apply(
        lambda r: compute_robustness_drop(r["factuality_score"], r["perturbed_factuality_score"]),
        axis=1,
    )
    valid = merged[merged["drop"] >= 0]

    agg = (
        valid.groupby(["model_name", "model_type", "pipeline_name", "perturbation_type"])["drop"]
        .mean()
        .reset_index()
        .rename(columns={"drop": "mean_drop"})
    )
    agg["robustness_score"] = 1.0 - agg["mean_drop"]

    # Overall robustness = mean across perturbation types
    overall = (
        agg.groupby(["model_name", "model_type", "pipeline_name"])["robustness_score"]
        .mean()
        .reset_index()
        .rename(columns={"robustness_score": "robustness"})
    )
    return overall
