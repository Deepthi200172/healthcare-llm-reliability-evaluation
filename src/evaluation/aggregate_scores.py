"""
Aggregates all four reliability metrics into a single final score.

Final Reliability = 0.30 * F + 0.30 * H_safety + 0.20 * C + 0.20 * R
"""

from datetime import datetime
from typing import Dict

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_WEIGHTS = {
    "factuality": 0.30,
    "hallucination_safety": 0.30,
    "consistency": 0.20,
    "robustness": 0.20,
}


def aggregate_metric(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Return mean score per (model_name, model_type, pipeline_name), ignoring -1 (errors)."""
    valid = df[df[score_col] >= 0].copy()
    return (
        valid.groupby(["model_name", "model_type", "pipeline_name"])[score_col]
        .agg(mean=("mean"), n=("count"))
        .reset_index()
        .rename(columns={"mean": score_col, "n": f"n_{score_col}"})
    )


def compute_final_reliability(
    factuality_df: pd.DataFrame,
    hallucination_df: pd.DataFrame,
    consistency_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """
    Merge all four metric DataFrames and compute the weighted final reliability score.

    Expected columns per input:
      factuality_df   : model_name, model_type, pipeline_name, factuality_score
      hallucination_df: model_name, model_type, pipeline_name, hallucination_safety
      consistency_df  : model_name, model_type, pipeline_name, consistency_score
      robustness_df   : model_name, model_type, pipeline_name, robustness
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    keys = ["model_name", "model_type", "pipeline_name"]

    # --- factuality ---
    f_agg = (
        factuality_df[factuality_df["factuality_score"] >= 0]
        .groupby(keys)
        .agg(factuality=("factuality_score", "mean"), n_factuality=("factuality_score", "count"))
        .reset_index()
    )

    # --- hallucination safety ---
    h_agg = (
        hallucination_df[hallucination_df["hallucination_safety"] >= 0]
        .groupby(keys)
        .agg(hallucination_safety=("hallucination_safety", "mean"))
        .reset_index()
    )

    # --- consistency ---
    c_agg = (
        consistency_df[consistency_df["consistency_score"] >= 0]
        .groupby(keys)
        .agg(consistency=("consistency_score", "mean"))
        .reset_index()
    )

    # robustness_df already has one row per model/pipeline
    r_agg = robustness_df[keys + ["robustness"]].copy()

    merged = f_agg.merge(h_agg, on=keys, how="outer")
    merged = merged.merge(c_agg, on=keys, how="outer")
    merged = merged.merge(r_agg, on=keys, how="outer")

    merged["final_reliability_score"] = (
        weights["factuality"] * merged["factuality"].fillna(0)
        + weights["hallucination_safety"] * merged["hallucination_safety"].fillna(0)
        + weights["consistency"] * merged["consistency"].fillna(0)
        + weights["robustness"] * merged["robustness"].fillna(0)
    )
    merged["num_samples"] = merged.get("n_factuality", 0).fillna(0).astype(int)
    merged["timestamp"] = datetime.utcnow().isoformat()

    output_cols = [
        "model_name", "model_type", "pipeline_name",
        "factuality", "hallucination_safety", "consistency", "robustness",
        "final_reliability_score", "num_samples", "timestamp",
    ]
    return merged[output_cols].sort_values(
        ["pipeline_name", "model_type", "final_reliability_score"],
        ascending=[True, True, False],
    ).reset_index(drop=True)
