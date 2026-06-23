"""Tests for metric utility functions and aggregate scoring."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

from src.utils.metrics_utils import compute_stats, normalize_score, extract_final_decision
from src.evaluation.aggregate_scores import compute_final_reliability


class TestComputeStats:
    def test_basic_stats(self):
        stats = compute_stats([0.8, 0.9, 1.0])
        assert abs(stats["mean"] - 0.9) < 1e-6
        assert stats["n"] == 3

    def test_ignores_negative_scores(self):
        stats = compute_stats([0.8, -1.0, 0.6])  # -1 = error, should be ignored
        assert stats["n"] == 2

    def test_empty_list_returns_zeros(self):
        stats = compute_stats([])
        assert stats["mean"] == 0.0
        assert stats["n"] == 0


class TestNormalizeScore:
    def test_midpoint(self):
        assert normalize_score(5.0, 0.0, 10.0) == 0.5

    def test_same_min_max_no_division_error(self):
        result = normalize_score(5.0, 5.0, 5.0)
        assert result == 0.0


class TestExtractFinalDecision:
    def test_extracts_yes(self):
        assert extract_final_decision("The answer is yes") == "yes"

    def test_extracts_no_at_end(self):
        assert extract_final_decision("Based on the evidence, no") == "no"

    def test_extracts_maybe(self):
        assert extract_final_decision("Final decision: maybe") == "maybe"

    def test_unknown_for_unrecognized(self):
        assert extract_final_decision("This is a complex topic.") == "unknown"


class TestComputeFinalReliability:
    def _make_df(self, n: int = 3) -> tuple:
        keys = ["model_name", "model_type", "pipeline_name"]
        rows = [{"model_name": "m1", "model_type": "proprietary", "pipeline_name": "baseline"}] * n

        fact = pd.DataFrame([{**r, "factuality_score": 0.8} for r in rows])
        hall = pd.DataFrame([{**r, "hallucination_safety": 0.7} for r in rows])
        cons = pd.DataFrame([{**r, "consistency_score": 0.9} for r in rows])
        rob = pd.DataFrame([{**r, "robustness": 0.85} for r in [rows[0]]])
        return fact, hall, cons, rob

    def test_final_score_computed(self):
        fact, hall, cons, rob = self._make_df()
        result = compute_final_reliability(fact, hall, cons, rob)
        assert len(result) == 1
        score = result["final_reliability_score"].iloc[0]
        expected = 0.30 * 0.8 + 0.30 * 0.7 + 0.20 * 0.9 + 0.20 * 0.85
        assert abs(score - expected) < 1e-6

    def test_output_columns_present(self):
        fact, hall, cons, rob = self._make_df()
        result = compute_final_reliability(fact, hall, cons, rob)
        required = [
            "model_name", "model_type", "pipeline_name",
            "factuality", "hallucination_safety", "consistency", "robustness",
            "final_reliability_score", "num_samples", "timestamp",
        ]
        for col in required:
            assert col in result.columns, f"Missing column: {col}"
