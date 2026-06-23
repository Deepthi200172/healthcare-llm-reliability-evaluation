"""Tests for evaluation logic (without calling the real judge API)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from unittest.mock import MagicMock, patch

from src.evaluation.hallucination import evaluate_hallucination
from src.evaluation.factuality import evaluate_factuality
from src.evaluation.consistency import evaluate_consistency, build_consistency_records
from src.evaluation.robustness import compute_robustness_drop


SAMPLE_ROW = {
    "sample_id": "s1",
    "question": "Does drug X treat disease Y?",
    "reference_answer": "Yes, drug X is effective.",
    "reference_decision": "yes",
    "generated_answer": "Drug X is highly effective for Y.",
    "retrieved_context": "Studies show drug X reduces symptoms of Y.",
    "model_name": "gpt-4o-mini",
    "model_type": "proprietary",
    "pipeline_name": "baseline",
    "run_id": 1,
}


def _make_mock_judge(score: float, reason: str = "test reason"):
    judge = MagicMock()
    judge.evaluate_factuality.return_value = {
        "score": score, "reason": reason, "evidence": "test", "dimension": "factuality"
    }
    judge.evaluate_hallucination.return_value = {
        "score": score, "reason": reason, "evidence": "test", "dimension": "hallucination"
    }
    judge.evaluate_consistency.return_value = {
        "score": score, "reason": reason, "evidence": "test", "dimension": "consistency"
    }
    return judge


class TestFactuality:
    def test_score_returned(self):
        judge = _make_mock_judge(0.9)
        result = evaluate_factuality(judge, SAMPLE_ROW)
        assert result["factuality_score"] == 0.9
        assert result["sample_id"] == "s1"

    def test_result_has_required_keys(self):
        judge = _make_mock_judge(0.5)
        result = evaluate_factuality(judge, SAMPLE_ROW)
        for key in ["sample_id", "model_name", "model_type", "pipeline_name", "factuality_score"]:
            assert key in result


class TestHallucination:
    def test_safety_is_inverse_of_severity(self):
        judge = _make_mock_judge(0.3)  # severity = 0.3
        result = evaluate_hallucination(judge, SAMPLE_ROW)
        assert abs(result["hallucination_safety"] - 0.7) < 1e-9

    def test_zero_severity_gives_full_safety(self):
        judge = _make_mock_judge(0.0)
        result = evaluate_hallucination(judge, SAMPLE_ROW)
        assert result["hallucination_safety"] == 1.0


class TestConsistency:
    def test_returns_score(self):
        judge = _make_mock_judge(0.8)
        result = evaluate_consistency(judge, "s1", "What is X?", ["A", "A", "A", "A", "A"])
        assert result["consistency_score"] == 0.8

    def test_pads_short_answer_list(self):
        judge = _make_mock_judge(0.6)
        # Only 2 answers instead of 5 — should not raise
        result = evaluate_consistency(judge, "s1", "What is X?", ["A", "B"])
        assert "consistency_score" in result


class TestRobustnessDrop:
    def test_positive_drop(self):
        drop = compute_robustness_drop(0.8, 0.5)
        assert abs(drop - 0.3) < 1e-9

    def test_no_drop_when_perturbed_better(self):
        drop = compute_robustness_drop(0.5, 0.8)
        assert drop == 0.0

    def test_invalid_scores_return_negative_one(self):
        drop = compute_robustness_drop(-1, 0.5)
        assert drop == -1.0
