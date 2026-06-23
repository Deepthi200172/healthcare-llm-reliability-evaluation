"""Tests for pipeline logic using the MockModel."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.models.base_model import ModelConfig
from src.models.proprietary_models import MockModel
from src.pipelines.baseline_pipeline import BaselinePipeline


SYSTEM_PROMPT = "You are a medical expert."
USER_TEMPLATE = "Question: {question}\nAnswer:"

SAMPLE = {
    "id": "test_001",
    "question": "Does aspirin reduce fever?",
    "long_answer": "Yes, aspirin is an antipyretic.",
    "final_decision": "yes",
}


@pytest.fixture
def mock_model():
    cfg = ModelConfig(
        name="mock",
        model_type="proprietary",
        provider="mock",
        model_id="mock",
    )
    return MockModel(cfg)


class TestBaselinePipeline:
    def test_run_returns_model_response(self, mock_model):
        pipeline = BaselinePipeline(mock_model, SYSTEM_PROMPT, USER_TEMPLATE)
        result = pipeline.run(SAMPLE, run_id=1)
        assert result.sample_id == "test_001"
        assert result.pipeline_name == "baseline"
        assert result.model_name == "mock"
        assert result.retrieved_context == ""
        assert result.generated_answer != ""

    def test_run_batch_returns_list(self, mock_model):
        pipeline = BaselinePipeline(mock_model, SYSTEM_PROMPT, USER_TEMPLATE)
        results = pipeline.run_batch([SAMPLE, SAMPLE], run_id=1)
        assert len(results) == 2

    def test_to_dict_has_required_fields(self, mock_model):
        pipeline = BaselinePipeline(mock_model, SYSTEM_PROMPT, USER_TEMPLATE)
        result = pipeline.run(SAMPLE, run_id=1)
        d = result.to_dict()
        required = [
            "sample_id", "question", "reference_answer", "reference_decision",
            "model_name", "model_type", "pipeline_name", "run_id",
            "generated_answer", "retrieved_context", "timestamp",
        ]
        for col in required:
            assert col in d, f"Missing column: {col}"
