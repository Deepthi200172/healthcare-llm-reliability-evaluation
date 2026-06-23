"""Tests for data loading and preprocessing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd

from src.data.load_pubmedqa import pubmedqa_to_records


class TestPubMedQAToRecords:
    def _make_item(self, pubid="123", question="Is X effective?",
                   contexts=["Abstract text."], long_answer="Yes it is.", decision="yes"):
        return {
            "pubid": pubid,
            "question": question,
            "context": {"contexts": contexts},
            "long_answer": long_answer,
            "final_decision": decision,
        }

    def test_basic_conversion(self):
        items = [self._make_item()]
        records = pubmedqa_to_records(items)
        assert len(records) == 1
        assert records[0]["question"] == "Is X effective?"
        assert records[0]["final_decision"] == "yes"
        assert "Abstract text." in records[0]["context"]

    def test_multiple_context_chunks_joined(self):
        item = self._make_item(contexts=["Chunk 1.", "Chunk 2."])
        records = pubmedqa_to_records([item])
        assert "Chunk 1." in records[0]["context"]
        assert "Chunk 2." in records[0]["context"]

    def test_empty_dataset(self):
        records = pubmedqa_to_records([])
        assert records == []

    def test_missing_fields_do_not_crash(self):
        item = {"pubid": "999"}  # minimal item
        records = pubmedqa_to_records([item])
        assert len(records) == 1
        assert records[0]["question"] == ""


class TestPreprocessPubMedQA:
    def test_label_filtering(self):
        from src.data.preprocess_pubmedqa import VALID_LABELS
        assert "yes" in VALID_LABELS
        assert "no" in VALID_LABELS
        assert "maybe" in VALID_LABELS
        assert "unknown" not in VALID_LABELS
