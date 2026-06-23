import json
from typing import Dict

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """Analyze the following biomedical question.

Question: {question}

Return valid JSON only — no other text:
{{
  "question_type": "yes_no_maybe_medical_question",
  "key_concepts": ["concept1", "concept2"],
  "retrieval_focus": "what specific evidence should be retrieved to answer this question"
}}"""


class QueryAnalyzerAgent:
    """
    Agent 1: Analyzes the medical question to guide better evidence retrieval.

    Output schema:
        question_type   — describes the expected answer format
        key_concepts    — list of important biomedical concepts
        retrieval_focus — free-text description of what evidence to look for
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def analyze(self, question: str) -> Dict:
        prompt = _PROMPT_TEMPLATE.format(question=question)
        response = self.model.generate(prompt)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"QueryAnalyzerAgent JSON parse failed: {e}")

        # Graceful fallback — use the raw question as retrieval focus
        return {
            "question_type": "yes_no_maybe_medical_question",
            "key_concepts": [],
            "retrieval_focus": question,
        }
