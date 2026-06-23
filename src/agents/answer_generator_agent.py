import json
from typing import Dict

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """You are a medical expert. Generate an answer using ONLY the evidence provided.
Do not make claims not supported by the evidence.

Question: {question}

Retrieved Evidence:
{evidence}

Return valid JSON only — no other text:
{{
  "answer": "your detailed answer here",
  "short_decision": "yes/no/maybe/unclear"
}}"""


class AnswerGeneratorAgent:
    """
    Agent 3: Generates the final answer grounded in retrieved evidence.

    Output schema:
        answer          — detailed answer text
        short_decision  — yes / no / maybe / unclear
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def generate(self, question: str, evidence: str) -> Dict:
        prompt = _PROMPT_TEMPLATE.format(question=question, evidence=evidence)
        response = self.model.generate(prompt)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"AnswerGeneratorAgent JSON parse failed: {e}")

        # Graceful fallback — treat the whole response as the answer
        return {"answer": response.strip(), "short_decision": "unclear"}
