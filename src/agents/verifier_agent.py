import json
from typing import Dict

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT_TEMPLATE = """Check whether the answer is fully supported by the retrieved evidence.

Question: {question}
Answer: {answer}

Retrieved Evidence:
{evidence}

Return valid JSON only — no other text:
{{
  "verified_answer": "final verified answer text (keep or revise the original answer)",
  "support_status": "fully_supported/partially_supported/unsupported",
  "verification_reason": "brief explanation of why the answer is or is not supported"
}}"""


class VerifierAgent:
    """
    Agent 4: Verifies whether the generated answer is supported by the evidence.

    Output schema:
        verified_answer     — revised or confirmed answer
        support_status      — fully_supported / partially_supported / unsupported
        verification_reason — explanation
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def verify(self, question: str, answer: str, evidence: str) -> Dict:
        prompt = _PROMPT_TEMPLATE.format(
            question=question, answer=answer, evidence=evidence
        )
        response = self.model.generate(prompt)
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"VerifierAgent JSON parse failed: {e}")

        return {
            "verified_answer": answer,
            "support_status": "partially_supported",
            "verification_reason": "Could not parse verification response.",
        }
