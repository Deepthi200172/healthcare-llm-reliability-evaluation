from typing import List

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

_PROMPT = """\
Paraphrase the following biomedical question while preserving its exact meaning.
Keep medical terminology intact. Only rephrase the sentence structure.
Return only the paraphrased question — no explanation, no prefix.

Original: {question}

Paraphrased:"""


class ParaphraseGenerator:
    """
    Generates semantically equivalent rephrasings of biomedical questions
    using an LLM. Used to test Robustness under paraphrase perturbation.
    """

    def __init__(self, model: BaseModel):
        self.model = model

    def generate(self, question: str) -> str:
        prompt = _PROMPT.format(question=question)
        try:
            result = self.model.generate(prompt).strip()
            # Strip common LLM prefixes
            for prefix in ("Paraphrased:", "Paraphrase:", "Answer:", "Rephrased:"):
                if result.startswith(prefix):
                    result = result[len(prefix):].strip()
            return result if result else question
        except Exception as e:
            logger.error(f"Paraphrase failed: {e}")
            return question

    def generate_batch(self, questions: List[str]) -> List[str]:
        return [self.generate(q) for q in questions]
