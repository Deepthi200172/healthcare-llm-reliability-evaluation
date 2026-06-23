import json
import os
from typing import Dict

from tenacity import retry, stop_after_attempt, wait_exponential

from src.models.base_model import ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

_JUDGE_SYSTEM = (
    "You are a medical AI evaluation expert. "
    "Always respond with valid JSON containing exactly the keys requested. "
    "Never add markdown formatting or extra text."
)


class LLMJudge:
    """
    Wrapper around a judge LLM (default: GPT-4o-mini via OpenAI).

    All evaluate_* methods return a dict with:
        score   : float in [0.0, 1.0]
        reason  : str
        evidence: str
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None

    def _init(self):
        if self._client is not None:
            return
        if self.config.provider == "openai":
            import openai
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise EnvironmentError("OPENAI_API_KEY is not set")
            self._client = openai.OpenAI(api_key=api_key)
        else:
            raise NotImplementedError(
                f"Judge provider {self.config.provider!r} not yet implemented. "
                "Use 'openai' for now."
            )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _call_openai(self, prompt: str) -> str:
        self._init()
        response = self._client.chat.completions.create(
            model=self.config.model_id,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _evaluate(self, prompt: str, dimension: str) -> Dict:
        """Core evaluation call with JSON parsing and graceful fallback."""
        try:
            raw = self._call_openai(prompt)
            result = json.loads(raw)
            score = float(result.get("score", -1.0))
            if not 0.0 <= score <= 1.0:
                logger.warning(
                    f"Judge returned out-of-range score={score} for {dimension}; clamping."
                )
                score = max(0.0, min(1.0, score))
            result["score"] = score
            result["dimension"] = dimension
            return result
        except Exception as e:
            logger.error(f"Judge failed for dimension={dimension}: {e}")
            return {
                "score": -1.0,
                "reason": f"Evaluation error: {e}",
                "evidence": "",
                "dimension": dimension,
            }

    def evaluate_factuality(self, prompt: str) -> Dict:
        return self._evaluate(prompt, "factuality")

    def evaluate_hallucination(self, prompt: str) -> Dict:
        return self._evaluate(prompt, "hallucination")

    def evaluate_consistency(self, prompt: str) -> Dict:
        return self._evaluate(prompt, "consistency")

    def evaluate_robustness_factuality(self, prompt: str) -> Dict:
        return self._evaluate(prompt, "robustness_factuality")
