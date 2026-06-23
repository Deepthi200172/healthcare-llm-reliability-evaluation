import time
from datetime import datetime
from typing import Dict, List

from src.models.base_model import BaseModel, ModelResponse
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselinePipeline:
    """
    Pipeline 1 — No retrieval.
    Sends the question directly to the LLM and returns a ModelResponse.
    """

    def __init__(self, model: BaseModel, system_prompt: str, user_prompt_template: str):
        self.model = model
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template

    def run(self, sample: Dict, run_id: int = 1) -> ModelResponse:
        """Run the model on a single sample dict."""
        question = sample["question"]
        prompt = self.user_prompt_template.format(question=question)

        start = time.time()
        error = None
        answer = ""
        try:
            answer = self.model.generate(prompt, self.system_prompt)
        except Exception as e:
            error = str(e)
            logger.error(f"Baseline error [{self.model.config.name}] sample={sample['id']}: {e}")
        elapsed = round(time.time() - start, 3)

        return ModelResponse(
            sample_id=str(sample["id"]),
            question=question,
            reference_answer=sample.get("long_answer", ""),
            reference_decision=sample.get("final_decision", ""),
            model_name=self.model.config.name,
            model_type=self.model.config.model_type,
            pipeline_name="baseline",
            run_id=run_id,
            generated_answer=answer,
            retrieved_context="",
            timestamp=datetime.utcnow().isoformat(),
            error=error,
        )

    def run_batch(self, samples: List[Dict], run_id: int = 1) -> List[ModelResponse]:
        results = []
        for i, sample in enumerate(samples, 1):
            logger.info(
                f"[baseline][{self.model.config.name}] run={run_id} "
                f"sample {i}/{len(samples)} (id={sample['id']})"
            )
            results.append(self.run(sample, run_id=run_id))
        return results
