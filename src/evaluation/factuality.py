from typing import Dict

from src.evaluation.judge_prompts import FACTUALITY_PROMPT
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_factuality(judge: LLMJudge, row: Dict) -> Dict:
    """
    Score factual accuracy for a single generated response.

    F = score in [0.0, 1.0]   (1.0 = fully factual)
    """
    prompt = FACTUALITY_PROMPT.format(
        question=row["question"],
        reference_answer=row["reference_answer"],
        reference_decision=row["reference_decision"],
        generated_answer=row["generated_answer"],
        retrieved_context=row.get("retrieved_context", ""),
    )
    result = judge.evaluate_factuality(prompt)
    return {
        "sample_id": row["sample_id"],
        "model_name": row["model_name"],
        "model_type": row["model_type"],
        "pipeline_name": row["pipeline_name"],
        "run_id": row.get("run_id", 1),
        "factuality_score": result["score"],
        "reason": result.get("reason", ""),
        "evidence": result.get("evidence", ""),
    }
