from typing import Dict

from src.evaluation.judge_prompts import HALLUCINATION_PROMPT
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_hallucination(judge: LLMJudge, row: Dict) -> Dict:
    """
    Score hallucination severity for a single response, then convert to safety.

    The judge returns hallucination SEVERITY in [0.0, 1.0]:
        0.0 = no hallucination
        1.0 = severe hallucination

    We store both severity and safety:
        hallucination_severity = raw judge score
        hallucination_safety   = 1 - severity  (higher = safer)
    """
    prompt = HALLUCINATION_PROMPT.format(
        question=row["question"],
        reference_answer=row["reference_answer"],
        retrieved_context=row.get("retrieved_context", ""),
        generated_answer=row["generated_answer"],
    )
    result = judge.evaluate_hallucination(prompt)
    severity = result["score"]
    safety = 1.0 - severity if severity >= 0 else -1.0

    return {
        "sample_id": row["sample_id"],
        "model_name": row["model_name"],
        "model_type": row["model_type"],
        "pipeline_name": row["pipeline_name"],
        "run_id": row.get("run_id", 1),
        "hallucination_severity": severity,
        "hallucination_safety": safety,
        "reason": result.get("reason", ""),
        "evidence": result.get("evidence", ""),
    }
