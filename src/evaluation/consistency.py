from typing import Dict, List

import pandas as pd

from src.evaluation.judge_prompts import CONSISTENCY_PROMPT
from src.evaluation.llm_judge import LLMJudge
from src.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_consistency(judge: LLMJudge, sample_id: str, question: str, answers: List[str]) -> Dict:
    """
    Score consistency across N runs for a single question.

    Expects exactly 5 answers (runs_per_question_for_consistency = 5).
    Pads with empty strings if fewer runs are available.

    C = score in [0.0, 1.0]   (1.0 = perfectly consistent)
    """
    padded = (answers + [""] * 5)[:5]
    prompt = CONSISTENCY_PROMPT.format(
        question=question,
        answer_1=padded[0],
        answer_2=padded[1],
        answer_3=padded[2],
        answer_4=padded[3],
        answer_5=padded[4],
    )
    result = judge.evaluate_consistency(prompt)
    return {
        "sample_id": sample_id,
        "consistency_score": result["score"],
        "reason": result.get("reason", ""),
        "evidence": result.get("evidence", ""),
    }


def build_consistency_records(df: pd.DataFrame) -> List[Dict]:
    """
    Group a consistency_responses DataFrame by (sample_id, model_name, pipeline_name)
    and return one record per group containing the question and list of answers.
    """
    records = []
    for (sample_id, model_name, model_type, pipeline_name), group in df.groupby(
        ["sample_id", "model_name", "model_type", "pipeline_name"]
    ):
        answers = group.sort_values("run_id")["generated_answer"].tolist()
        records.append({
            "sample_id": sample_id,
            "model_name": model_name,
            "model_type": model_type,
            "pipeline_name": pipeline_name,
            "question": group["question"].iloc[0],
            "answers": answers,
        })
    return records
