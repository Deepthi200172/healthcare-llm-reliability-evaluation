from typing import List, Dict
from src.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_ID = "qiaojin/PubMedQA"
DATASET_CONFIG = "pqa_labeled"


def load_pubmedqa_raw(
    split: str = "train",
    cache_dir: str = "data/raw/pubmedqa",
):
    from datasets import load_dataset

    logger.info(f"Loading {DATASET_ID} ({DATASET_CONFIG}), split={split}")
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=split,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    logger.info(f"Loaded {len(dataset)} raw examples")
    return dataset


def pubmedqa_to_records(dataset) -> List[Dict]:
    records = []
    for item in dataset:
        ctx = item.get("context", {})
        if isinstance(ctx, dict):
            ctx_str = " ".join(ctx.get("contexts", []))
        else:
            ctx_str = str(ctx)

        records.append({
            "id": str(item.get("pubid", "")),
            "question": item.get("question", ""),
            "context": ctx_str,
            "long_answer": item.get("long_answer", ""),
            "final_decision": item.get("final_decision", ""),
        })
    return records
