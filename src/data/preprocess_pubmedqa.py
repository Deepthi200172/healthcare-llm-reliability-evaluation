from typing import Optional

import pandas as pd

from src.data.load_pubmedqa import load_pubmedqa_raw, pubmedqa_to_records
from src.utils.file_utils import save_csv
from src.utils.logger import get_logger

logger = get_logger(__name__)

VALID_LABELS = {"yes", "no", "maybe"}


def preprocess_pubmedqa(
    sample_size: int = 100,
    seed: int = 42,
    split: str = "train",
    output_path: str = "data/processed/pubmedqa_sample.csv",
) -> pd.DataFrame:
    dataset = load_pubmedqa_raw(split=split)
    records = pubmedqa_to_records(dataset)

    df = pd.DataFrame(records)
    df["final_decision"] = df["final_decision"].str.lower().str.strip()
    df = df[df["final_decision"].isin(VALID_LABELS)]
    df = df.dropna(subset=["question", "long_answer", "final_decision"])
    df = df[df["question"].str.strip() != ""]
    df = df[df["long_answer"].str.strip() != ""]
    df = df.reset_index(drop=True)

    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    logger.info(f"Preprocessed {len(df)} samples")
    logger.info(f"Label distribution:\n{df['final_decision'].value_counts().to_string()}")

    save_csv(df.to_dict("records"), output_path)
    logger.info(f"Saved processed dataset to {output_path}")
    return df
