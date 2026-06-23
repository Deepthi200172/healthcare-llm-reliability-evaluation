from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_documents_from_pubmedqa(df: pd.DataFrame) -> List[Dict]:
    docs = []
    for _, row in df.iterrows():
        ctx = str(row.get("context", "")).strip()
        if ctx:
            docs.append({
                "id": str(row["id"]),
                "content": ctx,
                "source": "pubmedqa_context",
                "question": str(row["question"]),
            })
    logger.info(f"Loaded {len(docs)} documents from PubMedQA")
    return docs


def load_documents_from_directory(
    dir_path: str,
    extensions: List[str] = [".txt", ".md"],
) -> List[Dict]:
    docs = []
    for path in Path(dir_path).rglob("*"):
        if path.suffix in extensions:
            content = path.read_text(encoding="utf-8", errors="ignore").strip()
            if content:
                docs.append({"id": path.stem, "content": content, "source": str(path)})
    logger.info(f"Loaded {len(docs)} documents from {dir_path}")
    return docs
