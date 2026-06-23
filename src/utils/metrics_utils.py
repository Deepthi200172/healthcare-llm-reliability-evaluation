from typing import Dict, List

import numpy as np


def compute_stats(scores: List[float]) -> Dict[str, float]:
    arr = np.array([s for s in scores if s >= 0], dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "n": 0}
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(arr),
    }


def normalize_score(score: float, min_val: float = 0.0, max_val: float = 10.0) -> float:
    if max_val == min_val:
        return 0.0
    return (score - min_val) / (max_val - min_val)


def extract_final_decision(response: str) -> str:
    text = response.lower().strip()
    for label in ["yes", "no", "maybe"]:
        if text.endswith(label) or f"decision: {label}" in text or f"answer: {label}" in text:
            return label
    for label in ["yes", "no", "maybe"]:
        if label in text.split():
            return label
    return "unknown"
