"""
Step 4 — Create Perturbations

Generates three types of perturbed questions for each sample:
  1. paraphrase   — same meaning, different wording  (LLM-based)
  2. typo_noise   — light typographic errors          (rule-based)
  3. adversarial  — misleading framing                (LLM-based)

Output: data/perturbations/pubmedqa_perturbations.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yaml

from src.models.proprietary_models import build_model
from src.perturbation.paraphrase_generator import ParaphraseGenerator
from src.perturbation.noise_generator import NoiseGenerator
from src.perturbation.adversarial_generator import AdversarialGenerator
from src.utils.config_loader import get_judge_config, load_models_config
from src.utils.file_utils import save_csv
from src.utils.logger import get_logger

logger = get_logger("04_create_perturbations")

OUTPUT_PATH = "data/perturbations/pubmedqa_perturbations.csv"


def main():
    logger.info("=== Step 4: Create Perturbations ===")

    with open("configs/evaluation.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    df = pd.read_csv(eval_cfg["paths"]["processed_data"])
    logger.info(f"Loaded {len(df)} samples")

    # Use the judge model as the LLM for generating paraphrases and adversarials
    judge_cfg = get_judge_config()
    try:
        llm = build_model(judge_cfg)
    except Exception as e:
        logger.error(f"Could not load LLM for perturbation: {e}")
        raise

    paraphraser = ParaphraseGenerator(llm)
    noise_gen = NoiseGenerator(noise_rate=0.15, seed=42)
    adversarial_gen = AdversarialGenerator(llm)

    records = []
    for _, row in df.iterrows():
        original_q = row["question"]
        base = {
            "sample_id": row["id"],
            "original_question": original_q,
            "reference_answer": row["long_answer"],
            "reference_decision": row["final_decision"],
        }

        logger.info(f"  Perturbing sample {row['id']}")

        # Paraphrase
        records.append({
            **base,
            "perturbation_type": "paraphrase",
            "perturbed_question": paraphraser.generate(original_q),
        })
        # Typo noise (rule-based, no LLM needed)
        records.append({
            **base,
            "perturbation_type": "typo_noise",
            "perturbed_question": noise_gen.generate(original_q),
        })
        # Adversarial
        records.append({
            **base,
            "perturbation_type": "adversarial",
            "perturbed_question": adversarial_gen.generate(original_q),
        })

    save_csv(records, OUTPUT_PATH)
    logger.info(f"\nDone. {len(records)} perturbations saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
