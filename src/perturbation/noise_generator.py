import random
import re
from typing import List

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Common typo substitutions for function words (safe to swap in medical text)
_TYPO_MAP = {
    "the": "teh",
    "and": "adn",
    "that": "taht",
    "with": "iwth",
    "from": "form",
    "this": "thsi",
    "have": "hvae",
    "does": "deos",
    "were": "wree",
    "been": "bnee",
}


class NoiseGenerator:
    """
    Injects light typographic noise into biomedical questions.
    Targets only common function words to avoid corrupting medical terminology.
    Used to test Robustness under typo_noise perturbation.
    """

    def __init__(self, noise_rate: float = 0.15, seed: int = 42):
        self.noise_rate = noise_rate
        self.rng = random.Random(seed)

    def _apply_word_typos(self, text: str) -> str:
        words = text.split()
        result = []
        for word in words:
            lower = word.lower()
            if lower in _TYPO_MAP and self.rng.random() < self.noise_rate:
                # Preserve original capitalisation style
                typo = _TYPO_MAP[lower]
                if word[0].isupper():
                    typo = typo.capitalize()
                result.append(typo)
            else:
                result.append(word)
        return " ".join(result)

    def _drop_punctuation(self, text: str) -> str:
        """Randomly drop some punctuation marks."""
        return re.sub(
            r"([,;])",
            lambda m: "" if self.rng.random() < self.noise_rate else m.group(),
            text,
        )

    def generate(self, question: str) -> str:
        noisy = self._apply_word_typos(question)
        noisy = self._drop_punctuation(noisy)
        return noisy

    def generate_batch(self, questions: List[str]) -> List[str]:
        return [self.generate(q) for q in questions]
