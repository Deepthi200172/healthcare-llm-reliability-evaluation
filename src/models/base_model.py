from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class ModelConfig:
    name: str
    model_type: str          # "proprietary" or "open_source"
    provider: str            # "openai", "anthropic", "google", "huggingface"
    model_id: str = ""       # actual model identifier (filled from config or name)
    temperature: float = 0.0
    max_tokens: int = 512
    device_map: str = "auto"


@dataclass
class ModelResponse:
    """Standardized output record for every model/pipeline run."""
    sample_id: str
    question: str
    reference_answer: str
    reference_decision: str
    model_name: str
    model_type: str
    pipeline_name: str
    run_id: int
    generated_answer: str
    retrieved_context: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "reference_decision": self.reference_decision,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "pipeline_name": self.pipeline_name,
            "run_id": self.run_id,
            "generated_answer": self.generated_answer,
            "retrieved_context": self.retrieved_context,
            "timestamp": self.timestamp,
            "error": self.error or "",
        }


class BaseModel(ABC):
    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def generate(self, prompt: str, system_prompt: str = "") -> str:
        """Generate a response for the given prompt."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.config.name!r})"
