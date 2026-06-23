from src.models.base_model import BaseModel, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class HuggingFaceModel(BaseModel):
    """Wrapper for instruction-tuned HuggingFace causal LMs (local inference)."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._pipeline = None
        # model_id must be set in config; fall back to name if missing
        self._model_id = config.model_id or config.name

    def _load(self):
        if self._pipeline is not None:
            return
        import torch
        from transformers import AutoTokenizer, pipeline

        logger.info(f"Loading HuggingFace model: {self._model_id}")
        tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._pipeline = pipeline(
            "text-generation",
            model=self._model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.float16,
            device_map=self.config.device_map,
            max_new_tokens=self.config.max_tokens,
            temperature=self.config.temperature if self.config.temperature > 0 else None,
            do_sample=self.config.temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
        logger.info(f"Model loaded: {self.config.name}")

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        self._load()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        outputs = self._pipeline(messages)
        generated = outputs[0]["generated_text"]
        # Chat pipelines return a list of turns; extract the assistant turn
        if isinstance(generated, list):
            return generated[-1]["content"].strip()
        return str(generated).strip()
