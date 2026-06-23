import os
from typing import Dict

from src.models.base_model import BaseModel, ModelConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Map config name → actual API model identifier
_OPENAI_IDS: Dict[str, str] = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

_ANTHROPIC_IDS: Dict[str, str] = {
    "claude-3-5-sonnet": "claude-sonnet-4-6",
    "claude-3-haiku": "claude-haiku-4-5-20251001",
}

_GOOGLE_IDS: Dict[str, str] = {
    "gemini-1.5-flash": "gemini-1.5-flash",
    "gemini-1.5-pro": "gemini-1.5-pro",
}


class OpenAIModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import openai
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY is not set")
        self._client = openai.OpenAI(api_key=api_key)
        self._model_id = _OPENAI_IDS.get(config.name, config.name)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat.completions.create(
            model=self._model_id,
            messages=messages,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()


class AnthropicModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY is not set")
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model_id = _ANTHROPIC_IDS.get(config.name, config.name)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        kwargs = dict(
            model=self._model_id,
            max_tokens=self.config.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if system_prompt:
            kwargs["system"] = system_prompt

        response = self._client.messages.create(**kwargs)
        return response.content[0].text.strip()


class GoogleModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        import google.generativeai as genai
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY is not set")
        genai.configure(api_key=api_key)
        model_id = _GOOGLE_IDS.get(config.name, config.name)
        self._model = genai.GenerativeModel(model_id)

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        response = self._model.generate_content(full_prompt)
        return response.text.strip()


class MockModel(BaseModel):
    """Placeholder model for testing without API keys."""

    def generate(self, prompt: str, system_prompt: str = "") -> str:
        return (
            f"[MOCK] This is a placeholder response for model '{self.config.name}'. "
            "Replace with a real model implementation. Final decision: Maybe."
        )


def build_model(config: ModelConfig) -> BaseModel:
    """Instantiate the correct model class based on provider."""
    provider = config.provider.lower()
    if provider == "openai":
        return OpenAIModel(config)
    elif provider == "anthropic":
        return AnthropicModel(config)
    elif provider == "google":
        return GoogleModel(config)
    elif provider in ("huggingface", "huggingface_or_ollama"):
        from src.models.open_models import HuggingFaceModel
        return HuggingFaceModel(config)
    elif provider == "mock":
        return MockModel(config)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")
