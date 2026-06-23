from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.models.base_model import ModelConfig

_ROOT = Path(__file__).parent.parent.parent


def _load_yaml(rel_path: str) -> Dict:
    path = _ROOT / rel_path
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_models_config() -> Dict:
    return _load_yaml("configs/models.yaml")


def load_pipelines_config() -> List[Dict]:
    return _load_yaml("configs/pipelines.yaml")["pipelines"]


def load_evaluation_config() -> Dict:
    return _load_yaml("configs/evaluation.yaml")


def load_prompts_config() -> Dict:
    return _load_yaml("configs/prompts.yaml")


def get_all_model_configs(include_open: bool = True, include_proprietary: bool = True) -> List[ModelConfig]:
    """Return a flat list of ModelConfig objects from models.yaml."""
    raw = load_models_config()
    configs = []

    if include_proprietary:
        for m in raw.get("proprietary_models", []):
            model_id = raw.get("provider_model_ids", {}).get(m["name"], m["name"])
            configs.append(ModelConfig(
                name=m["name"],
                model_type=m["model_type"],
                provider=m["provider"],
                model_id=model_id,
                temperature=m.get("temperature", 0.0),
                max_tokens=m.get("max_tokens", 512),
            ))

    if include_open:
        for m in raw.get("open_source_models", []):
            configs.append(ModelConfig(
                name=m["name"],
                model_type=m["model_type"],
                provider=m["provider"],
                model_id=m.get("model_id", m["name"]),
                temperature=m.get("temperature", 0.0),
                max_tokens=m.get("max_tokens", 512),
            ))

    return configs


def get_judge_config() -> ModelConfig:
    raw = load_models_config()
    j = raw["judge_model"]
    return ModelConfig(
        name=j["name"],
        model_type="proprietary",
        provider=j["provider"],
        model_id=raw.get("provider_model_ids", {}).get(j["name"], j["name"]),
        temperature=j.get("temperature", 0.0),
        max_tokens=j.get("max_tokens", 1024),
    )
