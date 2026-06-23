"""
Step 5 — Run Robustness Generation

Runs all models and pipelines on the perturbed questions created in step 4.
The generated answers will be compared against the original factuality scores
to compute robustness drop.

Output: outputs/responses/robustness_responses.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import yaml

from src.models.proprietary_models import build_model
from src.pipelines.baseline_pipeline import BaselinePipeline
from src.pipelines.rag_pipeline import RAGPipeline
from src.pipelines.agentic_rag_pipeline import AgenticRAGPipeline
from src.retrieval.document_loader import load_documents_from_pubmedqa
from src.retrieval.chunking import chunk_documents
from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.retrieval.retriever import Retriever
from src.utils.config_loader import get_all_model_configs, load_pipelines_config, load_prompts_config
from src.utils.file_utils import append_csv
from src.utils.logger import get_logger

logger = get_logger("05_run_robustness_generation")

OUTPUT_PATH = "outputs/responses/robustness_responses.csv"


def _sample_row_to_dict(pert_row: dict) -> dict:
    """Convert a perturbation row into a sample dict for pipelines."""
    return {
        "id": pert_row["sample_id"],
        "question": pert_row["perturbed_question"],
        "long_answer": pert_row["reference_answer"],
        "final_decision": pert_row["reference_decision"],
    }


def main():
    logger.info("=== Step 5: Run Robustness Generation ===")

    with open("configs/evaluation.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    pert_df = pd.read_csv(eval_cfg["paths"]["perturbations"])
    orig_df = pd.read_csv(eval_cfg["paths"]["processed_data"])
    logger.info(f"Loaded {len(pert_df)} perturbed questions")

    pipeline_cfgs = load_pipelines_config()
    prompts = load_prompts_config()
    model_configs = get_all_model_configs(include_open=True, include_proprietary=True)

    rag_cfg = next(p for p in pipeline_cfgs if p["name"] == "rag")["retrieval"]
    docs = load_documents_from_pubmedqa(orig_df)
    chunks = chunk_documents(docs, rag_cfg["chunk_size"], rag_cfg["chunk_overlap"])
    emb_model = EmbeddingModel(rag_cfg["embedding_model"])
    store = ChromaVectorStore(persist_dir=rag_cfg["persist_dir"])
    retriever = Retriever(emb_model, store, top_k=rag_cfg["top_k"])
    if store.count() == 0:
        retriever.index_documents(chunks)

    system_prompts = prompts["system_prompts"]
    user_prompts = prompts["user_prompts"]

    for model_cfg in model_configs:
        logger.info(f"\n--- Model: {model_cfg.name} ---")
        try:
            model = build_model(model_cfg)
        except Exception as e:
            logger.error(f"Failed to load {model_cfg.name}: {e}")
            continue

        for pipe_cfg in pipeline_cfgs:
            pipe_name = pipe_cfg["name"]

            if pipe_name == "baseline":
                pipeline = BaselinePipeline(
                    model, system_prompts["baseline"], user_prompts["baseline"]
                )
            elif pipe_name == "rag":
                pipeline = RAGPipeline(
                    model, retriever, system_prompts["rag"], user_prompts["rag"]
                )
            elif pipe_name == "agentic_rag":
                pipeline = AgenticRAGPipeline(
                    model, retriever, system_prompts["agentic_rag"]
                )
            else:
                continue

            logger.info(f"  [{pipe_name}] running on {len(pert_df)} perturbed questions")
            for _, pert_row in pert_df.iterrows():
                sample = _sample_row_to_dict(pert_row.to_dict())
                response = pipeline.run(sample, run_id=1)
                record = response.to_dict()
                # Attach perturbation metadata for later joins
                record["perturbation_type"] = pert_row["perturbation_type"]
                record["original_question"] = pert_row["original_question"]
                append_csv(record, OUTPUT_PATH)

    logger.info(f"\nDone. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
