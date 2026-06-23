"""
Step 3 — Run Consistency Generation

Runs each question N times (default: 5) per model/pipeline to measure
how consistently the model answers the same question.

Output: outputs/responses/consistency_responses.csv
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

logger = get_logger("03_run_consistency_generation")

OUTPUT_PATH = "outputs/responses/consistency_responses.csv"


def main():
    logger.info("=== Step 3: Run Consistency Generation ===")

    with open("configs/evaluation.yaml") as f:
        eval_cfg = yaml.safe_load(f)

    n_runs = eval_cfg["runs_per_question_for_consistency"]
    df = pd.read_csv(eval_cfg["paths"]["processed_data"])
    samples = df.to_dict("records")
    logger.info(f"Loaded {len(samples)} samples — will run {n_runs} times each")

    pipeline_cfgs = load_pipelines_config()
    prompts = load_prompts_config()
    model_configs = get_all_model_configs(include_open=True, include_proprietary=True)

    rag_cfg = next(p for p in pipeline_cfgs if p["name"] == "rag")["retrieval"]
    docs = load_documents_from_pubmedqa(df)
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

            for run_id in range(1, n_runs + 1):
                logger.info(f"  [{pipe_name}] run {run_id}/{n_runs}")
                responses = pipeline.run_batch(samples, run_id=run_id)
                for r in responses:
                    append_csv(r.to_dict(), OUTPUT_PATH)

    logger.info(f"\nDone. Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
