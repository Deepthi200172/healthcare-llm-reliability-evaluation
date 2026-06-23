import time
from datetime import datetime
from typing import Dict, List

from src.agents.answer_generator_agent import AnswerGeneratorAgent
from src.agents.evidence_retriever_agent import EvidenceRetrieverAgent
from src.agents.query_analyzer_agent import QueryAnalyzerAgent
from src.agents.verifier_agent import VerifierAgent
from src.models.base_model import BaseModel, ModelResponse
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class AgenticRAGPipeline:
    """
    Pipeline 3 — Agentic RAG.

    Step 1: QueryAnalyzerAgent  — understands question + decides retrieval focus
    Step 2: EvidenceRetrieverAgent — fetches evidence using analysis output
    Step 3: AnswerGeneratorAgent — generates an answer grounded in evidence
    Step 4: VerifierAgent — checks whether answer is supported; may flag uncertainty

    Designed as plain Python classes so it can be migrated to LangGraph later.
    """

    def __init__(
        self,
        model: BaseModel,
        retriever: Retriever,
        system_prompt: str,
        max_verification_iterations: int = 2,
    ):
        self.model = model
        self.retriever = retriever
        self.system_prompt = system_prompt
        self.max_verification_iterations = max_verification_iterations

        self.query_analyzer = QueryAnalyzerAgent(model)
        self.evidence_retriever = EvidenceRetrieverAgent(model, retriever)
        self.answer_generator = AnswerGeneratorAgent(model)
        self.verifier = VerifierAgent(model)

    def run(self, sample: Dict, run_id: int = 1) -> ModelResponse:
        question = sample["question"]
        start = time.time()
        error = None
        final_answer = ""
        context = ""

        try:
            # Step 1: Analyse the question
            analysis = self.query_analyzer.analyze(question)
            logger.debug(f"Analysis: {analysis}")

            # Step 2: Retrieve evidence using the analysis focus
            context = self.evidence_retriever.retrieve(question, analysis)

            # Step 3: Generate answer
            gen_result = self.answer_generator.generate(question, context)
            final_answer = gen_result.get("answer", "")
            logger.debug(f"Generated decision: {gen_result.get('short_decision')}")

            # Step 4: Verify — optionally re-try if unsupported
            for iteration in range(self.max_verification_iterations):
                verification = self.verifier.verify(question, final_answer, context)
                support_status = verification.get("support_status", "fully_supported")
                logger.debug(
                    f"Verification iteration {iteration + 1}: "
                    f"status={support_status}"
                )
                if support_status == "fully_supported":
                    final_answer = verification.get("verified_answer", final_answer)
                    break
                # If unsupported, regenerate with additional retrieval using the gap
                gap = verification.get("verification_reason", "")
                if gap:
                    context = self.evidence_retriever.retrieve(
                        question, analysis, additional_query=gap
                    )
                gen_result = self.answer_generator.generate(question, context)
                final_answer = gen_result.get("answer", final_answer)

        except Exception as e:
            error = str(e)
            logger.error(
                f"AgenticRAG error [{self.model.config.name}] "
                f"sample={sample['id']}: {e}"
            )

        return ModelResponse(
            sample_id=str(sample["id"]),
            question=question,
            reference_answer=sample.get("long_answer", ""),
            reference_decision=sample.get("final_decision", ""),
            model_name=self.model.config.name,
            model_type=self.model.config.model_type,
            pipeline_name="agentic_rag",
            run_id=run_id,
            generated_answer=final_answer,
            retrieved_context=context,
            timestamp=datetime.utcnow().isoformat(),
            error=error,
        )

    def run_batch(self, samples: List[Dict], run_id: int = 1) -> List[ModelResponse]:
        results = []
        for i, sample in enumerate(samples, 1):
            logger.info(
                f"[agentic_rag][{self.model.config.name}] run={run_id} "
                f"sample {i}/{len(samples)} (id={sample['id']})"
            )
            results.append(self.run(sample, run_id=run_id))
        return results
