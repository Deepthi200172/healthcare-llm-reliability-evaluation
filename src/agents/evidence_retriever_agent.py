from typing import Dict, List

from src.models.base_model import BaseModel
from src.retrieval.retriever import Retriever
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EvidenceRetrieverAgent:
    """
    Agent 2: Retrieves relevant evidence from the vector store.

    Uses the retrieval_focus from QueryAnalyzerAgent to form a targeted query,
    optionally augmented with additional context when the verifier finds gaps.

    Output: joined string of retrieved evidence chunks.
    """

    def __init__(self, model: BaseModel, retriever: Retriever):
        self.model = model
        self.retriever = retriever

    def retrieve(
        self,
        question: str,
        analysis: Dict,
        additional_query: str = "",
    ) -> str:
        focus = analysis.get("retrieval_focus", question)
        query = focus if focus.strip() else question
        if additional_query:
            query = f"{query} {additional_query}"

        docs: List[str] = self.retriever.retrieve(query)
        logger.debug(f"EvidenceRetrieverAgent: retrieved {len(docs)} passages")
        return "\n\n---\n\n".join(docs)
