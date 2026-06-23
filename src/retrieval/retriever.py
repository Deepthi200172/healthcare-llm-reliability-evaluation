from typing import Dict, List

from src.retrieval.embeddings import EmbeddingModel
from src.retrieval.vector_store import ChromaVectorStore
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Retriever:
    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.top_k = top_k

    def index_documents(self, documents: List[Dict]) -> None:
        logger.info(f"Embedding {len(documents)} chunks...")
        texts = [doc["content"] for doc in documents]
        embeddings = self.embedding_model.embed(texts)
        self.vector_store.add_documents(documents, embeddings)
        logger.info("Indexing complete")

    def retrieve(self, query: str, top_k: int = None) -> List[str]:
        k = top_k or self.top_k
        qvec = self.embedding_model.embed_query(query)
        docs = self.vector_store.search(qvec, top_k=k)
        logger.debug(f"Retrieved {len(docs)} docs for: {query[:60]!r}")
        return docs


def build_retriever(
    documents: List[Dict],
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    persist_dir: str = "data/vector_store",
    top_k: int = 5,
    force_reindex: bool = False,
) -> Retriever:
    from src.retrieval.chunking import chunk_documents

    embedding_model = EmbeddingModel(embedding_model_name)
    vector_store = ChromaVectorStore(persist_dir=persist_dir)
    retriever = Retriever(embedding_model, vector_store, top_k=top_k)

    if force_reindex or vector_store.count() == 0:
        retriever.index_documents(documents)

    return retriever
