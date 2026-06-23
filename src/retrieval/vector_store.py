from typing import Dict, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "healthcare_docs",
        persist_dir: str = "data/vector_store",
    ):
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _init(self):
        if self._client is not None:
            return
        import chromadb
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB ready: {self.persist_dir} / {self.collection_name}")

    def add_documents(self, documents: List[Dict], embeddings: List[List[float]]) -> None:
        self._init()
        ids = [doc.get("chunk_id", doc["id"]) for doc in documents]
        texts = [doc["content"] for doc in documents]
        metas = [{k: str(v) for k, v in doc.items() if k != "content"} for doc in documents]
        self._collection.upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metas)
        logger.info(f"Upserted {len(documents)} chunks into ChromaDB")

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        self._init()
        results = self._collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results["documents"][0] if results["documents"] else []

    def count(self) -> int:
        self._init()
        return self._collection.count()

    def reset(self) -> None:
        self._init()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Vector store reset")
