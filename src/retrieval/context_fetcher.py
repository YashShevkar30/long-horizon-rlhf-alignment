"""Dynamic context retrieval for training augmentation."""
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class VectorContextFetcher:
    """Retrieves relevant context from vector DB for augmenting training."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._documents = []
        self._embeddings = []

    def index_documents(self, documents: list[dict], embeddings: np.ndarray):
        self._documents.extend(documents)
        self._embeddings.extend(embeddings.tolist())
        logger.info("Indexed %d documents (total: %d)", len(documents), len(self._documents))

    def fetch(self, query_embedding: np.ndarray, top_k: int = 3) -> list[dict]:
        if not self._embeddings:
            return []
        db_vectors = np.array(self._embeddings)
        query = query_embedding.reshape(1, -1)
        similarities = np.dot(db_vectors, query.T).flatten()
        similarities /= (np.linalg.norm(db_vectors, axis=1) * np.linalg.norm(query) + 1e-8)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {"document": self._documents[i], "similarity": round(float(similarities[i]), 4)}
            for i in top_indices if i < len(self._documents)
        ]

    def fetch_by_task_type(self, task_type: str, top_k: int = 5) -> list[dict]:
        return [
            doc for doc in self._documents
            if doc.get("task_type") == task_type
        ][:top_k]

    @property
    def size(self) -> int:
        return len(self._documents)
