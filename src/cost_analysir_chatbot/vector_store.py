from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np


@dataclass
class VectorDocument:
    text: str
    doc_id: str
    metadata: dict


class VectorStore:
    """Simple in-memory cosine similarity search."""

    def __init__(self) -> None:
        self.embeddings: List[np.ndarray] = []
        self.records: List[VectorDocument] = []

    def add(self, embedding: Sequence[float], *, text: str, doc_id: str, metadata: dict) -> None:
        vector = self._normalize(np.array(embedding, dtype=np.float32))
        self.embeddings.append(vector)
        self.records.append(VectorDocument(text=text, doc_id=doc_id, metadata=metadata))

    def search(self, embedding: Sequence[float], top_k: int = 5) -> List[tuple[VectorDocument, float]]:
        if not self.embeddings:
            return []
        query = self._normalize(np.array(embedding, dtype=np.float32))
        matrix = np.vstack(self.embeddings)
        scores = matrix @ query
        idx = np.argsort(scores)[::-1][:top_k]
        return [(self.records[i], float(scores[i])) for i in idx]

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

