from typing import List, Optional
import logging
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            self.model = SentenceTransformer(settings.embedding_model_name)
            logger.info(f"Loaded embedding model: {settings.embedding_model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                return []

            embeddings = self.model.encode(
                texts, batch_size=32, show_progress_bar=False, convert_to_tensor=False
            )

            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()

            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()

            return embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise

    async def embed_single_text(self, text: str) -> List[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()

    async def similarity_search(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[tuple]:
        try:
            query_tensor = torch.tensor(query_embedding).unsqueeze(0)
            candidate_tensor = torch.tensor(candidate_embeddings)

            similarities = torch.cosine_similarity(query_tensor, candidate_tensor)

            top_indices = similarities.argsort(descending=True)[:top_k]

            results = []
            for idx in top_indices:
                results.append((int(idx), float(similarities[idx])))

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
