from typing import List, Optional
import logging
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
import gc
import os

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self) -> None:
        self.model: Optional[SentenceTransformer] = None
        self.device = self._get_device()
        self._load_model()

    def _get_device(self) -> str:
        """Determine the best device for the current environment"""
        # Force CPU for cloud deployments to avoid CUDA memory issues
        if os.getenv('DEPLOYMENT_ENV') in ['vercel', 'render', 'heroku']:
            return 'cpu'
        # Use CPU if CUDA is not available or if explicitly set
        if not torch.cuda.is_available() or os.getenv('FORCE_CPU', 'false').lower() == 'true':
            return 'cpu'
        return 'cuda'

    def _load_model(self) -> None:
        try:
            # Set memory-efficient options
            self.model = SentenceTransformer(
                settings.embedding_model_name,
                device=self.device,
                # Use half precision for memory efficiency
                model_kwargs={'torch_dtype': torch.float16 if self.device == 'cuda' else torch.float32}
            )
            
            # Move to CPU if needed and set to eval mode
            if self.device == 'cpu':
                self.model = self.model.cpu()
            self.model.eval()
            
            # Disable gradients to save memory
            for param in self.model.parameters():
                param.requires_grad = False
            
            logger.info(f"Loaded embedding model: {settings.embedding_model_name} on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            if not texts:
                return []

            # Reduce batch size for memory efficiency
            batch_size = min(8, len(texts))  # Much smaller batch size
            
            # Process in smaller chunks to avoid OOM
            max_chunk_size = 50  # Process max 50 texts at once
            all_embeddings = []
            
            for i in range(0, len(texts), max_chunk_size):
                chunk = texts[i:i + max_chunk_size]
                
                # Clear cache before processing
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                embeddings = self.model.encode(
                    chunk,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_tensor=False,
                    normalize_embeddings=True,  # Normalize to save space
                    device=self.device
                )

                if isinstance(embeddings, torch.Tensor):
                    embeddings = embeddings.cpu().numpy()

                if isinstance(embeddings, np.ndarray):
                    # Convert to float32 to save memory (from float64)
                    embeddings = embeddings.astype(np.float32)
                    embeddings = embeddings.tolist()
                
                all_embeddings.extend(embeddings)
                
                # Clear memory after each chunk
                del embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            return all_embeddings

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Clean up on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            raise

    async def embed_single_text(self, text: str) -> List[float]:
        embeddings = await self.embed_texts([text])
        return embeddings[0] if embeddings else []

    def get_embedding_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
    
    def cleanup_memory(self) -> None:
        """Explicit memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
    def __del__(self):
        """Cleanup on object destruction"""
        self.cleanup_memory()

    async def similarity_search(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[List[float]],
        top_k: int = 5,
    ) -> List[tuple]:
        try:
            # Use numpy for better memory efficiency
            query_array = np.array(query_embedding, dtype=np.float32)
            candidate_array = np.array(candidate_embeddings, dtype=np.float32)
            
            # Compute cosine similarity using numpy (more memory efficient)
            # Normalize vectors
            query_norm = query_array / np.linalg.norm(query_array)
            candidate_norms = candidate_array / np.linalg.norm(candidate_array, axis=1, keepdims=True)
            
            # Compute similarities
            similarities = np.dot(candidate_norms, query_norm)
            
            # Get top k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                results.append((int(idx), float(similarities[idx])))
            
            # Clean up
            del query_array, candidate_array, similarities
            gc.collect()

            return results

        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            gc.collect()
            raise
