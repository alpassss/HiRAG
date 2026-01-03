import numpy as np
from typing import List
from ._utils import wrap_embedding_func_with_attrs
import logging

logger = logging.getLogger("HiRAG")

EMBEDDING_MODEL_AVAILABLE = False
try:
    from sentence_transformers import SentenceTransformer
    import torch
    # Test if torch and sentence_transformers can actually be used
    if torch.cuda.is_available():
        logger.info("CUDA is available for torch")
    EMBEDDING_MODEL_AVAILABLE = True
    logger.info("Successfully imported sentence_transformers and torch")
except Exception as e:
    logger.warning(f"sentence_transformers or torch not available ({e}), using mock embedding")
    EMBEDDING_MODEL_AVAILABLE = False

class LocalEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if EMBEDDING_MODEL_AVAILABLE:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Failed to load embedding model {model_name}: {e}")
                self.model = None
    
    @wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
    async def embed(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for the given texts using local model or mock values
        """
        if self.model is not None:
            # Convert texts to embeddings using local model
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return np.array(embeddings, dtype=np.float32)
        else:
            # Fallback: generate mock embeddings
            logger.warning("Using mock embeddings as local model is not available")
            # Generate consistent mock embeddings based on text content
            embeddings = []
            for text in texts:
                # Create a deterministic hash-based embedding
                text_hash = hash(text) % (2**32)
                np.random.seed(text_hash)
                mock_embedding = np.random.random(384).astype(np.float32)
                # Normalize the embedding
                mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
                embeddings.append(mock_embedding)
            return np.array(embeddings)

# Default instance
local_embedding_func = None

def get_local_embedding():
    global local_embedding_func
    if local_embedding_func is None:
        local_embedding_func = LocalEmbedding()
    return local_embedding_func

@wrap_embedding_func_with_attrs(embedding_dim=384, max_token_size=512)
async def local_embedding(texts: List[str]) -> np.ndarray:
    """
    Local embedding function that can be used instead of OpenAI's API
    """
    return await get_local_embedding().embed(texts)