from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticCacheEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):        
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed(self, text: str) -> np.ndarray:
        # Normalize for cosine similarity
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding

    def batch_embed(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
