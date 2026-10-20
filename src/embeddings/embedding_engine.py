import logging
from typing import List, Optional
import requests
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Mesin embedding on-premise untuk dokumen dan query (Zero Data Leakage).
    Mendukung Nomic Embed Text v1.5 dengan prefix task khusus, sentence-transformers, atau Ollama lokal.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: Optional[int] = None
    ):
        emb_cfg = settings.get("embedding", {})
        self.model_name = model_name or emb_cfg.get("model_name", "nomic-ai/nomic-embed-text-v1.5")
        self.provider = provider or emb_cfg.get("provider", "sentence-transformers")
        self.device = device or emb_cfg.get("device", "cpu")
        self.batch_size = batch_size or emb_cfg.get("batch_size", 32)
        
        self.doc_prefix = emb_cfg.get("document_prefix", "search_document: ")
        self.query_prefix = emb_cfg.get("query_prefix", "search_query: ")
        self.normalize = emb_cfg.get("normalize_embeddings", True)

        self._model = None

    def _get_sentence_transformer_model(self):
        """Lazy load SentenceTransformer model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Memuat model embedding lokal: {self.model_name} pada device: {self.device}")
            # trust_remote_code=True diperlukan untuk nomic-embed-text
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        return self._model

    def _embed_via_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        model = self._get_sentence_transformer_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize
        )
        return embeddings.tolist()

    def _embed_via_ollama(self, texts: List[str], is_query: bool = False) -> List[List[float]]:
        """Fallback atau integrasi embedding melalui API Ollama lokal."""
        ollama_url = "http://localhost:11434/api/embeddings"
        results: List[List[float]] = []
        for text in texts:
            response = requests.post(
                ollama_url,
                json={"model": "nomic-embed-text", "prompt": text},
                timeout=30
            )
            response.raise_for_status()
            results.append(response.json()["embedding"])
        return results

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Menghasilkan representasi vektor untuk kumpulan dokumen teks."""
        if not documents:
            return []

        # Tambahkan prefix nomic search_document jika model menggunakan nomic
        prefixed_texts = [
            f"{self.doc_prefix}{doc}" if "nomic" in self.model_name.lower() and not doc.startswith(self.doc_prefix) else doc
            for doc in documents
        ]

        if self.provider == "ollama":
            return self._embed_via_ollama(prefixed_texts, is_query=False)
        else:
            return self._embed_via_sentence_transformers(prefixed_texts)

    def embed_query(self, query: str) -> List[float]:
        """Menghasilkan representasi vektor untuk satu kueri pertanyaan."""
        prefixed_query = (
            f"{self.query_prefix}{query}"
            if "nomic" in self.model_name.lower() and not query.startswith(self.query_prefix)
            else query
        )

        if self.provider == "ollama":
            return self._embed_via_ollama([prefixed_query], is_query=True)[0]
        else:
            return self._embed_via_sentence_transformers([prefixed_query])[0]

