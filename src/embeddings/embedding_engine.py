import os
import gc
import logging
from typing import List, Optional
import requests

# Atur mode offline/online berdasarkan konfigurasi .env (default offline kecuali diset 0 seperti di Google Colab)
os.environ.setdefault("HF_HUB_OFFLINE", os.getenv("HF_HUB_OFFLINE", "1"))
os.environ.setdefault("TRANSFORMERS_OFFLINE", os.getenv("TRANSFORMERS_OFFLINE", "1"))

# Hanya proteksi XPU jika user tidak secara eksplisit memilih XPU
if os.getenv("EMBEDDING_DEVICE", "cpu").lower() != "xpu":
    try:
        import torch
        if hasattr(torch, "xpu"):
            torch.xpu.is_available = lambda: False
    except Exception:
        pass

    try:
        import transformers.utils.import_utils
        transformers.utils.import_utils.is_torch_xpu_available = lambda check_device=False: False
    except Exception:
        pass

from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Mesin embedding on-premise untuk dokumen dan query (Zero Data Leakage).
    Mendukung Nomic Embed Text v1.5 dengan prefix task khusus, sentence-transformers, atau Ollama lokal.
    Dilengkapi manajemen memori XPU/iGPU (Intel Arc) otomatis untuk mencegah Out of Memory.
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
        self.device = str(device or emb_cfg.get("device", "cpu"))
        
        # Default batch_size aman: 16 untuk XPU (mencegah OOM alokasi buffer besar pada shared iGPU RAM)
        configured_bs = batch_size or emb_cfg.get("batch_size", 16)
        try:
            self.batch_size = int(configured_bs)
        except (ValueError, TypeError):
            self.batch_size = 16

        # Batasi batch_size pada XPU agar tidak memicu alokasi tensor contiguous raksasa (>2GB)
        if self.device.lower() == "xpu" and self.batch_size > 32:
            logger.info(
                f"Batch size {self.batch_size} terlalu besar untuk shared VRAM Intel Arc iGPU. "
                "Menyesuaikan batch_size ke 16 untuk mencegah XPU Out of Memory."
            )
            self.batch_size = 16

        self.doc_prefix = emb_cfg.get("document_prefix", "search_document: ")
        self.query_prefix = emb_cfg.get("query_prefix", "search_query: ")
        self.normalize = emb_cfg.get("normalize_embeddings", True)

        self._model = None

    def _empty_accelerator_cache(self):
        """Membebaskan cache VRAM pada XPU/CUDA dan memicu garbage collection."""
        gc.collect()
        try:
            import torch
            if hasattr(torch, "xpu"):
                torch.xpu.empty_cache()
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _get_sentence_transformer_model(self):
        """Lazy load SentenceTransformer model dengan dukungan CUDA (Google Colab), XPU, dan fallback CPU."""
        if self._model is None:
            offline_flag = os.getenv("HF_HUB_OFFLINE", "0" if str(self.device).lower() == "cuda" else "1")
            os.environ["HF_HUB_OFFLINE"] = offline_flag
            os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", offline_flag)
            local_only = offline_flag == "1"

            if str(self.device).lower() != "xpu":
                try:
                    import torch
                    if hasattr(torch, "xpu"):
                        torch.xpu.is_available = lambda: False
                except Exception:
                    pass

            # Validasi ketersediaan CUDA jika device disetel ke cuda
            if str(self.device).lower().startswith("cuda"):
                try:
                    import torch
                    if not torch.cuda.is_available():
                        logger.warning(
                            "Device 'cuda' dipilih tetapi GPU NVIDIA CUDA tidak terdeteksi pada runtime ini. "
                            "Pastikan Runtime Google Colab sudah disetel ke GPU (T4/L4/A100). Beralih ke 'cpu'..."
                        )
                        self.device = "cpu"
                except Exception:
                    self.device = "cpu"

            from transformers import PreTrainedModel
            if not hasattr(PreTrainedModel, "get_extended_attention_mask"):
                def _nomic_get_extended_attention_mask(model_self, attention_mask, input_shape, device_or_dtype=None):
                    if attention_mask is None:
                        return None
                    if attention_mask.dim() == 3:
                        ext_mask = attention_mask[:, None, :, :]
                    elif attention_mask.dim() == 2:
                        ext_mask = attention_mask[:, None, None, :]
                    else:
                        ext_mask = attention_mask
                    ext_mask = ext_mask.to(dtype=getattr(model_self, "dtype", None))
                    return (1.0 - ext_mask) * -10000.0
                PreTrainedModel.get_extended_attention_mask = _nomic_get_extended_attention_mask

            from sentence_transformers import SentenceTransformer
            logger.info(f"Memuat model embedding: {self.model_name} pada device: {self.device} (local_files_only={local_only})")
            try:
                self._model = SentenceTransformer(
                    self.model_name,
                    trust_remote_code=True,
                    device=self.device,
                    model_kwargs={"local_files_only": local_only}
                )
            except Exception as e:
                err_msg = str(e).lower()
                if "out of memory" in err_msg or str(self.device).lower() in ("xpu", "cuda"):
                    logger.warning(
                        f"Gagal memuat model embedding pada device '{self.device}' ({e}). "
                        "Mencoba mengunduh ulang tanpa batasan local_files_only atau fallback ke 'cpu'..."
                    )
                    self._empty_accelerator_cache()
                    try:
                        os.environ["HF_HUB_OFFLINE"] = "0"
                        os.environ["TRANSFORMERS_OFFLINE"] = "0"
                        self._model = SentenceTransformer(
                            self.model_name,
                            trust_remote_code=True,
                            device=self.device
                        )
                    except Exception:
                        self.device = "cpu"
                        self._model = SentenceTransformer(self.model_name, trust_remote_code=True, device="cpu")
                else:
                    os.environ["HF_HUB_OFFLINE"] = "0"
                    os.environ["TRANSFORMERS_OFFLINE"] = "0"
                    self._model = SentenceTransformer(self.model_name, trust_remote_code=True, device=self.device)
        return self._model

    def _embed_via_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Menghasilkan embedding dengan pemrosesan mini-batch aman dan mitigasi XPU OOM."""
        if not texts:
            return []

        model = self._get_sentence_transformer_model()
        effective_bs = self.batch_size
        all_embeddings: List[List[float]] = []

        total_texts = len(texts)
        i = 0

        while i < total_texts:
            chunk_slice = texts[i : i + effective_bs]
            sub_bs = len(chunk_slice)
            success = False

            while not success and sub_bs > 0:
                current_input = chunk_slice[:sub_bs]
                try:
                    embeddings = model.encode(
                        current_input,
                        batch_size=sub_bs,
                        show_progress_bar=False,
                        normalize_embeddings=self.normalize
                    )
                    all_embeddings.extend(embeddings.tolist())
                    i += sub_bs
                    success = True

                    # Bersihkan cache XPU setelah setiap mini-batch sukses
                    if str(self.device).lower() == "xpu":
                        self._empty_accelerator_cache()

                except Exception as e:
                    err_text = str(e).lower()
                    if "out of memory" in err_text or "xpu out of memory" in err_text or "cuda out of memory" in err_text:
                        logger.warning(
                            f"Terdeteksi Out of Memory ({e.__class__.__name__}) pada batch size {sub_bs}. "
                            "Mengosongkan cache memori (torch.xpu.empty_cache)..."
                        )
                        self._empty_accelerator_cache()

                        if sub_bs > 1:
                            sub_bs = max(1, sub_bs // 2)
                            logger.info(f"Mencoba ulang dengan ukuran batch lebih kecil: {sub_bs}")
                        else:
                            # Jika batch size 1 sekalipun OOM di XPU, alihkan ke CPU
                            logger.error(
                                "XPU VRAM tidak mencukupi untuk 1 chunk sekalipun. "
                                "Mengalihkan model ke CPU secara otomatis untuk menyelesaikan pemrosesan..."
                            )
                            try:
                                self._empty_accelerator_cache()
                                model = model.to("cpu")
                                self.device = "cpu"
                                embeddings = model.encode(
                                    current_input,
                                    batch_size=1,
                                    show_progress_bar=False,
                                    normalize_embeddings=self.normalize
                                )
                                all_embeddings.extend(embeddings.tolist())
                                i += 1
                                success = True
                            except Exception as cpu_err:
                                logger.error(f"Gagal memproses embedding bahkan pada CPU: {cpu_err}")
                                raise cpu_err
                    else:
                        raise e

        # Final memory cleanup
        self._empty_accelerator_cache()
        return all_embeddings

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

