import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from src.processing.chunking import DocumentChunk
from config.settings import settings

logger = logging.getLogger(__name__)

class ChromaVectorStore:
    """Modul antarmuka ke ChromaDB yang berjalan di Docker (port 8000).
    Mendukung penyimpanan vektor, metadata filtering (RBAC), dan similarity search.
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        collection_name: Optional[str] = None,
        persist_directory: Optional[str] = None
    ):
        import os
        chroma_cfg = settings.get("chromadb", {})
        self.host = host or chroma_cfg.get("host", "localhost")
        self.port = port or chroma_cfg.get("port", 8000)
        self.collection_name = collection_name or chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")
        self.distance_function = chroma_cfg.get("distance_function", "cosine")
        self.persist_directory = persist_directory or os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_colab")
        self.mode = os.getenv("CHROMA_MODE", "auto").lower()

        self.client = None
        self.collection = None

        if self.mode == "persistent":
            logger.info(f"Menggunakan ChromaDB PersistentClient pada direktori: {self.persist_directory}")
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )
            self.collection = self.get_or_create_collection(self.collection_name)
        else:
            try:
                http_client = chromadb.HttpClient(
                    host=self.host,
                    port=self.port,
                    settings=Settings(anonymized_telemetry=False)
                )
                http_client.heartbeat()
                self.client = http_client
                self.mode = "http"
                self.collection = self.get_or_create_collection(self.collection_name)
            except Exception as e:
                if self.mode == "auto" and os.getenv("RUNTIME_ENV", "").lower() == "colab_gpu":
                    logger.warning(
                        f"Server HTTP ChromaDB ({self.host}:{self.port}) belum aktif ({e}). "
                        f"Beralih otomatis ke ChromaDB PersistentClient di '{self.persist_directory}'..."
                    )
                    self.client = chromadb.PersistentClient(
                        path=self.persist_directory,
                        settings=Settings(anonymized_telemetry=False, allow_reset=True)
                    )
                    self.mode = "persistent"
                    self.collection = self.get_or_create_collection(self.collection_name)
                else:
                    logger.error(f"Gagal menginisialisasi koneksi ke ChromaDB pada {self.host}:{self.port} - {e}")

    def is_healthy(self) -> bool:
        """Memeriksa kesehatan koneksi ke ChromaDB (HTTP Docker maupun PersistentClient Colab)."""
        if self.client is None:
            return False
        try:
            res = self.client.heartbeat()
            return res > 0
        except Exception as e:
            logger.error(f"Gagal terhubung ke ChromaDB pada {self.host}:{self.port} - {e}")
            return False

    def get_or_create_collection(self, collection_name: str):
        """Membuat atau mengambil koleksi vektor dengan metric jarak yang ditentukan."""
        return self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": self.distance_function}
        )

    def upsert_chunks(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 100
    ) -> int:
        """Menyimpan potongan teks dan embedding ke ChromaDB secara batch."""
        if not chunks:
            return 0

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Jumlah chunks ({len(chunks)}) tidak sama dengan jumlah embeddings ({len(embeddings)})"
            )

        total_upserted = 0
        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = [c.chunk_id for c in batch_chunks]
            documents = [c.text for c in batch_chunks]
            # ChromaDB mewajibkan nilai metadata berupa primitive (str, int, float, bool)
            metadatas = []
            for c in batch_chunks:
                clean_meta = {}
                for k, v in c.metadata.items():
                    if isinstance(v, (str, int, float, bool)):
                        clean_meta[k] = v
                    else:
                        clean_meta[k] = str(v)
                metadatas.append(clean_meta)

            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas
            )
            total_upserted += len(batch_chunks)
            logger.info(f"Berhasil menyimpan {total_upserted}/{total_chunks} chunks ke ChromaDB.")

        return total_upserted

    def delete_by_source(self, source_name: str) -> int:
        """Menghapus seluruh chunk dokumen berdasarkan nama file sumber (berguna untuk update dokumen termodifikasi)."""
        count_before = self.collection.count()
        try:
            self.collection.delete(where={"sumber": source_name})
        except Exception as e:
            logger.debug(f"Hapus sumber '{source_name}': {e}")

        try:
            self.collection.delete(where={"file_name": source_name})
        except Exception as e:
            logger.debug(f"Hapus file_name '{source_name}': {e}")

        count_after = self.collection.count()
        deleted = max(0, count_before - count_after)
        if deleted > 0:
            logger.info(f"Berhasil menghapus {deleted} chunk lama untuk '{source_name}'.")
        return deleted

    def search_similarity(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        allowed_classifications: Optional[List[str]] = None,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Pencarian vektor dengan Document-level RBAC metadata filtering."""
        final_where = {}
        if where_filter:
            final_where.update(where_filter)

        # Terapkan filter RBAC keamanan jika ditentukan
        if allowed_classifications:
            if len(allowed_classifications) == 1:
                final_where["classification"] = allowed_classifications[0]
            else:
                final_where["classification"] = {"$in": allowed_classifications}

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=final_where if final_where else None
        )

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                doc_id = results["ids"][0][idx]
                doc_text = results["documents"][0][idx] if results.get("documents") else ""
                metadata = results["metadatas"][0][idx] if results.get("metadatas") else {}
                distance = results["distances"][0][idx] if results.get("distances") else None

                formatted_results.append({
                    "id": doc_id,
                    "content": doc_text,
                    "metadata": metadata,
                    "distance": distance
                })

        return formatted_results

    def get_stats(self) -> Dict[str, Any]:
        """Statistik data dalam koleksi saat ini."""
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_vectors": count,
            "host": self.host,
            "port": self.port,
            "distance_metric": self.distance_function
        }

