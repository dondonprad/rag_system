#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from src.embeddings.embedding_engine import EmbeddingEngine
from src.vector_store.chroma_db import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("test_retrieval")

def run_query(query_text: str, n_results: int = 3, classification: str = None):
    logger.info(f"Pertanyaan / Query: '{query_text}'")
    
    # 1. Hubungkan ke ChromaDB
    chroma_cfg = settings.get("chromadb", {})
    vector_store = ChromaVectorStore(
        host=chroma_cfg.get("host", "localhost"),
        port=chroma_cfg.get("port", 8000),
        collection_name=chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")
    )

    if not vector_store.is_healthy():
        logger.error("ChromaDB tidak aktif!")
        sys.exit(1)

    # 2. Inisialisasi engine embedding untuk query
    engine = EmbeddingEngine()
    query_vector = engine.embed_query(query_text)

    # 3. Filter RBAC jika ditentukan
    allowed_classes = [classification] if classification else None

    # 4. Pencarian kesamaan vektor
    results = vector_store.search_similarity(
        query_embedding=query_vector,
        n_results=n_results,
        allowed_classifications=allowed_classes
    )

    print("\n" + "=" * 60)
    print(f" HASIL RETRIEVAL DOKUMEN RELEVAN (Top {len(results)})")
    print("=" * 60)
    if not results:
        print("Tidak ada dokumen yang cocok dengan filter atau kueri.")
    else:
        for i, res in enumerate(results, start=1):
            meta = res["metadata"]
            sumber = meta.get('sumber', meta.get('file_name'))
            tipe = meta.get('tipe', meta.get('file_type'))
            halaman = meta.get('halaman', meta.get('page'))
            klasifikasi = meta.get('klasifikasi', meta.get('classification'))
            print(f"[{i}] Dokumen Sumber : {sumber} [{tipe}] (Halaman {halaman})")
            print(f"    Klasifikasi Akses : {klasifikasi}")
            print(f"    Cosine Distance   : {res.get('distance', 0):.4f}")
            print(f"    Kutipan Teks:\n{res.get('content')}")
            print("-" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Uji coba retrieval vektor dari ChromaDB")
    parser.add_argument("query", type=str, help="Teks pertanyaan / kueri pencarian")
    parser.add_argument("--top-k", type=int, default=3, help="Jumlah hasil teratas")
    parser.add_argument("--access", type=str, default=None, help="Filter klasifikasi RBAC (e.g. INTERNAL, SECRET)")
    args = parser.parse_args()

    run_query(args.query, n_results=args.top_k, classification=args.access)

