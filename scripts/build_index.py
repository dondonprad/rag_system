#!/usr/bin/env python3
import sys
import os
import time
import argparse
import logging
from pathlib import Path

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from src.ingestion.loaders import DocumentLoader
from src.processing.chunking import TextChunker
from src.embeddings.embedding_engine import EmbeddingEngine
from src.vector_store.chroma_db import ChromaVectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("build_index")

def run_indexing(
    data_dir: Path,
    reset_collection: bool = False,
    override_classification: str = None,
    device: str = None,
    batch_size: int = None
):
    logger.info("=== Memulai Pipeline Indexing Dokumen PUSDIKZI (Project Emas) ===")
    start_time = time.time()

    # 1. Inisialisasi Vector Store (ChromaDB)
    chroma_cfg = settings.get("chromadb", {})
    vector_store = ChromaVectorStore(
        host=chroma_cfg.get("host", "localhost"),
        port=chroma_cfg.get("port", 8000),
        collection_name=chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")
    )

    if not vector_store.is_healthy():
        logger.error(
            f"Koneksi ke ChromaDB gagal pada {vector_store.host}:{vector_store.port}!\n"
            "Pastikan container ChromaDB Docker sudah berjalan dengan perintah:\n"
            "   docker compose up -d"
        )
        sys.exit(1)

    logger.info(f"Terhubung ke ChromaDB Docker pada {vector_store.host}:{vector_store.port}")

    if reset_collection:
        logger.warning(f"Mereset koleksi '{vector_store.collection_name}'...")
        try:
            vector_store.client.delete_collection(vector_store.collection_name)
            vector_store.collection = vector_store.get_or_create_collection(vector_store.collection_name)
            logger.info("Koleksi berhasil direset.")
        except Exception as e:
            logger.warning(f"Catatan saat reset koleksi: {e}")

    # 2. Ingestion Dokumen Mentah
    loader = DocumentLoader(raw_dir=str(data_dir))
    if override_classification:
        loader.default_classification = override_classification.upper()

    logger.info(f"Memuat dokumen dari direktori: {data_dir.resolve()}")
    pages = loader.load_directory()

    if not pages:
        logger.warning(f"Tidak ada dokumen yang ditemukan di {data_dir.resolve()}.")
        logger.info("Silakan letakkan file .pdf, .txt, atau .md di direktori data/raw/.")
        return

    logger.info(f"Total bagian/halaman dokumen yang berhasil diekstrak: {len(pages)}")

    # 3. Processing & Chunking
    chunker = TextChunker()
    chunks = chunker.chunk_pages(pages)
    logger.info(f"Total teks terpotong menjadi {len(chunks)} chunks.")

    if not chunks:
        logger.warning("Tidak ada teks valid setelah proses sanitasi/chunking.")
        return

    # 4. Generate Embedding
    emb_cfg = settings.get("embedding", {})
    logger.info(f"Menginisialisasi Embedding Engine (Model: {emb_cfg.get('model_name')})...")
    engine = EmbeddingEngine(device=device, batch_size=batch_size)
    
    texts_to_embed = [c.text for c in chunks]
    logger.info(f"Memproses kalkulasi {len(texts_to_embed)} vector embedding secara lokal (Zero Data Leakage)...")
    t0 = time.time()
    embeddings = engine.embed_documents(texts_to_embed)
    t_emb = time.time() - t0
    logger.info(f"Selesai menghasilkan embedding dalam {t_emb:.2f} detik.")

    # 5. Simpan ke ChromaDB Docker
    logger.info("Menyimpan chunks dan vektor ke ChromaDB...")
    total_stored = vector_store.upsert_chunks(chunks, embeddings)

    # Bebaskan cache VRAM accelerator
    if hasattr(engine, "_empty_accelerator_cache"):
        engine._empty_accelerator_cache()

    # 6. Ringkasan
    elapsed = time.time() - start_time
    stats = vector_store.get_stats()
    logger.info("==================================================")
    logger.info("        LAPORAN INDEXING DATABASE VEKTOR         ")
    logger.info("==================================================")
    logger.info(f"Status              : Sukses")
    logger.info(f"Dokumen Pages       : {len(pages)}")
    logger.info(f"Total Chunks        : {total_stored}")
    logger.info(f"Total Vektor di DB  : {stats.get('total_vectors')}")
    logger.info(f"Koleksi ChromaDB    : {stats.get('collection_name')}")
    logger.info(f"Waktu Eksekusi      : {elapsed:.2f} detik")
    logger.info("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script build & update vector index untuk Project Emas PUSDIKZI")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw",
        help="Direktori berisi file dokumen mentah (PDF/TXT/MD)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Hapus dan buat ulang koleksi ChromaDB sebelum indexing"
    )
    parser.add_argument(
        "--classification",
        type=str,
        default=None,
        help="Paksa klasifikasi keamanan RBAC (misal: SECRET, INTERNAL, PUBLIC)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device embedding: xpu atau cpu (default dari config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size embedding (default: 16 untuk mencegah XPU OOM)"
    )
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    run_indexing(
        data_path,
        reset_collection=args.reset,
        override_classification=args.classification,
        device=args.device,
        batch_size=args.batch_size
    )

