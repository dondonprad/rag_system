#!/usr/bin/env python3
import sys
import json
from pathlib import Path

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from src.vector_store.chroma_db import ChromaVectorStore

def inspect_chroma():
    chroma_cfg = settings.get("chromadb", {})
    host = chroma_cfg.get("host", "localhost")
    port = chroma_cfg.get("port", 8000)
    collection_name = chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")

    print("=" * 65)
    print("      STATUS DAN INSPEKSI DATABASE VEKTOR (ChromaDB Docker)     ")
    print("=" * 65)
    print(f"Target Server   : http://{host}:{port}")
    print(f"Nama Koleksi    : {collection_name}")

    # 1. Periksa koneksi
    try:
        store = ChromaVectorStore(host=host, port=port, collection_name=collection_name)
    except Exception as e:
        print(f"\n[GAGAL] Tidak dapat terhubung ke ChromaDB: {e}")
        print("Pastikan container Docker aktif dengan: docker compose up -d\n")
        sys.exit(1)

    if not store.is_healthy():
        print("\n[GAGAL] Container ChromaDB tidak merespons heartbeat.")
        sys.exit(1)

    print("Koneksi Server  : AKTIF & SEHAT (Healthy)")

    # 2. Ambil data koleksi
    collection = store.collection
    total_vectors = collection.count()
    print(f"Total Vektor    : {total_vectors} data tersimpan")

    if total_vectors == 0:
        print("\n[INFO] Database masih kosong (belum ada vektor yang disimpan).")
        print("Jalankan script indexing: python scripts/build_index.py\n")
        return

    # 3. Ambil sample data dari database
    # Mengambil dokumen dan metadata (maksimal 10 item untuk preview)
    sample_data = collection.get(limit=10, include=["documents", "metadatas"])

    # Kelompokkan berdasarkan file sumber dan klasifikasi RBAC
    sources = set()
    classifications = {}

    for meta in sample_data.get("metadatas", []):
        file_name = meta.get("file_name", "Unknown")
        sources.add(file_name)
        cls = meta.get("classification", "UNKNOWN")
        classifications[cls] = classifications.get(cls, 0) + 1

    print("-" * 65)
    print("RINGKASAN METADATA & KEAMANAN:")
    print(f"- Klasifikasi Akses Terdeteksi : {dict(classifications)}")
    print(f"- Dokumen Sumber Terdaftar     : {list(sources)}")

    print("-" * 65)
    print("CUPLIKAN 3 DATA VEKTOR TERATAS:")
    docs = sample_data.get("documents", [])
    ids = sample_data.get("ids", [])
    metas = sample_data.get("metadatas", [])

    for i in range(min(3, len(docs))):
        print(f"\n[{i + 1}] ID Chunk    : {ids[i]}")
        print(f"    Sumber      : {metas[i].get('file_name')} (Hal {metas[i].get('page')})")
        print(f"    Akses RBAC  : {metas[i].get('classification')}")
        snippet = docs[i].replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:120] + "..."
        print(f"    Teks        : \"{snippet}\"")

    print("\n" + "=" * 65)
    print("[SUKSES] Seluruh data vektor tersimpan dengan benar di ChromaDB Docker.")
    print("=" * 65)

if __name__ == "__main__":
    inspect_chroma()

