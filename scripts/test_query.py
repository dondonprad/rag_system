#!/usr/bin/env python3
"""Script Pengujian End-to-End RAG PUSDIKZI dengan LLM Lokal (Ollama + Intel IPEX-LLM).

Menguji 4 aturan utama:
1. Integrasi kueri dengan database vektor ChromaDB untuk Context Injection & Prompt Engineering Militer.
2. Strict Grounding: Menjawab hanya berdasarkan potongan konteks yang disuplai oleh ChromaDB.
3. Penanganan kondisi No Match ("Informasi tidak ditemukan dalam dokumen doktrin PUSDIKZI.").
4. Penyertaan sitasi wajib (Nama Dokumen dan Halaman).
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import httpx

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from src.embeddings.embedding_engine import EmbeddingEngine
from src.vector_store.chroma_db import ChromaVectorStore
from src.llm.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("test_query")


def execute_rag_query(
    query_text: str,
    top_k: int = 3,
    access_classifications: Optional[List[str]] = None,
    max_distance: Optional[float] = None,
    use_api: bool = False,
    api_url: str = "http://localhost:8080/api/v1/query",
    output_json: bool = False,
    vector_store: Optional[ChromaVectorStore] = None,
    embedding_engine: Optional[EmbeddingEngine] = None,
    llm_client: Optional[OllamaClient] = None,
):
    result = None
    if use_api:
        try:
            payload = {
                "query": query_text,
                "top_k": top_k,
                "allowed_classifications": access_classifications,
                "max_distance": max_distance,
            }
            resp = httpx.post(api_url, json=payload, timeout=300.0)
            resp.raise_for_status()
            result = resp.json()
        except httpx.ConnectError:
            logger.warning(
                f"Server FastAPI ({api_url}) belum berjalan. "
                "Beralih otomatis ke mode eksekusi pipeline langsung (ChromaDB + Ollama IPEX-LLM)..."
            )

    if result is None:
        chroma_cfg = settings.get("chromadb", {})
        vec_store = vector_store or ChromaVectorStore(
            host=chroma_cfg.get("host", "localhost"),
            port=chroma_cfg.get("port", 8000),
            collection_name=chroma_cfg.get("collection_name", "pusdikzi_knowledge_base"),
        )
        if not vec_store.is_healthy():
            logger.error("ChromaDB tidak aktif! Jalankan: docker compose up -d")
            sys.exit(1)

        emb_engine = embedding_engine or EmbeddingEngine()
        ollama = llm_client or OllamaClient()

        if not ollama.is_healthy():
            logger.error(
                f"Container Ollama IPEX-LLM ({ollama.base_url}) tidak merespons! "
                "Pastikan container aktif dengan: docker compose up -d"
            )
            sys.exit(1)

        query_vec = emb_engine.embed_query(query_text)
        chunks = vec_store.search_similarity(
            query_embedding=query_vec,
            n_results=top_k,
            allowed_classifications=access_classifications,
        )
        result = ollama.answer_with_context(
            query=query_text,
            retrieved_chunks=chunks,
            max_distance_threshold=max_distance,
        )

    if output_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result

    print("\n" + "=" * 75)
    print(" SISTEM TANYA JAWAB RAG MILITER PUSDIKZI (OLLAMA + INTEL IPEX-LLM)")
    print("=" * 75)
    print(f"Model LLM       : {result.get('model', 'qwen2.5-7b-int4')} (4-bit GGUF / INT4)")
    print(f"Kueri Pengguna  : {query_text}")
    print(f"Filter RBAC     : {access_classifications or ['ALL (INTERNAL + SECRET)']}")
    print(f"Status No-Match : {result.get('no_match', False)}")
    print("-" * 75)
    print("JAWABAN TERINTEGRASI:")
    print(result.get("answer", ""))
    print("-" * 75)

    citations = result.get("citations", [])
    if citations:
        print(f"SITASI DOKUMEN ({len(citations)} Sumber):")
        for idx, c in enumerate(citations, start=1):
            print(
                f"  [{idx}] Nama Dokumen : {c.get('document')} | "
                f"Halaman : {c.get('page')} | "
                f"Klasifikasi : {c.get('classification')} | "
                f"Cosine Dist : {c.get('distance')}"
            )
    else:
        print("SITASI DOKUMEN  : - (Tidak ada konteks dokumen yang cocok)")
    print("=" * 75 + "\n")
    return result


def run_verification_suite(use_api: bool = False):
    """Menjalankan 3 skenario pengujian otomatis sesuai aturan PRD."""
    chroma_cfg = settings.get("chromadb", {})
    vec_store = ChromaVectorStore(
        host=chroma_cfg.get("host", "localhost"),
        port=chroma_cfg.get("port", 8000),
        collection_name=chroma_cfg.get("collection_name", "pusdikzi_knowledge_base"),
    )
    emb_engine = EmbeddingEngine()
    llm_client = OllamaClient()

    scenarios = [
        {
            "title": "SKENARIO 1: Kueri Dokumen INTERNAL (Strict Grounding + Sitasi Wajib)",
            "query": "Dimana kedudukan PUSDIKZI dan sebutkan tugas pokok Korps Zeni?",
            "access": ["INTERNAL"],
        },
        {
            "title": "SKENARIO 2: Kueri Dokumen Rahasia / RBAC SECRET (Protokol Jihandak)",
            "query": "Berapa personel minimal dalam tim Jihandak dan berapa jarak isolasi aman bahan peledak di bawah 5 kg TNT?",
            "access": ["SECRET"],
        },
        {
            "title": "SKENARIO 3: Kondisi No Match (Pertanyaan Di Luar Dokumen PUSDIKZI)",
            "query": "Bagaimana resep dan langkah-langkah membuat kue bolu coklat kukus?",
            "access": ["INTERNAL"],
        },
    ]

    for sc in scenarios:
        print(f"\n>>> {sc['title']}")
        execute_rag_query(
            query_text=sc["query"],
            top_k=3,
            access_classifications=sc["access"],
            use_api=use_api,
            vector_store=vec_store,
            embedding_engine=emb_engine,
            llm_client=llm_client,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Uji coba RAG Query dengan Ollama + Intel IPEX-LLM (Qwen 2.5-7B 4-bit GGUF)"
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Dimana kedudukan PUSDIKZI dan apa tugas pokok Korps Zeni?",
        help="Pertanyaan untuk diajukan ke sistem RAG PUSDIKZI",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Jumlah chunk konteks dari ChromaDB")
    parser.add_argument(
        "--access",
        type=str,
        default=None,
        help="Filter RBAC klasifikasi keamanan (misal: INTERNAL, SECRET)",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Ambang batas maksimum Cosine Distance (default: 0.42)",
    )
    parser.add_argument(
        "--suite",
        action="store_true",
        help="Jalankan paket pengujian 3 skenario (INTERNAL, SECRET, dan No-Match)",
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Uji melalui endpoint HTTP FastAPI (/api/v1/query)",
    )
    parser.add_argument("--json", action="store_true", help="Output hasil dalam format JSON")

    args = parser.parse_args()
    access_list = [c.strip().upper() for c in args.access.split(",")] if args.access else None

    try:
        if args.suite:
            run_verification_suite(use_api=args.api)
        else:
            execute_rag_query(
                query_text=args.query,
                top_k=args.top_k,
                access_classifications=access_list,
                max_distance=args.max_distance,
                use_api=args.api,
                output_json=args.json,
            )
    except KeyboardInterrupt:
        logger.warning("\nEksekusi dihentikan oleh pengguna (KeyboardInterrupt).")
        sys.exit(130)
    except httpx.TimeoutException as e:
        logger.error(
            f"Waktu tunggu inferensi habis (Timeout): {e}. "
            "Model 7B sedang dimuat ke VRAM GPU Intel untuk pertama kali. Silakan ulangi perintah."
        )
        sys.exit(1)
    except httpx.HTTPError as e:
        logger.error(f"Kesalahan komunikasi HTTP ke layanan LLM/RAG: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()

