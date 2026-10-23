import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

# Root Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Muat file .env jika ada
load_dotenv(BASE_DIR / ".env")

# Mitigasi kestabilan driver level-zero GPU pada CPU
if os.getenv("EMBEDDING_DEVICE", "cpu").lower() != "xpu":
    try:
        import torch
        if hasattr(torch, "xpu"):
            torch.xpu.is_available = lambda: False
    except Exception:
        pass

CONFIG_FILE = BASE_DIR / "config" / "config.yaml"

def load_config() -> Dict[str, Any]:
    """Membaca konfigurasi dari config.yaml dengan fallback override dari environment variable."""
    config: Dict[str, Any] = {}
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    # Override ChromaDB dengan env jika ada
    if "chromadb" not in config:
        config["chromadb"] = {}
    config["chromadb"]["host"] = os.getenv("CHROMA_HOST", config["chromadb"].get("host", "localhost"))
    config["chromadb"]["port"] = int(os.getenv("CHROMA_PORT", config["chromadb"].get("port", 8000)))
    config["chromadb"]["collection_name"] = os.getenv(
        "CHROMA_COLLECTION_NAME", config["chromadb"].get("collection_name", "pusdikzi_knowledge_base")
    )

    # Override Embedding dengan env jika ada
    if "embedding" not in config:
        config["embedding"] = {}
    config["embedding"]["model_name"] = os.getenv(
        "EMBEDDING_MODEL_NAME", config["embedding"].get("model_name", "nomic-ai/nomic-embed-text-v1.5")
    )
    config["embedding"]["device"] = os.getenv("EMBEDDING_DEVICE", config["embedding"].get("device", "cpu"))
    config["embedding"]["batch_size"] = int(
        os.getenv("EMBEDDING_BATCH_SIZE", config["embedding"].get("batch_size", 32))
    )

    # Override Ingestion dengan env jika ada
    if "ingestion" not in config:
        config["ingestion"] = {}
    config["ingestion"]["data_raw_dir"] = os.getenv(
        "DATA_RAW_DIR", config["ingestion"].get("data_raw_dir", "./data/raw")
    )
    config["ingestion"]["data_processed_dir"] = os.getenv(
        "DATA_PROCESSED_DIR", config["ingestion"].get("data_processed_dir", "./data/processed")
    )

    # Pastikan konfigurasi chunking
    if "chunking" not in config["ingestion"]:
        config["ingestion"]["chunking"] = {}
    config["ingestion"]["chunking"]["chunk_size"] = int(
        os.getenv("CHUNK_SIZE", config["ingestion"]["chunking"].get("chunk_size", 256))
    )
    config["ingestion"]["chunking"]["chunk_overlap"] = int(
        os.getenv("CHUNK_OVERLAP", config["ingestion"]["chunking"].get("chunk_overlap", 26))
    )
    config["ingestion"]["chunking"]["unit"] = config["ingestion"]["chunking"].get("unit", "tokens")
    config["ingestion"]["chunking"]["structure_aware"] = config["ingestion"]["chunking"].get("structure_aware", True)

    # Security
    if "security" not in config:
        config["security"] = {}
    config["security"]["default_classification"] = os.getenv(
        "DEFAULT_SECURITY_CLASSIFICATION", config["security"].get("default_classification", "INTERNAL")
    )

    return config

# Global config instance
settings = load_config()

