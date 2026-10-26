#!/usr/bin/env python3
"""Script Khusus Penanganan Download & Sinkronisasi File JSON Vektor dari VM Google Colab
Dipadukan dengan `scripts/update_index.py` agar HANYA data index dokumen baru/berubah
yang disimpan ke database ChromaDB lokal.

Cara Penggunaan:
  1. Otomatis mendeteksi URL tunnel / output dari notebook/build_index.ipynb & jalankan update_index:
     ./venv/bin/python scripts/pull_colab_index.py

  2. Menggunakan URL download langsung dari output Sel 6 di VM Google Colab:
     ./venv/bin/python scripts/pull_colab_index.py --url https://xxxx.trycloudflare.com/colab_indexed_vectors.json.xz
"""

import re
import sys
import json
import lzma
import base64
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import requests

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("pull_colab_index")

DEFAULT_NOTEBOOK_PATH = BASE_DIR / "notebook" / "build_index.ipynb"
DEFAULT_JSON_PATH = BASE_DIR / "data" / "processed" / "colab_indexed_vectors.json"
SECRET_JSON_PATH = BASE_DIR / "data" / "raw" / "SECRET" / "colab_indexed_vectors.json"
DEFAULT_ZIP_DIR = BASE_DIR / "data" / "zip"
DEFAULT_ZIP_PATH = DEFAULT_ZIP_DIR / "data_colab.zip"


def _write_json_to_targets(json_bytes: bytes, primary_path: Path) -> Path:
    """Menyimpan file JSON ke data/processed/ dan menyalinnya juga ke data/raw/SECRET/."""
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    primary_path.write_bytes(json_bytes)
    try:
        SECRET_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        if SECRET_JSON_PATH.resolve() != primary_path.resolve():
            SECRET_JSON_PATH.write_bytes(json_bytes)
    except Exception:
        pass
    return primary_path


def inspect_notebook_for_colab_data(notebook_path: Path) -> Dict[str, Any]:
    """Memeriksa file notebook/build_index.ipynb untuk menemukan:
    1. Payload LZMA Base64 yang disematkan di output Sel 6 (`application/vnd.pusdikzi.colab_vectors+lzma`)
    2. URL tunnel `trycloudflare.com` terakhir yang tercetak di stdout Sel 6
    3. Status apakah Sel 6 sudah selesai dieksekusi (execution_count atau keyword penanda)
    """
    result: Dict[str, Any] = {
        "embedded_b64": None,
        "tunnel_url": None,
        "cell_completed": False,
        "total_vectors": None,
        "stdout_summary": ""
    }
    if not notebook_path.exists():
        return result

    try:
        nb_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell in reversed(nb_data.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            src = "".join(cell.get("source", []))
            # Identifikasi apakah ini sel ekspor terakhir (Sel 6)
            is_export_cell = "colab_indexed_vectors" in src or "ChromaVectorStore" in src or "cloudflared" in src

            outputs = cell.get("outputs", [])
            exec_count = cell.get("execution_count")

            if is_export_cell and exec_count is not None:
                result["cell_completed"] = True

            for out in outputs:
                # Cek display_data metadata
                data_field = out.get("data", {})
                if "application/vnd.pusdikzi.colab_vectors+lzma" in data_field:
                    val = data_field["application/vnd.pusdikzi.colab_vectors+lzma"]
                    result["embedded_b64"] = "".join(val) if isinstance(val, list) else str(val)
                    result["cell_completed"] = True

                # Cek teks stdout untuk URL trycloudflare atau penanda sukses
                text_lines = out.get("text", [])
                full_text = "".join(text_lines) if isinstance(text_lines, list) else str(text_lines)

                if "Total vektor di dalam koleksi" in full_text or "Seluruh folder 'data/' berhasil dikemas" in full_text:
                    result["cell_completed"] = True
                    result["stdout_summary"] = full_text.strip()
                    m_vec = re.search(r"Total vektor di dalam koleksi [^:]+: (\d+)", full_text)
                    if m_vec:
                        result["total_vectors"] = int(m_vec.group(1))

                # Cari URL tunnel fleksibel
                m = re.search(
                    r"https://[a-zA-Z0-9-]+\.trycloudflare\.com[^\s\"\'<>]*",
                    full_text
                )
                if m and not result["tunnel_url"]:
                    url_candidate = m.group(0).rstrip(".,)")
                    # Standarisasi jika hanya base domain
                    if not url_candidate.endswith((".zip", ".xz", ".json")):
                        result["tunnel_url"] = f"{url_candidate}/data_colab.zip"
                    else:
                        result["tunnel_url"] = url_candidate
                    result["cell_completed"] = True

            if is_export_cell:
                break
    except Exception as e:
        logger.debug(f"Gagal memeriksa output notebook: {e}")

    return result


def download_from_colab_url(url: str, target_json_path: Path) -> Path:
    """Mengunduh file data_colab.zip, colab_indexed_vectors.json.xz, atau .json dari VM Google Colab."""
    import io
    import zipfile

    target_json_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ZIP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Mengunduh data hasil indexing dari VM Google Colab: {url}")

    resp = requests.get(url, stream=True, timeout=(10, 180))
    resp.raise_for_status()
    raw_content = resp.content

    if url.endswith(".zip") or raw_content[:4] == b"PK\x03\x04":
        zip_save_path = DEFAULT_ZIP_PATH
        zip_save_path.write_bytes(raw_content)
        logger.info(f"File zip berhasil disimpan ke: {zip_save_path.resolve()} ({len(raw_content) / (1024 * 1024):.2f} MB)")
        logger.info(f"Mengekstrak arsip folder data ke {BASE_DIR / 'data'}...")
        with zipfile.ZipFile(io.BytesIO(raw_content), "r") as zf:
            zf.extractall(BASE_DIR)
        if target_json_path.exists():
            _write_json_to_targets(target_json_path.read_bytes(), target_json_path)
            return target_json_path
        elif SECRET_JSON_PATH.exists():
            _write_json_to_targets(SECRET_JSON_PATH.read_bytes(), target_json_path)
            return target_json_path

    if url.endswith(".xz") or raw_content[:6] == b"\xfd7zXZ\x00":
        logger.info(f"Mengekstrak kompresi LZMA ({len(raw_content) / (1024 * 1024):.2f} MB)...")
        json_bytes = lzma.decompress(raw_content)
    else:
        json_bytes = raw_content

    _write_json_to_targets(json_bytes, target_json_path)
    logger.info(
        f"Berhasil menyimpan JSON hasil indexing Colab ke: {target_json_path} & {SECRET_JSON_PATH} "
        f"({len(json_bytes) / (1024 * 1024):.2f} MB)"
    )
    return target_json_path


def fetch_colab_vectors_json(
    colab_url: Optional[str] = None,
    notebook_path: Path = DEFAULT_NOTEBOOK_PATH,
    output_json_path: Path = DEFAULT_JSON_PATH
) -> Path:
    """Mengambil file JSON hasil indexing dari VM Google Colab melalui:
    1. URL download eksplisit (`--colab-url` / `--url`)
    2. URL tunnel atau embedded output di dalam `notebook/build_index.ipynb`
    3. File arsip zip di `data/zip/data_colab.zip`
    4. File lokal `data/processed/colab_indexed_vectors.json` (atau `.json.xz`) jika sudah ada
    """
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_ZIP_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Jika URL diberikan secara eksplisit, unduh langsung dari VM Colab
    if colab_url:
        return download_from_colab_url(colab_url.strip(), output_json_path)

    # 2. Periksa notebook/build_index.ipynb untuk embedded data atau URL tunnel aktif
    nb_info = inspect_notebook_for_colab_data(notebook_path)

    if nb_info.get("tunnel_url"):
        try:
            return download_from_colab_url(nb_info["tunnel_url"], output_json_path)
        except Exception as e:
            logger.warning(f"Tunnel URL di notebook sudah tidak aktif ({e}), mencoba metode berikutnya...")

    if nb_info.get("embedded_b64"):
        try:
            logger.info(f"Mengekstrak data vektor Colab langsung dari output {notebook_path.name}...")
            xz_bytes = base64.b64decode(nb_info["embedded_b64"])
            json_bytes = lzma.decompress(xz_bytes)
            _write_json_to_targets(json_bytes, output_json_path)
            logger.info(
                f"Berhasil mengekstrak JSON vektor ke: {output_json_path} & {SECRET_JSON_PATH} "
                f"({len(json_bytes) / (1024 * 1024):.2f} MB)"
            )
            return output_json_path
        except Exception as e:
            logger.warning(f"Gagal mengekstrak embedded output dari notebook: {e}")

    # 3. Cek jika ada file ZIP lokal di data/zip/data_colab.zip (atau root data_colab.zip)
    zip_candidate = DEFAULT_ZIP_PATH if DEFAULT_ZIP_PATH.exists() else (BASE_DIR / "data_colab.zip")
    if zip_candidate.exists() and (
        not output_json_path.exists() or zip_candidate.stat().st_mtime > output_json_path.stat().st_mtime
    ):
        logger.info(f"Mengekstrak file arsip lokal {zip_candidate.name} dari {zip_candidate.parent}...")
        import zipfile
        with zipfile.ZipFile(zip_candidate, "r") as zf:
            zf.extractall(BASE_DIR)
        if output_json_path.exists():
            return output_json_path

    # 4. Cek jika ada file .json.xz lokal yang baru diunduh
    xz_local = output_json_path.with_suffix(".json.xz")
    if xz_local.exists() and (
        not output_json_path.exists() or xz_local.stat().st_mtime > output_json_path.stat().st_mtime
    ):
        logger.info(f"Mengekstrak file lokal {xz_local.name}...")
        json_bytes = lzma.decompress(xz_local.read_bytes())
        output_json_path.write_bytes(json_bytes)
        return output_json_path

    # 5. Gunakan file colab_indexed_vectors.json lokal bila sudah tersedia di data/raw/SECRET atau data/processed
    if SECRET_JSON_PATH.exists() and not output_json_path.exists():
        output_json_path.write_bytes(SECRET_JSON_PATH.read_bytes())
        return output_json_path
    if output_json_path.exists():
        if not SECRET_JSON_PATH.exists():
            try:
                SECRET_JSON_PATH.write_bytes(output_json_path.read_bytes())
            except Exception:
                pass
        logger.info(f"Menggunakan file vektor Colab lokal: {output_json_path}")
        return output_json_path

    raise FileNotFoundError(
        "File hasil indexing Google Colab belum ditemukan.\n"
        "Pastikan Anda telah menjalankan Sel 6 di `notebook/build_index.ipynb` pada runtime Colab, lalu:\n"
        "  • Tekan Ctrl+S pada notebook di VS Code, ATAU\n"
        "  • Salin URL dari output Sel 6 dan jalankan:\n"
        "    ./venv/bin/python scripts/update_index.py --from-colab --colab-url <URL>"
    )


def load_colab_vectors_by_filename(json_path: Path) -> Dict[str, Dict[str, List[Any]]]:
    """Membaca `colab_indexed_vectors.json` dan mengelompokkan chunks beserta embeddings
    berdasarkan nama file sumber (`sumber` / `file_name`).
    """
    with open(json_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    ids = bundle.get("ids", [])
    docs = bundle.get("documents", [])
    metas = bundle.get("metadatas", [])
    embs = bundle.get("embeddings", [])

    grouped: Dict[str, Dict[str, List[Any]]] = {}
    for cid, doc_text, meta, emb in zip(ids, docs, metas, embs):
        if not isinstance(meta, dict):
            continue
        fname = meta.get("sumber") or meta.get("file_name")
        if not fname:
            continue
        if fname not in grouped:
            grouped[fname] = {"ids": [], "documents": [], "metadatas": [], "embeddings": []}
        grouped[fname]["ids"].append(cid)
        grouped[fname]["documents"].append(doc_text)
        grouped[fname]["metadatas"].append(meta)
        grouped[fname]["embeddings"].append(emb)

    return grouped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unduh file JSON hasil indexing dari VM Google Colab & simpan hanya dokumen baru ke ChromaDB lokal"
    )
    parser.add_argument(
        "--url",
        "--colab-url",
        dest="colab_url",
        type=str,
        default=None,
        help="URL download langsung dari output Sel 6 di VM Google Colab"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw/SECRET",
        help="Direktori dokumen yang diperiksa terhadap manifest lokal (default: ./data/raw/SECRET)"
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Hanya unduh/ekstrak file colab_indexed_vectors.json tanpa menjalankan update_index"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Paksa impor ulang seluruh dokumen dari JSON Colab meskipun sudah tercatat di manifest"
    )
    args = parser.parse_args()

    json_file = fetch_colab_vectors_json(colab_url=args.colab_url)
    if not args.download_only:
        from scripts.update_index import run_incremental_update
        run_incremental_update(
            data_dir=Path(args.data_dir),
            force_all=args.force,
            from_colab=True,
            colab_json_path=json_file
        )

