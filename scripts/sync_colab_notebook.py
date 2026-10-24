#!/usr/bin/env python3
"""Script Sinkronisasi Dokumen Inkremental ke Notebook Google Colab (sync_colab_notebook.py)

Fitur:
- Membedakan dokumen LAMA (sudah terindeks di .index_manifest.json) vs dokumen BARU / TERMODIFIKASI
- Melewatkan (SKIP) dokumen lama agar tidak diproses ulang
- Hanya mengekstrak dan mengompresi (LZMA) dokumen BARU / TERMODIFIKASI ke dalam `notebook/build_index.ipynb`
- Mendukung opsi `--status` (cek daftar file lama vs baru), `--mark-indexed` (tandai sudah terindeks di manifest),
  dan `--all` / `--force` (jika ingin menyertakan seluruh dokumen termasuk dokumen lama)
"""

import sys
import json
import lzma
import time
import base64
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.ingestion.loaders import DocumentLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("sync_colab_notebook")

MANIFEST_DEFAULT_PATH = BASE_DIR / "data" / "processed" / ".index_manifest.json"


def compute_file_hash(file_path: Path) -> str:
    """Menghitung hash SHA-256 dari file dokumen."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Membaca file .index_manifest.json."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Gagal membaca manifest {manifest_path}: {e}")
    return {"version": "1.0", "last_updated": None, "documents": {}}


def save_manifest(manifest: Dict[str, Any], manifest_path: Path):
    """Menyimpan file .index_manifest.json secara atomik."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["last_updated"] = datetime.now().isoformat()
    temp_path = manifest_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    temp_path.replace(manifest_path)


def classify_documents(
    data_dir: Path,
    manifest: Dict[str, Any],
    supported_extensions: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Memisahkan file di data_dir menjadi:
    - new_files: Dokumen baru yang belum ada di manifest
    - modified_files: Dokumen lama yang isinya berubah (SHA-256 berbeda)
    - unchanged_files: Dokumen lama yang tidak berubah (SKIP)
    """
    manifest_docs = manifest.get("documents", {})

    # Buat indeks pencarian berdasarkan nama file maupun relative path di manifest
    manifest_by_name: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for key_path, meta in manifest_docs.items():
        fname = meta.get("file_name") or Path(key_path).name
        manifest_by_name[fname] = (key_path, meta)
        manifest_by_name[key_path] = (key_path, meta)

    files = sorted(
        [
            p for p in data_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in supported_extensions
            and "Zone.Identifier" not in p.name
        ],
        key=lambda x: x.name
    )

    new_files = []
    modified_files = []
    unchanged_files = []

    raw_root = BASE_DIR / "data" / "raw"

    for f_path in files:
        try:
            rel_key = str(f_path.relative_to(raw_root))
        except ValueError:
            rel_key = f_path.name

        stat = f_path.stat()
        curr_hash = compute_file_hash(f_path)
        info = {
            "path": f_path,
            "name": f_path.name,
            "rel_key": rel_key,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "sha256": curr_hash,
        }

        matched = manifest_by_name.get(rel_key) or manifest_by_name.get(f_path.name)
        if matched:
            _, old_meta = matched
            if old_meta.get("sha256") == curr_hash:
                info["chunks_count"] = old_meta.get("chunks_count", 0)
                info["indexed_at"] = old_meta.get("indexed_at", "-")
                unchanged_files.append(info)
            else:
                info["old_sha256"] = old_meta.get("sha256")
                modified_files.append(info)
        else:
            new_files.append(info)

    return new_files, modified_files, unchanged_files


def sync_secret_to_notebook(
    data_dir: Path,
    notebook_path: Path,
    manifest_path: Path = MANIFEST_DEFAULT_PATH,
    classification: str = None,
    force_all: bool = False,
    dry_run: bool = False,
    mark_indexed: bool = False
) -> dict:
    start_time = time.time()

    if not data_dir.exists():
        logger.error(f"Direktori sumber dokumen tidak ditemukan: {data_dir.resolve()}")
        sys.exit(1)

    if not notebook_path.exists():
        logger.error(f"File notebook tidak ditemukan: {notebook_path.resolve()}")
        sys.exit(1)

    loader = DocumentLoader(raw_dir=str(data_dir))
    if classification:
        loader.default_classification = classification.upper()
    elif data_dir.name.upper() in ["PUBLIC", "INTERNAL", "RESTRICTED", "CONFIDENTIAL", "SECRET"]:
        loader.default_classification = data_dir.name.upper()

    manifest = load_manifest(manifest_path)
    new_files, modified_files, unchanged_files = classify_documents(
        data_dir=data_dir,
        manifest=manifest,
        supported_extensions=loader.supported_extensions
    )

    if force_all:
        files_to_process = new_files + modified_files + unchanged_files
        skipped_files = []
    else:
        files_to_process = new_files + modified_files
        skipped_files = unchanged_files

    print("\n" + "=" * 72)
    print("   STATUS PEMILAHAN DOKUMEN LAMA VS BARU (INCREMENTAL COLAB SYNC)")
    print("=" * 72)
    print(f"Folder Target           : {data_dir.resolve()}")
    print(f"Manifest Acuan          : {manifest_path.resolve()}")
    print(f"Mode Sinkronisasi       : {'SEMUA DOKUMEN (--all)' if force_all else 'HANYA DOKUMEN BARU / BERUBAH (Skip Dokumen Lama)'}")
    print("-" * 72)
    print(f"• Dokumen Baru          : {len(new_files)} file (Akan diproses)")
    print(f"• Dokumen Termodifikasi : {len(modified_files)} file (Akan diproses ulang)")
    print(f"• Dokumen Lama (Tetap)  : {len(unchanged_files)} file ({'Ikut diproses' if force_all else 'DILEWATI / SKIP'})")
    print("-" * 72)

    if skipped_files:
        print(f"Daftar Dokumen Lama yang Dilewati (Skip - {len(skipped_files)} file):")
        for idx, f in enumerate(skipped_files, 1):
            print(f"  [SKIP {idx:02d}] {f['name']} ({f['size'] / 1024:.1f} KB)")
        print("-" * 72)

    if files_to_process:
        print(f"Daftar Dokumen Baru/Berubah yang Akan Diproses ({len(files_to_process)} file):")
        for idx, f in enumerate(files_to_process, 1):
            tag = "BARU" if f in new_files else ("UBAH" if f in modified_files else "LAMA")
            print(f"  [{tag:4s} {idx:02d}] {f['name']} ({f['size'] / 1024:.1f} KB)")
    else:
        print("✅ Tidak ada dokumen baru atau berubah. Seluruh dokumen sudah terindeks di manifest!")
    print("=" * 72)

    if dry_run:
        print("\nMode --status / --dry-run selesai. Tidak ada perubahan pada notebook.\n")
        return {
            "new_count": len(new_files),
            "modified_count": len(modified_files),
            "skipped_count": len(skipped_files),
        }

    # Ekstrak HANYA dokumen yang masuk daftar files_to_process
    pages = []
    processed_file_infos = []
    for f_info in files_to_process:
        f_path = f_info["path"]
        doc_pages = loader.load_file(f_path, classification=loader.default_classification)
        if doc_pages:
            pages.extend(doc_pages)
            processed_file_infos.append(f_info)
            logger.info(f"Dimuat {len(doc_pages)} bagian/halaman dari dokumen baru: {f_path.name}")

    # Serialisasi ke JSON & kompresi menggunakan LZMA (preset=9)
    records = [p.to_dict() for p in pages]
    raw_json_bytes = json.dumps(records, ensure_ascii=False).encode("utf-8")
    compressed_bytes = lzma.compress(raw_json_bytes, preset=9)
    b64_payload = base64.b64encode(compressed_bytes).decode("ascii")

    b64_lines = [
        f'    "{b64_payload[i : i + 120]}"\n'
        for i in range(0, len(b64_payload), 120)
    ]

    total_files = len(processed_file_infos)
    total_pages = len(pages)

    # Baca dan perbarui notebook/build_index.ipynb
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    updated_cell = False
    for cell in nb_data.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source_lines = cell.get("source", [])
        start_idx = None
        end_idx = None

        for idx, line in enumerate(source_lines):
            if "_BUNDLED_SECRET_PAGES_LZMA_B64 = (" in line:
                start_idx = idx
            elif start_idx is not None and line.strip() == ")":
                end_idx = idx
                break

        if start_idx is not None and end_idx is not None:
            if start_idx >= 2 and source_lines[start_idx - 2].startswith("# Snapshot terkompresi (LZMA)"):
                source_lines[start_idx - 2] = (
                    f"# Snapshot terkompresi (LZMA) dari {total_files} dokumen BARU di folder "
                    f"{data_dir.as_posix()} ({total_pages} bagian/lembar kerja, {len(skipped_files)} dokumen lama di-skip)\n"
                )

            new_source = (
                source_lines[: start_idx + 1]
                + b64_lines
                + source_lines[end_idx:]
            )

            for idx_line, s_line in enumerate(new_source):
                if "Modul Ingestion, TextCleaner, TextChunker & Bundel" in s_line:
                    new_source[idx_line] = (
                        f'print("✅ Modul Ingestion, TextCleaner, TextChunker & Bundel '
                        f'{total_files} Dokumen BARU SECRET ({len(skipped_files)} dokumen lama di-skip) siap digunakan!")\n'
                    )

            cell["source"] = new_source
            updated_cell = True
            break

    if not updated_cell:
        logger.error(
            f"Variabel `_BUNDLED_SECRET_PAGES_LZMA_B64` tidak ditemukan di dalam {notebook_path.name}!"
        )
        sys.exit(1)

    temp_nb = notebook_path.with_suffix(".ipynb.tmp")
    with open(temp_nb, "w", encoding="utf-8") as f:
        json.dump(nb_data, f, indent=1, ensure_ascii=False)
    temp_nb.replace(notebook_path)

    # Jika --mark-indexed diaktifkan, catat dokumen yang diproses ke dalam .index_manifest.json
    if mark_indexed and processed_file_infos:
        for f_info in processed_file_infos:
            manifest["documents"][f_info["rel_key"]] = {
                "file_name": f_info["name"],
                "file_path": str(f_info["path"].resolve()),
                "sha256": f_info["sha256"],
                "size_bytes": f_info["size"],
                "mtime": f_info["mtime"],
                "classification": loader.default_classification,
                "indexed_at": datetime.now().isoformat(),
                "status": "synced_to_colab",
            }
        save_manifest(manifest, manifest_path)
        logger.info(f"Manifest {manifest_path.name} diperbarui untuk {len(processed_file_infos)} dokumen.")

    elapsed = time.time() - start_time
    raw_kb = len(raw_json_bytes) / 1024
    b64_kb = len(b64_payload) / 1024
    ratio = (1 - b64_kb / raw_kb) * 100 if raw_kb > 0 else 0.0

    print("\n" + "=" * 72)
    print("   LAPORAN SINKRONISASI SNAPSHOT LZMA KE NOTEBOOK COLAB")
    print("=" * 72)
    print(f"Dokumen Lama Dilewati   : {len(skipped_files)} file (Tidak diproses ulang)")
    print(f"Dokumen Baru Diproses   : {total_files} file ({total_pages} halaman/sheet)")
    print(f"Ukuran Teks Mentah      : {raw_kb:.1f} KB")
    print(f"Ukuran Snapshot LZMA    : {b64_kb:.1f} KB (Hemat {ratio:.1f}%)")
    print(f"Waktu Pemrosesan        : {elapsed:.2f} detik")
    print("=" * 72)
    print("✅ Snapshot LZMA (khusus dokumen baru) di build_index.ipynb berhasil diperbarui!\n")

    return {
        "total_files": total_files,
        "skipped_files": len(skipped_files),
        "total_pages": total_pages,
        "raw_kb": round(raw_kb, 1),
        "b64_kb": round(b64_kb, 1),
        "elapsed": round(elapsed, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sinkronisasi inkremental dokumen baru data/raw/SECRET ke snapshot LZMA di notebook/build_index.ipynb"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw/SECRET",
        help="Folder dokumen yang akan dipindai (default: ./data/raw/SECRET)"
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="./notebook/build_index.ipynb",
        help="Path ke file notebook target (default: ./notebook/build_index.ipynb)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_DEFAULT_PATH),
        help="Path ke file manifest riwayat indexing (default: ./data/processed/.index_manifest.json)"
    )
    parser.add_argument(
        "--classification",
        type=str,
        default=None,
        help="Paksa klasifikasi keamanan RBAC (misal: SECRET, INTERNAL, PUBLIC)"
    )
    parser.add_argument(
        "--all",
        "--force",
        dest="force_all",
        action="store_true",
        help="Proses ulang seluruh dokumen termasuk dokumen lama (abaikan manifest)"
    )
    parser.add_argument(
        "--status",
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Hanya tampilkan perbandingan dokumen lama vs dokumen baru tanpa mengubah notebook"
    )
    parser.add_argument(
        "--mark-indexed",
        action="store_true",
        help="Tandai dokumen baru yang disinkronkan ke dalam .index_manifest.json agar dianggap dokumen lama pada sinkronisasi berikutnya"
    )
    args = parser.parse_args()

    sync_secret_to_notebook(
        data_dir=Path(args.data_dir),
        notebook_path=Path(args.notebook),
        manifest_path=Path(args.manifest),
        classification=args.classification,
        force_all=args.force_all,
        dry_run=args.dry_run,
        mark_indexed=args.mark_indexed
    )
