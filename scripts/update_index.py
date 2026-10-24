#!/usr/bin/env python3
"""Script Pembaruan Data Inkremental (Incremental Index Update) PUSDIKZI RAG

Fitur:
- Membedakan dokumen lama, baru, termodifikasi, dan terhapus pada folder yang sama
- Memproses HANYA dokumen baru dan termodifikasi (hemat CPU dan memori)
- Melewatkan (skip) dokumen lama yang tidak berubah tanpa parsing atau re-embedding
- Membersihkan chunk lama jika dokumen diedit atau dihapus dari folder
- Menyimpan riwayat manifest sidik jari dokumen (SHA-256, ukuran, waktu modifikasi)
- Mendukung mode pengecekan status (--status / --dry-run) tanpa mengubah database
"""

import sys
import os
import time
import json
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from src.ingestion.loaders import DocumentLoader
from src.processing.chunking import TextChunker
from src.embeddings.embedding_engine import EmbeddingEngine
from src.vector_store.chroma_db import ChromaVectorStore

# Cek ketersediaan library rich untuk tampilan terminal
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("update_index")

MANIFEST_DEFAULT_PATH = BASE_DIR / "data" / "processed" / ".index_manifest.json"

def compute_file_hash(file_path: Path) -> str:
    """Menghitung hash SHA-256 dari isi file untuk sidik jari dokumen."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()

def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Membaca file manifest riwayat indexing dokumen."""
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Gagal membaca manifest di {manifest_path} ({e}). Membuat manifest baru.")
    return {"version": "1.0", "last_updated": None, "documents": {}}

def save_manifest(manifest: Dict[str, Any], manifest_path: Path):
    """Menyimpan manifest riwayat indexing dokumen secara atomik."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest["last_updated"] = datetime.now().isoformat()
    temp_path = manifest_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    temp_path.replace(manifest_path)

def scan_folder_and_classify(
    target_dir: Path,
    manifest: Dict[str, Any],
    supported_extensions: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Memindai folder dan mengelompokkan dokumen ke dalam 4 kategori:
    1. new_files: Dokumen baru yang belum pernah diindeks
    2. modified_files: Dokumen lama yang isinya berubah (hash berbeda)
    3. unchanged_files: Dokumen lama yang tidak mengalami perubahan (skip)
    4. deleted_files: Dokumen yang ada di manifest tapi filenya sudah dihapus dari folder
    """
    manifest_docs = manifest.get("documents", {})

    # Indeks tambahan berdasarkan nama file agar tetap cocok baik dipindai dari ./data/raw maupun ./data/raw/SECRET
    manifest_by_name: Dict[str, Tuple[str, Dict[str, Any]]] = {}
    for k_path, m_val in manifest_docs.items():
        fname = m_val.get("file_name") or Path(k_path).name
        manifest_by_name[fname] = (k_path, m_val)
        manifest_by_name[k_path] = (k_path, m_val)

    raw_base_dir = (BASE_DIR / "data" / "raw").resolve()

    found_files = [
        p for p in target_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in supported_extensions and "Zone.Identifier" not in p.name
    ]

    new_files = []
    modified_files = []
    unchanged_files = []
    seen_rel_paths = set()

    for file_path in found_files:
        try:
            rel_path = str(file_path.resolve().relative_to(raw_base_dir))
        except ValueError:
            try:
                rel_path = str(file_path.relative_to(target_dir))
            except ValueError:
                rel_path = file_path.name

        seen_rel_paths.add(rel_path)
        seen_rel_paths.add(file_path.name)
        file_stat = file_path.stat()
        file_size = file_stat.st_size
        file_mtime = file_stat.st_mtime
        curr_hash = compute_file_hash(file_path)

        matched_entry = manifest_by_name.get(rel_path) or manifest_by_name.get(file_path.name)
        canonical_key = matched_entry[0] if matched_entry else rel_path
        seen_rel_paths.add(canonical_key)

        file_info = {
            "path": file_path,
            "rel_path": canonical_key,
            "name": file_path.name,
            "size": file_size,
            "mtime": file_mtime,
            "hash": curr_hash
        }

        # Cek apakah file sudah tercatat di manifest
        if matched_entry:
            _, old_meta = matched_entry
            old_hash = old_meta.get("sha256")

            if old_hash == curr_hash:
                # Dokumen lama tidak berubah
                file_info["old_chunks"] = old_meta.get("chunks_count", 0)
                file_info["last_indexed"] = old_meta.get("indexed_at", "-")
                unchanged_files.append(file_info)
            else:
                # Dokumen lama telah termodifikasi
                file_info["old_chunks"] = old_meta.get("chunks_count", 0)
                file_info["old_hash"] = old_hash
                modified_files.append(file_info)
        else:
            # Dokumen baru
            new_files.append(file_info)

    # Deteksi dokumen terhapus
    deleted_files = []
    for rel_path, old_meta in manifest_docs.items():
        if rel_path not in seen_rel_paths:
            # Pastikan dokumen memang berada di bawah direktori yang sedang dipindai
            old_path = Path(old_meta.get("file_path", target_dir / rel_path))
            if not old_path.exists():
                deleted_files.append(rel_path)

    return new_files, modified_files, unchanged_files, deleted_files

def display_status_table(
    target_dir: Path,
    new_files: List[Dict[str, Any]],
    modified_files: List[Dict[str, Any]],
    unchanged_files: List[Dict[str, Any]],
    deleted_files: List[str]
):
    """Menampilkan tabel perbandingan dokumen lama vs baru pada folder yang sama."""
    total_found = len(new_files) + len(modified_files) + len(unchanged_files)

    if RICH_AVAILABLE:
        console.print(Panel(
            f"[bold cyan]Folder Target:[/bold cyan] {target_dir.resolve()}\n"
            f"[bold green]Total Dokumen:[/bold green] {total_found} file  |  "
            f"[bold]Baru:[/bold] [green]{len(new_files)}[/green]  |  "
            f"[bold]Termodifikasi:[/bold] [yellow]{len(modified_files)}[/yellow]  |  "
            f"[bold]Lama (Tetap):[/bold] [cyan]{len(unchanged_files)}[/cyan]  |  "
            f"[bold]Dihapus:[/bold] [red]{len(deleted_files)}[/red]",
            title="[bold white]STATUS PEMBEDIAN DOKUMEN DALAM FOLDER[/bold white]",
            border_style="blue"
        ))

        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Nama Dokumen", style="white bold", ratio=3)
        table.add_column("Status", justify="center", ratio=2)
        table.add_column("Ukuran", justify="right", ratio=1)
        table.add_column("Waktu Modifikasi", ratio=2)
        table.add_column("Aksi yang Diambil", style="italic", ratio=3)

        # 1. Dokumen Baru
        for f in new_files:
            mtime_str = datetime.fromtimestamp(f["mtime"]).strftime("%Y-%m-%d %H:%M")
            size_kb = f"{f['size'] / 1024:.1f} KB"
            table.add_row(
                f"[green]+ {f['name']}[/green]",
                "[bold black on green] BARU [/bold black on green]",
                size_kb,
                mtime_str,
                "[green]Akan diproses & diindeks[/green]"
            )

        # 2. Dokumen Termodifikasi
        for f in modified_files:
            mtime_str = datetime.fromtimestamp(f["mtime"]).strftime("%Y-%m-%d %H:%M")
            size_kb = f"{f['size'] / 1024:.1f} KB"
            table.add_row(
                f"[yellow]* {f['name']}[/yellow]",
                "[bold black on yellow] TERMODIFIKASI [/bold black on yellow]",
                size_kb,
                mtime_str,
                "[yellow]Hapus chunk lama, re-indeks baru[/yellow]"
            )

        # 3. Dokumen Lama (Tetap)
        for f in unchanged_files:
            mtime_str = datetime.fromtimestamp(f["mtime"]).strftime("%Y-%m-%d %H:%M")
            size_kb = f"{f['size'] / 1024:.1f} KB"
            table.add_row(
                f"[cyan]= {f['name']}[/cyan]",
                "[cyan] LAMA (TETAP) [/cyan]",
                size_kb,
                mtime_str,
                "[dim cyan]Dilewati (Skip - sudah terindeks)[/dim cyan]"
            )

        # 4. Dokumen Dihapus
        for rel_p in deleted_files:
            table.add_row(
                f"[red]- {rel_p}[/red]",
                "[bold white on red] DIHAPUS [/bold white on red]",
                "-",
                "-",
                "[red]File fisik hilang dari folder[/red]"
            )

        console.print(table)
    else:
        print("=" * 70)
        print(f"PEMBEDIAN STATUS DOKUMEN: {target_dir.resolve()}")
        print("=" * 70)
        print(f"Total: {total_found} (Baru: {len(new_files)}, Modifikasi: {len(modified_files)}, Tetap: {len(unchanged_files)}, Dihapus: {len(deleted_files)})")
        print("-" * 70)
        for f in new_files:
            print(f"[BARU]          : {f['name']} ({f['size']/1024:.1f} KB) -> Akan diindeks")
        for f in modified_files:
            print(f"[TERMODIFIKASI] : {f['name']} ({f['size']/1024:.1f} KB) -> Re-indeks")
        for f in unchanged_files:
            print(f"[LAMA / TETAP]  : {f['name']} ({f['size']/1024:.1f} KB) -> Dilewati (Skip)")
        for r in deleted_files:
            print(f"[DIHAPUS]       : {r} -> Hilang dari folder")
        print("=" * 70)

def run_incremental_update(
    data_dir: Path,
    dry_run: bool = False,
    force_all: bool = False,
    clean_deleted: bool = False,
    override_classification: Optional[str] = None,
    manifest_path: Optional[Path] = None,
    output_json: bool = False,
    device: Optional[str] = None,
    batch_size: Optional[int] = None,
    from_colab: bool = False,
    colab_url: Optional[str] = None,
    colab_json_path: Optional[Path] = None
):
    manifest_file = manifest_path or MANIFEST_DEFAULT_PATH
    manifest = load_manifest(manifest_file)

    supported_exts = settings.get("ingestion", {}).get(
        "supported_extensions", [".pdf", ".txt", ".md", ".docx", ".doc", ".xlsx", ".xls"]
    )

    if not data_dir.exists():
        logger.error(f"Direktori target {data_dir} tidak ditemukan!")
        sys.exit(1)

    # 1. Pindai dan bedakan dokumen lama vs baru pada folder yang sama
    new_files, modified_files, unchanged_files, deleted_files = scan_folder_and_classify(
        target_dir=data_dir,
        manifest=manifest,
        supported_extensions=supported_exts
    )

    # Jika force_all diaktifkan, anggap semua file lama sebagai termodifikasi
    if force_all:
        modified_files.extend(unchanged_files)
        unchanged_files = []

    # 2. Tampilkan status jika requested atau normal
    if output_json:
        status_payload = {
            "target_dir": str(data_dir.resolve()),
            "summary": {
                "total_files": len(new_files) + len(modified_files) + len(unchanged_files),
                "new": len(new_files),
                "modified": len(modified_files),
                "unchanged": len(unchanged_files),
                "deleted": len(deleted_files)
            },
            "new_files": [f["name"] for f in new_files],
            "modified_files": [f["name"] for f in modified_files],
            "unchanged_files": [f["name"] for f in unchanged_files],
            "deleted_files": deleted_files
        }
        if dry_run:
            print(json.dumps(status_payload, indent=2))
            return
    else:
        display_status_table(data_dir, new_files, modified_files, unchanged_files, deleted_files)

    # Jika hanya mode dry-run / cek status, berhenti di sini
    if dry_run:
        if RICH_AVAILABLE:
            console.print("\n[bold yellow]Mode Dry-Run Aktif:[/bold yellow] Tidak ada data yang diubah di database ChromaDB.\n")
        else:
            print("\nMode Dry-Run: Selesai pemeriksaan status dokumen.\n")
        return

    files_to_process = new_files + modified_files

    if not files_to_process and not (clean_deleted and deleted_files):
        msg = "Seluruh dokumen sudah terindeks dan tidak ada perubahan. Tidak ada dokumen yang perlu diproses."
        if output_json:
            print(json.dumps({"status": "no_changes", "message": msg}))
        else:
            if RICH_AVAILABLE:
                console.print(f"\n[bold green]✓ {msg}[/bold green]\n")
            else:
                print(f"\n{msg}\n")
        return

    # 3. Hubungkan ke ChromaDB
    chroma_cfg = settings.get("chromadb", {})
    vector_store = ChromaVectorStore(
        host=chroma_cfg.get("host", "localhost"),
        port=chroma_cfg.get("port", 8000),
        collection_name=chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")
    )

    if not vector_store.is_healthy():
        err = f"Koneksi ke ChromaDB gagal pada {vector_store.host}:{vector_store.port}!"
        logger.error(err)
        sys.exit(1)

    # 4. Tangani dokumen yang dihapus bila opsi clean_deleted aktif
    if clean_deleted and deleted_files:
        logger.info(f"Membersihkan chunk dari {len(deleted_files)} dokumen yang dihapus dari folder...")
        for del_rel in deleted_files:
            del_name = Path(del_rel).name
            vector_store.delete_by_source(del_name)
            manifest["documents"].pop(del_rel, None)
        save_manifest(manifest, manifest_file)

    # 5. Inisialisasi pipeline pemrosesan (atau ambil vektor pra-komputasi dari VM Google Colab)
    colab_vectors_by_file: Dict[str, Dict[str, List[Any]]] = {}
    if from_colab or colab_url or colab_json_path:
        from scripts.pull_colab_index import fetch_colab_vectors_json, load_colab_vectors_by_filename
        resolved_json = colab_json_path or fetch_colab_vectors_json(colab_url=colab_url)
        colab_vectors_by_file = load_colab_vectors_by_filename(resolved_json)
        logger.info(
            f"Mode Colab Aktif: Memuat {len(colab_vectors_by_file)} dokumen dari '{resolved_json.name}'. "
            "HANYA dokumen baru/berubah yang akan disimpan ke ChromaDB lokal!"
        )

    loader = DocumentLoader(raw_dir=str(data_dir))
    if override_classification:
        loader.default_classification = override_classification.upper()

    chunker = None
    engine = None

    total_chunks_added = 0
    start_time = time.time()

    # 6. Eksekusi HANYA dokumen yang baru atau termodifikasi
    for idx, f_info in enumerate(files_to_process, start=1):
        f_path: Path = f_info["path"]
        f_name: str = f_info["name"]
        rel_path: str = f_info["rel_path"]
        is_mod = f_info in modified_files

        action_label = "Memperbarui (Update)" if is_mod else "Memproses Dokumen Baru"
        logger.info(f"[{idx}/{len(files_to_process)}] {action_label}: {f_name}")

        # Jika dokumen termodifikasi, hapus chunk versi lama terlebih dahulu
        if is_mod:
            deleted_old = vector_store.delete_by_source(f_name)
            if deleted_old > 0:
                logger.info(f"  -> Dihapus {deleted_old} chunk versi lama dari database.")

        # Tentukan klasifikasi keamanan
        classification = loader.default_classification
        if data_dir.name.upper() in ["PUBLIC", "INTERNAL", "RESTRICTED", "CONFIDENTIAL", "SECRET"]:
            classification = data_dir.name.upper()
        try:
            rel_parts = f_path.relative_to(data_dir).parts
            if len(rel_parts) > 1:
                potential_class = rel_parts[0].upper()
                if potential_class in ["PUBLIC", "INTERNAL", "RESTRICTED", "CONFIDENTIAL", "SECRET"]:
                    classification = potential_class
        except ValueError:
            pass

        # JALUR A: Jika vektor dokumen baru ini sudah tersedia dari hasil GPU Google Colab
        if f_name in colab_vectors_by_file:
            c_bundle = colab_vectors_by_file[f_name]
            c_ids = c_bundle["ids"]
            c_docs = c_bundle["documents"]
            c_metas = c_bundle["metadatas"]
            c_embs = c_bundle["embeddings"]
            b_sz = 200
            for b_i in range(0, len(c_ids), b_sz):
                vector_store.collection.upsert(
                    ids=c_ids[b_i : b_i + b_sz],
                    documents=c_docs[b_i : b_i + b_sz],
                    metadatas=c_metas[b_i : b_i + b_sz],
                    embeddings=c_embs[b_i : b_i + b_sz]
                )
            stored = len(c_ids)
            total_chunks_added += stored

            manifest["documents"][rel_path] = {
                "file_name": f_name,
                "file_path": str(f_path.resolve()),
                "sha256": f_info["hash"],
                "size_bytes": f_info["size"],
                "mtime": f_info["mtime"],
                "classification": classification,
                "indexed_at": datetime.now().isoformat(),
                "chunks_count": stored,
                "status": "indexed_via_colab"
            }
            save_manifest(manifest, manifest_file)
            logger.info(f"  -> [COLAB GPU] Sukses menyimpan {stored} chunks baru untuk '{f_name}' ke ChromaDB lokal.")
            continue

        if from_colab:
            logger.warning(
                f"  -> Dokumen baru '{f_name}' belum ditemukan di dalam bundel JSON Colab. "
                "Jalankan `python scripts/sync_colab_notebook.py` lalu eksekusi notebook di Colab, "
                "atau jalankan tanpa `--from-colab` untuk memproses secara lokal."
            )
            continue

        # JALUR B: Pemrosesan lokal (jika tidak menggunakan --from-colab)
        if chunker is None:
            chunker = TextChunker()
        if engine is None:
            engine = EmbeddingEngine(device=device, batch_size=batch_size)

        # Ekstraksi halaman dokumen
        pages = loader.load_file(f_path, classification=classification)
        if not pages:
            logger.warning(f"  -> Tidak ada teks yang dapat diekstrak dari {f_name}.")
            continue

        # Chunking structure-aware 256 token
        chunks = chunker.chunk_pages(pages)
        if not chunks:
            logger.warning(f"  -> Tidak ada chunk valid yang dihasilkan dari {f_name}.")
            continue

        # Kalkulasi vector embedding
        texts_to_embed = [c.text for c in chunks]
        embeddings = engine.embed_documents(texts_to_embed)

        # Simpan ke ChromaDB
        stored = vector_store.upsert_chunks(chunks, embeddings)
        total_chunks_added += stored

        # Catat ke manifest sidik jari
        manifest["documents"][rel_path] = {
            "file_name": f_name,
            "file_path": str(f_path.resolve()),
            "sha256": f_info["hash"],
            "size_bytes": f_info["size"],
            "mtime": f_info["mtime"],
            "classification": classification,
            "indexed_at": datetime.now().isoformat(),
            "chunks_count": len(chunks),
            "status": "indexed"
        }

        # Simpan manifest secara bertahap agar aman
        save_manifest(manifest, manifest_file)
        logger.info(f"  -> Sukses mengindeks {len(chunks)} chunks baru untuk '{f_name}'.")

        # Bebaskan VRAM/cache XPU setelah setiap dokumen selesai
        if hasattr(engine, "_empty_accelerator_cache"):
            engine._empty_accelerator_cache()

    elapsed = time.time() - start_time
    stats = vector_store.get_stats()

    if output_json:
        res_payload = {
            "status": "success",
            "files_processed": len(files_to_process),
            "total_chunks_added": total_chunks_added,
            "database_total_vectors": stats.get("total_vectors"),
            "elapsed_seconds": round(elapsed, 2)
        }
        print(json.dumps(res_payload, indent=2))
    else:
        if RICH_AVAILABLE:
            summary_panel = (
                f"[bold green]Pembaruan Inkremental Selesai![/bold green]\n\n"
                f"• Dokumen Diproses     : [bold cyan]{len(files_to_process)} file[/bold cyan] (Baru/Termodifikasi)\n"
                f"• Dokumen Lama Dilewati: [bold white]{len(unchanged_files)} file[/bold white] (Tanpa beban CPU)\n"
                f"• Total Chunks Baru    : [bold green]{total_chunks_added} chunks[/bold green]\n"
                f"• Total Vektor di DB   : [bold yellow]{stats.get('total_vectors')} data[/bold yellow]\n"
                f"• Waktu Pemrosesan     : {elapsed:.2f} detik"
            )
            console.print(Panel(summary_panel, title="[bold white]LAPORAN UPDATE DATA CHROMA[/bold white]", border_style="green"))
        else:
            print("\n" + "=" * 55)
            print("       LAPORAN PEMBARUAN INKREMENTAL CHROMA")
            print("=" * 55)
            print(f"Dokumen Diproses      : {len(files_to_process)} file (Baru/Termodifikasi)")
            print(f"Dokumen Lama Dilewati : {len(unchanged_files)} file (Skip)")
            print(f"Total Chunks Baru     : {total_chunks_added}")
            print(f"Total Vektor di DB    : {stats.get('total_vectors')}")
            print(f"Waktu Eksekusi        : {elapsed:.2f} detik")
            print("=" * 55 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script pembaruan data inkremental ChromaDB (hanya memproses dokumen baru/berubah)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  python scripts/update_index.py                    # Cek folder data/raw dan update file baru/berubah
  python scripts/update_index.py --status           # Hanya cek perbandingan file lama vs baru (tanpa ubah DB)
  python scripts/update_index.py --clean-deleted    # Hapus juga data dari dokumen yang sudah dihapus
  python scripts/update_index.py --force            # Paksa proses ulang semua dokumen
  python scripts/update_index.py --data-dir ./data/raw/SECRET  # Update folder tertentu
        """
    )
    parser.add_argument("--data-dir", type=str, default="./data/raw", help="Folder dokumen yang dipindai (default: ./data/raw)")
    parser.add_argument("--status", "--dry-run", dest="dry_run", action="store_true", help="Cek status dokumen lama vs baru tanpa indexing")
    parser.add_argument("--force", action="store_true", help="Paksa indeks ulang semua dokumen")
    parser.add_argument("--clean-deleted", action="store_true", help="Hapus chunk dari dokumen yang sudah tidak ada di folder")
    parser.add_argument("--classification", type=str, default=None, help="Paksa klasifikasi RBAC tertentu")
    parser.add_argument("--device", type=str, default=None, help="Device embedding: xpu atau cpu (default dari config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size embedding (default: 16 untuk mencegah XPU OOM)")
    parser.add_argument("--json", action="store_true", help="Output laporan dalam format JSON")
    parser.add_argument(
        "--from-colab",
        action="store_true",
        help="Unduh/ambil hasil indexing dari VM Google Colab dan simpan HANYA dokumen baru ke ChromaDB lokal"
    )
    parser.add_argument(
        "--colab-url",
        type=str,
        default=None,
        help="URL download langsung file colab_indexed_vectors.json.xz dari output Sel 6 VM Google Colab"
    )

    args = parser.parse_args()

    run_incremental_update(
        data_dir=Path(args.data_dir),
        dry_run=args.dry_run,
        force_all=args.force,
        clean_deleted=args.clean_deleted,
        override_classification=args.classification,
        output_json=args.json,
        device=args.device,
        batch_size=args.batch_size,
        from_colab=args.from_colab or bool(args.colab_url),
        colab_url=args.colab_url
    )

