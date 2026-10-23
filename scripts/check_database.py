#!/usr/bin/env python3
"""Script untuk Memeriksa dan Menampilkan Isi Database ChromaDB (Project Emas PUSDIKZI)

Fitur:
- Cek konektivitas server & status container ChromaDB Docker
- Tampilkan statistik ringkasan koleksi (total vektor, distribusi klasifikasi, daftar dokumen)
- Tampilkan cuplikan dokumen/chunk dengan metadata terstruktur
- Filter berdasarkan file sumber, klasifikasi keamanan (RBAC), tipe file, atau kata kunci pencarian
- Dukungan format tampilan tabel Rich (terminal interaktif) dan format JSON
"""

import sys
import os
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config.settings import settings
from src.vector_store.chroma_db import ChromaVectorStore

# Cek ketersediaan library rich untuk UI terminal yang menarik
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

def get_chroma_store(collection_name: Optional[str] = None) -> Optional[ChromaVectorStore]:
    """Inisialisasi klien ChromaVectorStore sesuai konfigurasi."""
    chroma_cfg = settings.get("chromadb", {})
    host = chroma_cfg.get("host", "localhost")
    port = chroma_cfg.get("port", 8000)
    col_name = collection_name or chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")

    try:
        store = ChromaVectorStore(host=host, port=port, collection_name=col_name)
        if not store.is_healthy():
            return None
        return store
    except Exception:
        return None

def inspect_database(
    limit: int = 5,
    show_all: bool = False,
    file_filter: Optional[str] = None,
    access_filter: Optional[str] = None,
    type_filter: Optional[str] = None,
    search_keyword: Optional[str] = None,
    target_id: Optional[str] = None,
    show_full_text: bool = False,
    output_json: bool = False,
    list_collections_only: bool = False,
    collection_name: Optional[str] = None
):
    chroma_cfg = settings.get("chromadb", {})
    host = chroma_cfg.get("host", "localhost")
    port = chroma_cfg.get("port", 8000)
    col_name = collection_name or chroma_cfg.get("collection_name", "pusdikzi_knowledge_base")

    store = get_chroma_store(col_name)
    if not store:
        err_msg = (
            f"[ERROR] Gagal terhubung ke ChromaDB pada http://{host}:{port}!\n"
            "Pastikan container Docker aktif dengan perintah: docker compose up -d"
        )
        if output_json:
            print(json.dumps({"status": "error", "message": err_msg}, indent=2))
        else:
            if RICH_AVAILABLE:
                console.print(Panel(err_msg, title="[bold red]Koneksi Gagal[/bold red]", border_style="red"))
            else:
                print(f"\n{err_msg}\n")
        sys.exit(1)

    # 1. Tampilkan daftar semua koleksi jika diminta
    if list_collections_only:
        all_cols = store.client.list_collections()
        col_list = [{"name": c.name, "count": c.count(), "metadata": c.metadata} for c in all_cols]
        if output_json:
            print(json.dumps({"collections": col_list}, indent=2))
            return

        if RICH_AVAILABLE:
            table = Table(title="Daftar Koleksi di Server ChromaDB", box=box.ROUNDED)
            table.add_column("Nama Koleksi", style="cyan bold")
            table.add_column("Total Vektor", justify="right", style="green")
            table.add_column("Metrik Jarak", style="magenta")
            for c in col_list:
                dist = c["metadata"].get("hnsw:space", "default") if c["metadata"] else "default"
                table.add_row(c["name"], str(c["count"]), dist)
            console.print(table)
        else:
            print("\nDaftar Koleksi di Server ChromaDB:")
            for c in col_list:
                print(f" - {c['name']}: {c['count']} vektor")
        return

    collection = store.collection
    total_count = collection.count()

    # 2. Ambil data dengan filter metadata jika ditentukan
    where_conditions = {}
    if access_filter:
        where_conditions["klasifikasi"] = access_filter.lower()
    if type_filter:
        where_conditions["tipe"] = type_filter.lower().replace(".", "")
    if file_filter:
        where_conditions["sumber"] = file_filter

    fetch_limit = total_count if (show_all or total_count < 100) else max(limit, 100)
    
    query_kwargs = {
        "limit": fetch_limit if fetch_limit > 0 else 10,
        "include": ["documents", "metadatas"]
    }
    if where_conditions:
        query_kwargs["where"] = where_conditions

    try:
        raw_data = collection.get(**query_kwargs) if total_count > 0 else {"ids": [], "documents": [], "metadatas": []}
    except Exception as e:
        # Fallback jika filter klasifikasi di DB tersimpan huruf kapital
        if access_filter and "klasifikasi" in where_conditions:
            where_conditions["klasifikasi"] = access_filter.upper()
            query_kwargs["where"] = where_conditions
            try:
                raw_data = collection.get(**query_kwargs)
            except Exception:
                raw_data = {"ids": [], "documents": [], "metadatas": []}
        else:
            raw_data = {"ids": [], "documents": [], "metadatas": []}

    ids = raw_data.get("ids", [])
    docs = raw_data.get("documents", [])
    metas = raw_data.get("metadatas", [])

    # Filter manual untuk kata kunci dan target ID bila ada
    filtered_items = []
    for i in range(len(ids)):
        c_id = ids[i]
        c_doc = docs[i] if i < len(docs) else ""
        c_meta = metas[i] if i < len(metas) else {}

        if target_id and target_id not in c_id:
            continue
        if search_keyword and search_keyword.lower() not in c_doc.lower():
            continue
        filtered_items.append((c_id, c_doc, c_meta))

    # Agregasi statistik koleksi
    sources_count: Dict[str, int] = {}
    classes_count: Dict[str, int] = {}
    types_count: Dict[str, int] = {}

    for _, _, meta in filtered_items if where_conditions else list(zip(ids, docs, metas)):
        s = meta.get("sumber", meta.get("file_name", "Unknown"))
        c = meta.get("klasifikasi", meta.get("classification", "UNKNOWN")).upper()
        t = meta.get("tipe", meta.get("file_type", "unknown"))

        sources_count[s] = sources_count.get(s, 0) + 1
        classes_count[c] = classes_count.get(c, 0) + 1
        types_count[t] = types_count.get(t, 0) + 1

    # Format JSON
    if output_json:
        display_items = filtered_items if show_all else filtered_items[:limit]
        out_payload = {
            "status": "healthy",
            "server": f"http://{host}:{port}",
            "collection": col_name,
            "total_vectors_in_collection": total_count,
            "matched_vectors": len(filtered_items),
            "statistics": {
                "classifications": classes_count,
                "file_types": types_count,
                "sources": sources_count
            },
            "chunks": [
                {
                    "id": item[0],
                    "metadata": item[2],
                    "content": item[1]
                }
                for item in display_items
            ]
        }
        print(json.dumps(out_payload, indent=2))
        return

    # Tampilan Terminal Berwarna (Rich Console)
    if RICH_AVAILABLE:
        header_text = (
            f"[bold cyan]Server:[/bold cyan] http://{host}:{port}  |  "
            f"[bold cyan]Koleksi:[/bold cyan] {col_name}  |  "
            f"[bold green]Status:[/bold green] Online (Healthy)"
        )
        console.print(Panel(header_text, title="[bold white]CHROMA VECTOR DATABASE INSPECTOR[/bold white]", border_style="blue"))

        # Ringkasan Statistik
        summary_table = Table(title="Ringkasan Statistik Koleksi", box=box.SIMPLE_HEAVY)
        summary_table.add_column("Metrik", style="bold yellow")
        summary_table.add_column("Nilai / Rincian", style="white")

        summary_table.add_row("Total Vektor di DB", f"[bold green]{total_count}[/bold green] chunks tersimpan")
        summary_table.add_row("Total Cocok Filter", f"[bold cyan]{len(filtered_items)}[/bold cyan] chunks")
        summary_table.add_row("Distribusi Klasifikasi", ", ".join(f"[bold]{k}[/bold]: {v}" for k, v in classes_count.items()) or "-")
        summary_table.add_row("Distribusi Format File", ", ".join(f"[bold]{k}[/bold]: {v}" for k, v in types_count.items()) or "-")
        summary_table.add_row("Dokumen Terdaftar", f"{len(sources_count)} file sumber")
        console.print(summary_table)

        if total_count == 0:
            console.print("\n[yellow]Pemberitahuan:[/yellow] Database masih kosong. Jalankan [cyan]python scripts/build_index.py[/cyan] untuk mengisi data.\n")
            return

        if not filtered_items:
            console.print("\n[yellow]Tidak ditemukan data yang cocok dengan kriteria filter yang diberikan.[/yellow]\n")
            return

        # Tampilkan Detail Chunks
        display_items = filtered_items if show_all else filtered_items[:limit]
        console.print(f"\n[bold green]Menampilkan {len(display_items)} dari {len(filtered_items)} chunks:[/bold green]")

        for idx, (chunk_id, doc_text, meta) in enumerate(display_items, start=1):
            sumber = meta.get("sumber", meta.get("file_name", "Unknown"))
            tipe = meta.get("tipe", meta.get("file_type", "-"))
            halaman = meta.get("halaman", meta.get("page", 1))
            klasifikasi = meta.get("klasifikasi", meta.get("classification", "INTERNAL")).upper()
            bab = meta.get("judul_bab", "-")
            tokens = meta.get("token_count", "-")

            badge_color = "red" if "SECRET" in klasifikasi or "RAHASIA" in klasifikasi else "green"

            chunk_meta_info = (
                f"[bold]ID:[/bold] [yellow]{chunk_id}[/yellow] | "
                f"[bold]Sumber:[/bold] [cyan]{sumber}[/cyan] ({tipe}) | "
                f"[bold]Hal:[/bold] {halaman} | "
                f"[bold]Akses:[/bold] [{badge_color}]{klasifikasi}[/{badge_color}] | "
                f"[bold]Bab:[/bold] {bab} | "
                f"[bold]Tokens:[/bold] {tokens}"
            )

            # Potong teks jika tidak meminta full text
            content_display = doc_text.strip()
            if not show_full_text and len(content_display) > 250:
                content_display = content_display[:250] + " ... [dim](gunakan --full untuk teks lengkap)[/dim]"

            console.print(Panel(content_display, title=f"Chunk #{idx} - {chunk_meta_info}", border_style="dim"))

        if len(filtered_items) > len(display_items):
            console.print(f"\n[dim]* Masih ada {len(filtered_items) - len(display_items)} data lainnya. Gunakan opsi --limit N atau --all untuk melihat seluruhnya.[/dim]\n")

    else:
        # Fallback Plain ASCII Output
        print("=" * 65)
        print("           STATUS DAN INSPEKSI DATABASE VEKTOR (ChromaDB)       ")
        print("=" * 65)
        print(f"Server          : http://{host}:{port}")
        print(f"Koleksi Target  : {col_name}")
        print(f"Total Vektor    : {total_count} data tersimpan")
        print(f"Klasifikasi     : {dict(classes_count)}")
        print(f"Format File     : {dict(types_count)}")
        print(f"File Sumber     : {list(sources_count.keys())}")
        print("-" * 65)

        if total_count == 0:
            print("\nDatabase masih kosong. Jalankan: python scripts/build_index.py\n")
            return

        display_items = filtered_items if show_all else filtered_items[:limit]
        print(f"\nCUPLIKAN {len(display_items)} DATA CHUNKS:")
        for idx, (chunk_id, doc_text, meta) in enumerate(display_items, start=1):
            sumber = meta.get("sumber", meta.get("file_name", "Unknown"))
            tipe = meta.get("tipe", meta.get("file_type", "-"))
            halaman = meta.get("halaman", meta.get("page", 1))
            klasifikasi = meta.get("klasifikasi", meta.get("classification", "INTERNAL"))
            snippet = doc_text.replace("\n", " ").strip()
            if not show_full_text and len(snippet) > 150:
                snippet = snippet[:150] + "..."

            print(f"\n[{idx}] ID       : {chunk_id}")
            print(f"    Sumber   : {sumber} [{tipe}] (Halaman {halaman})")
            print(f"    Akses    : {klasifikasi}")
            print(f"    Teks     : \"{snippet}\"")
        print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Script untuk menampilkan dan memeriksa isi database vektor ChromaDB PUSDIKZI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  python scripts/check_database.py                    # Menampilkan ringkasan dan 5 data teratas
  python scripts/check_database.py --limit 10          # Menampilkan 10 data teratas
  python scripts/check_database.py --all               # Menampilkan seluruh data
  python scripts/check_database.py --full              # Menampilkan teks lengkap tanpa dipotong
  python scripts/check_database.py --file "sop.md"     # Filter dokumen berdasarkan nama file sumber
  python scripts/check_database.py --access SECRET     # Filter data klasifikasi SECRET
  python scripts/check_database.py --type pdf          # Filter data dokumen PDF
  python scripts/check_database.py --search "zeni"     # Mencari teks yang mengandung kata 'zeni'
  python scripts/check_database.py --collections       # Menampilkan daftar seluruh koleksi di database
  python scripts/check_database.py --json              # Output data dalam format JSON
        """
    )

    parser.add_argument("--limit", type=int, default=5, help="Jumlah chunk yang ingin ditampilkan (default: 5)")
    parser.add_argument("--all", action="store_true", help="Tampilkan seluruh data tanpa batasan limit")
    parser.add_argument("--full", action="store_true", help="Tampilkan teks penuh setiap chunk tanpa dipotong")
    parser.add_argument("--file", type=str, default=None, help="Filter berdasarkan nama file sumber (e.g. sop.md)")
    parser.add_argument("--access", type=str, default=None, help="Filter berdasarkan klasifikasi akses (e.g. SECRET, INTERNAL)")
    parser.add_argument("--type", type=str, default=None, help="Filter berdasarkan tipe file (e.g. pdf, docx, xlsx)")
    parser.add_argument("--search", type=str, default=None, help="Cari kata kunci dalam teks dokumen")
    parser.add_argument("--id", type=str, default=None, help="Cari chunk spesifik berdasarkan ID")
    parser.add_argument("--collections", action="store_true", help="Tampilkan seluruh koleksi yang tersedia di server")
    parser.add_argument("--collection-name", type=str, default=None, help="Target nama koleksi spesifik")
    parser.add_argument("--json", action="store_true", help="Tampilkan hasil dalam format JSON")

    args = parser.parse_args()

    inspect_database(
        limit=args.limit,
        show_all=args.all,
        file_filter=args.file,
        access_filter=args.access,
        type_filter=args.type,
        search_keyword=args.search,
        target_id=args.id,
        show_full_text=args.full,
        output_json=args.json,
        list_collections_only=args.collections,
        collection_name=args.collection_name
    )
