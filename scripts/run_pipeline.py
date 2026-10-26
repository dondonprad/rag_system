#!/usr/bin/env python3
"""Script Pipeline Terpadu PUSDIKZI RAG (Colab GPU to Local ChromaDB)

Pipeline Otomatis:
1. Sinkronisasi dokumen baru/berubah ke `notebook/build_index.ipynb` (sync_colab_notebook)
2. Membuka notebook & memantau eksekusi runtime Colab di VS Code (watcher)
3. Mengambil file JSON vektor / LZMA hasil indexing Colab (pull_colab_index)
4. Memperbarui database ChromaDB lokal & manifest dokumen (update_index)

Contoh Penggunaan:
  # 1. Alur standar interaktif (Sync -> Watcher Colab VS Code -> Pull -> Update):
  ./venv/bin/python scripts/run_pipeline.py

  # 2. Proses seluruh dokumen termasuk dokumen lama:
  ./venv/bin/python scripts/run_pipeline.py --all

  # 3. Langsung ambil dari URL tunnel Cloudflare (lewati tahap sync):
  ./venv/bin/python scripts/run_pipeline.py --colab-url https://xxxx.trycloudflare.com/colab_indexed_vectors.json.xz

  # 4. Hanya pull dan update dari notebook yang sudah selesai dijalankan:
  ./venv/bin/python scripts/run_pipeline.py --pull-only
"""

import os
import re
import sys
import time
import json
import select
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Pastikan root workspace ada di sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Cek ketersediaan library rich untuk tampilan terminal interaktif
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich import box
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None

from scripts.sync_colab_notebook import (
    sync_secret_to_notebook,
    classify_documents,
    load_manifest,
    MANIFEST_DEFAULT_PATH
)
from scripts.pull_colab_index import (
    fetch_colab_vectors_json,
    inspect_notebook_for_colab_data,
    DEFAULT_NOTEBOOK_PATH,
    DEFAULT_JSON_PATH,
    SECRET_JSON_PATH
)
from scripts.update_index import run_incremental_update

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s"
)
logger = logging.getLogger("run_pipeline")


def print_banner():
    """Menampilkan header informasi pipeline."""
    if RICH_AVAILABLE:
        banner_text = (
            "[bold cyan]PUSDIKZI RAG PIPELINE EXECUTOR[/bold cyan]\n"
            "[dim]Siklus Penuh: Sync Dokumen ➡️ Colab GPU Indexing ➡️ Pull Vektor ➡️ Local ChromaDB[/dim]"
        )
        console.print(Panel(banner_text, box=box.ROUNDED, style="cyan"))
    else:
        print("\n" + "=" * 76)
        print("   PUSDIKZI RAG PIPELINE EXECUTOR (Colab GPU to Local ChromaDB)")
        print("=" * 76)


def clear_notebook_cell_outputs(notebook_path: Path):
    """Membersihkan output pada sel terakhir (Cell 6) di build_index.ipynb

    Tujuan: Menjamin watcher hanya menangkap output dari eksekusi Colab yang BARU,
    bukan output lama dari sesi sebelumnya.
    """
    if not notebook_path.exists():
        return

    try:
        nb_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        modified = False
        for cell in nb_data.get("cells", []):
            if cell.get("cell_type") == "code":
                # Jika sel mengandung logika ekspor vektor atau cloudflared
                src = "".join(cell.get("source", []))
                if "colab_indexed_vectors" in src or "cloudflared" in src:
                    if cell.get("outputs"):
                        cell["outputs"] = []
                        cell["execution_count"] = None
                        modified = True

        if modified:
            temp_path = notebook_path.with_suffix(".ipynb.tmp")
            temp_path.write_text(json.dumps(nb_data, indent=1, ensure_ascii=False), encoding="utf-8")
            temp_path.replace(notebook_path)
            logger.debug(f"Output sel ekspor di {notebook_path.name} telah direset.")
    except Exception as e:
        logger.warning(f"Gagal mereset output notebook: {e}")


def check_for_colab_errors(notebook_path: Path) -> Optional[str]:
    """Mengecek apakah ada error traceback di dalam sel notebook."""
    try:
        nb_data = json.loads(notebook_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb_data.get("cells", [])):
            if cell.get("cell_type") == "code":
                for out in cell.get("outputs", []):
                    if out.get("output_type") == "error":
                        ename = out.get("ename", "Error")
                        evalue = out.get("evalue", "")
                        return f"Error pada Sel {idx} ({ename}): {evalue}"
    except Exception:
        pass
    return None


def check_stdin_input() -> Optional[str]:
    """Mengecek input non-blocking dari terminal user (Enter, URL Cloudflare, dll)."""
    try:
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            return line.strip()
    except Exception:
        pass
    return None


def wait_for_colab_execution(
    notebook_path: Path,
    timeout_seconds: int = 1800,
    poll_interval: int = 2
) -> Dict[str, Any]:
    """Memantau (watch) file notebook hingga eksekusi Sel 6 di Colab selesai.

    Mendeteksi:
    1. Payload LZMA Base64 di display_data metadata
    2. URL Tunnel Cloudflare di stdout
    3. Status selesai Sel 6 (execution_count atau penanda stdout)
    4. Error exception pada sel mana pun
    5. Input keyboard [ENTER] atau URL yang di-paste manual oleh user di terminal
    """
    start_time = time.time()
    last_msg = ""
    prompt_shown_time = 0

    if RICH_AVAILABLE:
        with Progress(
            SpinnerColumn(spinner_name="dots"),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            task = progress.add_task(
                "[yellow]Menunggu eksekusi Colab... [dim](Klik 'Run All' & Tekan Ctrl+S setelah selesai, atau Tekan [ENTER] di sini)[/dim]",
                total=None
            )

            while time.time() - start_time < timeout_seconds:
                # 1. Cek input interaktif dari user di terminal
                user_input = check_stdin_input()
                if user_input is not None:
                    # Jika user paste URL cloudflare langsung
                    if user_input.startswith("http://") or user_input.startswith("https://"):
                        progress.stop()
                        console.print(f"[bold green]✔ Menerima URL download langsung dari terminal:[/bold green] {user_input}")
                        return {"tunnel_url": user_input, "embedded_b64": None, "cell_completed": True}

                    # Jika user menekan ENTER untuk memicu pengecekan manual
                    colab_data = inspect_notebook_for_colab_data(notebook_path)
                    if colab_data.get("embedded_b64") or colab_data.get("tunnel_url") or colab_data.get("cell_completed"):
                        progress.stop()
                        console.print("[bold green]✔ Deteksi manual berhasil dipicu oleh pengguna![/bold green]")
                        return colab_data
                    else:
                        console.print(
                            "\n[bold yellow]⚠️ Peringatan:[/bold yellow] Output di file notebook disk masih kosong!\n"
                            "[dim]👉 Pastikan Anda telah menekan [bold cyan]Ctrl + S[/bold cyan] pada tab notebook di VS Code agar output tersimpan ke disk.[/dim]\n"
                        )

                # 2. Cek jika ada error fatal di sel
                err = check_for_colab_errors(notebook_path)
                if err:
                    progress.stop()
                    console.print(f"[bold red]❌ Terdeteksi kesalahan eksekusi di Colab:[/bold red] {err}")
                    raise RuntimeError(err)

                # 3. Cek output vektor otomatis
                colab_data = inspect_notebook_for_colab_data(notebook_path)
                if colab_data.get("embedded_b64") or colab_data.get("tunnel_url"):
                    progress.stop()
                    console.print("[bold green]✅ Eksekusi Colab selesai terdeteksi otomatis![/bold green]")
                    return colab_data

                # Jika sel ekspor selesai dieksekusi (execution_count terisi / ada ringkasan)
                if colab_data.get("cell_completed"):
                    progress.stop()
                    console.print("[bold green]✅ Sel ekspor Colab terdeteksi selesai dijalankan![/bold green]")
                    return colab_data

                elapsed = int(time.time() - start_time)
                progress.update(
                    task,
                    description=f"[yellow]Menunggu Colab selesai [dim](Klik 'Run All' & Tekan Ctrl+S, atau Tekan [ENTER] jika sudah)[/dim] [{elapsed}s]"
                )
                time.sleep(poll_interval)
    else:
        print("\n⏳ Menunggu eksekusi notebook di Google Colab selesai...")
        print("   (Buka notebook/build_index.ipynb di VS Code, klik 'Run All', lalu tekan Ctrl+S)")
        print("   👉 Anda juga dapat menekan [ENTER] di terminal ini kapan saja jika eksekusi sudah selesai!\n")

        while time.time() - start_time < timeout_seconds:
            user_input = check_stdin_input()
            if user_input is not None:
                if user_input.startswith("http://") or user_input.startswith("https://"):
                    print(f"\n✔ Menerima URL download langsung dari terminal: {user_input}")
                    return {"tunnel_url": user_input, "embedded_b64": None, "cell_completed": True}

                colab_data = inspect_notebook_for_colab_data(notebook_path)
                if colab_data.get("embedded_b64") or colab_data.get("tunnel_url") or colab_data.get("cell_completed"):
                    print("\n✔ Deteksi manual berhasil dipicu oleh pengguna!")
                    return colab_data
                else:
                    print("\n⚠️ Output di disk masih kosong. Pastikan Anda sudah menekan 'Ctrl + S' di VS Code!")

            err = check_for_colab_errors(notebook_path)
            if err:
                print(f"\n❌ Terdeteksi kesalahan eksekusi di Colab: {err}")
                raise RuntimeError(err)

            colab_data = inspect_notebook_for_colab_data(notebook_path)
            if colab_data.get("embedded_b64") or colab_data.get("tunnel_url") or colab_data.get("cell_completed"):
                print("\n✅ Eksekusi Colab selesai terdeteksi!")
                return colab_data

            elapsed = int(time.time() - start_time)
            msg = f"\r   Menunggu respon eksekusi Colab... (Tekan [ENTER] jika sudah selesai) [{elapsed}s]"
            if msg != last_msg:
                sys.stdout.write(msg)
                sys.stdout.flush()
                last_msg = msg
            time.sleep(poll_interval)

    raise TimeoutError(
        f"Waktu tunggu eksekusi Colab melebihi batas waktu ({timeout_seconds} detik). "
        "Pastikan notebook sudah dihubungkan ke runtime Colab, klik 'Run All', lalu tekan Ctrl+S."
    )


def open_notebook_in_editor(notebook_path: Path):
    """Membuka file notebook di editor VS Code."""
    code_bin = shutil.which("code")
    if code_bin and notebook_path.exists():
        try:
            subprocess.run([code_bin, str(notebook_path.resolve())], check=False)
            logger.info(f"File notebook dibuka di VS Code: {notebook_path.name}")
        except Exception as e:
            logger.debug(f"Gagal membuka notebook via code CLI: {e}")


def run_pipeline(
    data_dir: Path = BASE_DIR / "data" / "raw" / "SECRET",
    notebook_path: Path = DEFAULT_NOTEBOOK_PATH,
    manifest_path: Path = MANIFEST_DEFAULT_PATH,
    colab_url: Optional[str] = None,
    colab_json_path: Optional[Path] = None,
    force_all: bool = False,
    skip_sync: bool = False,
    clean_deleted: bool = False,
    timeout: int = 1800,
    override_classification: Optional[str] = None
):
    """Menjalankan siklus penuh build index RAG PUSDIKZI."""
    overall_start_time = time.time()
    print_banner()

    # =========================================================================
    # TAHAP 1: Sinkronisasi Dokumen Inkremental ke Notebook
    # =========================================================================
    if not skip_sync and not colab_url and not colab_json_path:
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]▶ LANGKAH 1/4: Sinkronisasi Dokumen ke Notebook[/bold cyan]")
        else:
            print("\n" + "-" * 76)
            print("▶ LANGKAH 1/4: Sinkronisasi Dokumen ke Notebook")
            print("-" * 76)

        # Cek status dokumen
        manifest = load_manifest(manifest_path)
        from src.ingestion.loaders import DocumentLoader
        loader = DocumentLoader(raw_dir=str(data_dir))
        new_files, modified_files, unchanged_files = classify_documents(
            data_dir=data_dir,
            manifest=manifest,
            supported_extensions=loader.supported_extensions
        )

        total_pending = len(new_files) + len(modified_files)
        if total_pending == 0 and not force_all:
            if RICH_AVAILABLE:
                console.print(
                    Panel(
                        f"[bold green]✨ Semua {len(unchanged_files)} dokumen sudah terindeks secara up-to-date![/bold green]\n"
                        f"[dim]Tidak ada file baru atau termodifikasi di {data_dir.resolve()}.\n"
                        f"Gunakan opsi [bold cyan]--all[/bold cyan] jika ingin memproses ulang seluruh dokumen.[/dim]",
                        box=box.ROUNDED,
                        style="green"
                    )
                )
            else:
                print(f"\n✨ Semua {len(unchanged_files)} dokumen sudah terindeks secara up-to-date!")
                print(f"   Tidak ada file baru/termodifikasi di {data_dir.resolve()}.")
                print("   Gunakan opsi --all jika ingin memproses ulang seluruh dokumen.\n")
            return

        # Bersihkan output lama di Cell 6 agar watcher akurat
        clear_notebook_cell_outputs(notebook_path)

        # Jalankan sinkronisasi dokumen ke notebook
        sync_secret_to_notebook(
            data_dir=data_dir,
            notebook_path=notebook_path,
            manifest_path=manifest_path,
            classification=override_classification,
            force_all=force_all,
            dry_run=False,
            mark_indexed=False  # Baru ditandai setelah berhasil masuk ChromaDB di Langkah 4
        )
    else:
        logger.info("Melewati Langkah 1 (Sinkronisasi dilewati karena opsi pengguna/URL eksplisit).")

    # =========================================================================
    # TAHAP 2: Eksekusi Notebook di Runtime Colab (Watcher)
    # =========================================================================
    if not colab_url and not colab_json_path:
        if RICH_AVAILABLE:
            console.print("\n[bold cyan]▶ LANGKAH 2/4: Eksekusi GPU di Google Colab Runtime[/bold cyan]")
            guide_panel = (
                "[bold yellow]TINDAKAN DIPERLUKAN DI VS CODE:[/bold yellow]\n"
                "1. Buka tab notebook [bold green]notebook/build_index.ipynb[/bold green]\n"
                "2. Pastikan kernel di pojok kanan atas terhubung ke [bold green]Google Colab[/bold green]\n"
                "3. Klik tombol [bold green]▶▶ Run All[/bold green]\n"
                "4. [bold cyan]PENTING:[/bold cyan] Begitu Sel 6 selesai, tekan [bold cyan]Ctrl + S[/bold cyan] (Simpan) jika editor belum auto-save!\n\n"
                "[bold white]OPSI INTERAKTIF TERMINAL:[/bold white]\n"
                "• Tekan [bold green][ENTER][/bold green] di terminal ini kapan saja jika eksekusi sudah selesai & tersimpan\n"
                "• Atau paste link URL trycloudflare langsung ke terminal ini"
            )
            console.print(Panel(guide_panel, box=box.ROUNDED, style="yellow"))
        else:
            print("\n" + "-" * 76)
            print("▶ LANGKAH 2/4: Eksekusi GPU di Google Colab Runtime")
            print("-" * 76)
            print("TINDAKAN DIPERLUKAN DI VS CODE:")
            print("1. Buka tab notebook notebook/build_index.ipynb")
            print("2. Pastikan kernel terhubung ke Google Colab")
            print("3. Klik tombol 'Run All'")
            print("4. Tekan 'Ctrl + S' (Simpan) begitu eksekusi selesai!")
            print("\nOPSI INTERAKTIF TERMINAL:")
            print("• Tekan [ENTER] di terminal ini kapan saja jika eksekusi sudah selesai & tersimpan")
            print("• Atau paste link URL trycloudflare langsung ke sini")
            print("-" * 76)

        # Buka notebook di VS Code secara otomatis jika belum terbuka
        open_notebook_in_editor(notebook_path)

        # Tunggu hingga selesai dieksekusi di Colab
        wait_for_colab_execution(notebook_path=notebook_path, timeout_seconds=timeout)
    else:
        logger.info("Melewati Langkah 2 (Menggunakan URL download atau file JSON yang sudah ada).")

    # =========================================================================
    # TAHAP 3: Tarik File Vektor Hasil Indexing dari Colab
    # =========================================================================
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]▶ LANGKAH 3/4: Menarik Vektor Hasil Indexing dari Colab[/bold cyan]")
    else:
        print("\n" + "-" * 76)
        print("▶ LANGKAH 3/4: Menarik Vektor Hasil Indexing dari Colab")
        print("-" * 76)

    resolved_json = colab_json_path
    if not resolved_json or not resolved_json.exists():
        resolved_json = fetch_colab_vectors_json(
            colab_url=colab_url,
            notebook_path=notebook_path,
            output_json_path=DEFAULT_JSON_PATH
        )

    if not resolved_json or not resolved_json.exists():
        logger.error("Gagal mendapatkan file JSON vektor dari Colab!")
        sys.exit(1)

    json_size_mb = resolved_json.stat().st_size / (1024 * 1024)
    logger.info(f"File vektor siap diproses: {resolved_json.resolve()} ({json_size_mb:.2f} MB)")

    # =========================================================================
    # TAHAP 4: Update Database Lokal ChromaDB & Manifest
    # =========================================================================
    if RICH_AVAILABLE:
        console.print("\n[bold cyan]▶ LANGKAH 4/4: Menyimpan Vektor ke ChromaDB Lokal[/bold cyan]")
    else:
        print("\n" + "-" * 76)
        print("▶ LANGKAH 4/4: Menyimpan Vektor ke ChromaDB Lokal")
        print("-" * 76)

    run_incremental_update(
        data_dir=data_dir,
        manifest_path=manifest_path,
        dry_run=False,
        force_all=force_all,
        clean_deleted=clean_deleted,
        override_classification=override_classification,
        from_colab=True,
        colab_json_path=resolved_json
    )

    total_elapsed = time.time() - overall_start_time

    if RICH_AVAILABLE:
        success_summary = (
            f"[bold green]🎉 PIPELINE BUILD INDEX BERHASIL DISELESAIKAN DENGAN SUKSES![/bold green]\n\n"
            f"• Waktu Total Eksekusi : [bold cyan]{total_elapsed:.1f} detik[/bold cyan]\n"
            f"• Sumber Vektor       : [bold]{resolved_json.name}[/bold] ({json_size_mb:.2f} MB)\n"
            f"• Database Lokal      : [bold]ChromaDB (localhost:8000)[/bold]\n"
            f"• Manifest Status     : [bold green]{manifest_path.name} Terbarui[/bold green]\n\n"
            f"[dim]Data baru sudah siap digunakan untuk retrieval dan inference RAG PUSDIKZI.[/dim]"
        )
        console.print("\n", Panel(success_summary, box=box.DOUBLE, style="green"))
    else:
        print("\n" + "=" * 76)
        print("🎉 PIPELINE BUILD INDEX BERHASIL DISELESAIKAN DENGAN SUKSES!")
        print("=" * 76)
        print(f"• Waktu Total Eksekusi : {total_elapsed:.1f} detik")
        print(f"• Sumber Vektor       : {resolved_json.name} ({json_size_mb:.2f} MB)")
        print(f"• Database Lokal      : ChromaDB (localhost:8000)")
        print(f"• Manifest Status     : {manifest_path.name} Terbarui")
        print("=" * 76 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline Terpadu Build Index RAG PUSDIKZI (Colab GPU to Local ChromaDB)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/raw/SECRET",
        help="Direktori sumber dokumen (default: ./data/raw/SECRET)"
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default=str(DEFAULT_NOTEBOOK_PATH),
        help="Path ke notebook/build_index.ipynb (default: ./notebook/build_index.ipynb)"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_DEFAULT_PATH),
        help="Path ke manifest .index_manifest.json"
    )
    parser.add_argument(
        "--all",
        "--force",
        dest="force_all",
        action="store_true",
        help="Paksa proses ulang seluruh dokumen termasuk dokumen lama"
    )
    parser.add_argument(
        "--clean-deleted",
        action="store_true",
        help="Hapus vektor dari ChromaDB untuk dokumen yang filenya sudah dihapus dari folder data"
    )
    parser.add_argument(
        "--skip-sync",
        action="store_true",
        help="Lewati Langkah 1 (jangan perbarui snapshot dokumen di notebook)"
    )
    parser.add_argument(
        "--pull-only",
        action="store_true",
        help="Hanya jalankan Langkah 3 dan 4 (Tarik vektor dari notebook/file yang sudah ada dan update ke ChromaDB)"
    )
    parser.add_argument(
        "--colab-url",
        type=str,
        default=None,
        help="URL download langsung data_colab.zip atau colab_indexed_vectors.json dari tunnel Cloudflare"
    )
    parser.add_argument(
        "--colab-json",
        type=str,
        default=None,
        help="Path ke file JSON vektor yang sudah diunduh secara manual"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Batas waktu maksimum menunggu eksekusi Colab dalam detik (default: 1800s / 30 menit)"
    )
    parser.add_argument(
        "--classification",
        type=str,
        default=None,
        help="Override klasifikasi keamanan RBAC (misal: SECRET, INTERNAL, PUBLIC)"
    )

    args = parser.parse_args()

    # Jadikan executable jika dijalankan langsung
    run_pipeline(
        data_dir=Path(args.data_dir),
        notebook_path=Path(args.notebook),
        manifest_path=Path(args.manifest),
        colab_url=args.colab_url,
        colab_json_path=Path(args.colab_json) if args.colab_json else None,
        force_all=args.force_all,
        skip_sync=args.skip_sync or args.pull_only,
        clean_deleted=args.clean_deleted,
        timeout=args.timeout,
        override_classification=args.classification
    )

