# RAG PUSDIKZI (Project Emas) - Knowledge Base System

Sistem Retrieval-Augmented Generation (RAG) on-premise untuk Pusat Pendidikan Zeni (PUSDIKZI) TNI AD. Proyek ini mengintegrasikan dokumen operasional dan administratif dengan pengamanan berbasis klasifikasi (RBAC: `PUBLIC`, `INTERNAL`, `SECRET`) serta alur komputasi hybrid: **pemrosesan embedding di GPU Google Colab** dan **penyimpanan vektor di ChromaDB lokal**.

---

## 📑 Daftar Isi
- [Arsitektur Hybrid Pipeline](#-arsitektur-hybrid-pipeline)
- [Prasyarat Sistem](#-prasyarat-sistem)
- [Tutorial: Menjalankan Pipeline Otomatis (Colab GPU ke ChromaDB Lokal)](#-tutorial-menjalankan-pipeline-otomatis)
  - [1. Persiapan Dokumen](#1-persiapan-dokumen)
  - [2. Eksekusi Script Pipeline](#2-eksekusi-script-pipeline)
  - [3. Opsi dan Parameter CLI](#3-opsi-dan-parameter-cli)
- [Detail Komponen Script](#-detail-komponen-script)
- [Pengujian & Verifikasi](#-pengujian--verifikasi)

---

## 🏗 Arsitektur Hybrid Pipeline

Untuk mengatasi keterbatasan resource komputasi laptop lokal saat memproses model embedding mutakhir (*Nomic Embed Text v1.5*), pipeline ini membagi beban kerja secara cerdas:

```
[Laptop / Server Lokal]                                   [Google Colab VM (GPU)]
┌──────────────────────────────────────┐                   ┌───────────────────────────┐
│ 1. data/raw/SECRET                   │                   │                           │
│    (Dokumen PDF, DOCX, XLSX, TXT)    │                   │                           │
│                 │                    │                   │                           │
│                 ▼                    │                   │                           │
│ 2. scripts/sync_colab_notebook.py    │                   │                           │
│    (Filter dokumen baru/berubah,     │                   │                           │
│     kompresi LZMA ke notebook)       │                   │                           │
│                 │                    │                   │                           │
│                 ▼                    │                   │                           │
│ 3. notebook/build_index.ipynb ───────┼── VS Code Colab ──┼─► 4. Eksekusi Sel GPU     │
│    (Watcher di scripts/run_pipeline) │   Extension       │      - Tokenisasi & Chunk │
│                                      │                   │      - Nomic Embed CUDA   │
│                 ▲                    │                   │      - Ekspor Vektor JSON │
│                 │                    │                   │            │              │
│ 5. scripts/pull_colab_index.py ◄─────┼── Base64 / Tunnel ┼────────────┘              │
│    (Ekstrak payload hasil Colab)     │                   │                           │
│                 │                    │                   │                           │
│                 ▼                    │                   │                           │
│ 6. scripts/update_index.py           │                   │                           │
│    (Simpan chunk & vektor ke         │                   │                           │
│     ChromaDB localhost:8000)         │                   │                           │
│                 │                    │                   │                           │
│                 ▼                    │                   │                           │
│ 7. .index_manifest.json (Updated)    │                   │                           │
└──────────────────────────────────────┘                   └───────────────────────────┘
```

---

## ⚙️ Prasyarat Sistem

1. **Docker & ChromaDB Aktif**:
   Container database vektor harus sudah berjalan di lokal:
   ```bash
   docker compose up -d
   # Verifikasi heartbeat ChromaDB
   curl http://localhost:8000/api/v2/heartbeat
   ```
2. **Virtual Environment Python**:
   ```bash
   source venv/bin/activate
   ```
3. **Ekstensi VS Code Google Colab**:
   Pastikan ekstensi resmi `google.colab` sudah terpasang di VS Code dan akun Google Anda sudah login.

---

## 🚀 Tutorial: Menjalankan Pipeline Otomatis

Seluruh tahapan (sinkronisasi $\rightarrow$ pemantauan Colab $\rightarrow$ download vektor $\rightarrow$ penyimpanan ke database) kini dirangkum dalam satu script terpadu: **`scripts/run_pipeline.py`**.

### 1. Persiapan Dokumen
Letakkan dokumen administratif atau rahasia yang ingin diindeks ke dalam folder:
* Dokumen Umum/Internal: `data/raw/`
* Dokumen Rahasia (RBAC SECRET): `data/raw/SECRET/`
* Arsip Hasil Download Colab: `data/zip/` (Otomatis dibuat untuk menyimpan file `.zip`)

> [!IMPORTANT]
> **Keamanan Dokumen (Zero Leakage):**
> Seluruh isi folder `data/` (`data/raw/`, `data/raw/SECRET/`, `data/processed/`, `data/zip/`, dan database vektor) **100% diabaikan oleh `.gitignore`**, sehingga dokumen sensitif maupun file `.zip` tidak akan pernah terunggah ke repositori GitHub.

### 2. Eksekusi Script Pipeline
Jalankan perintah berikut di terminal:

```bash
./venv/bin/python scripts/run_pipeline.py
```

Setelah perintah dijalankan, sistem akan memproses langkah-langkah berikut secara berurutan:

1. **Langkah 1 (Sinkronisasi Inkremental)**:
   * Script membaca `.index_manifest.json`. Dokumen lama yang sudah terindeks dan tidak berubah akan otomatis **dilewati (SKIP)**.
   * Dokumen baru atau yang mengalami perubahan isi akan diekstrak teksnya, dikompresi dengan LZMA, dan disematkan ke dalam [`notebook/build_index.ipynb`](notebook/build_index.ipynb).
   * Sel output lama di notebook dibersihkan agar watcher bekerja presisi.
2. **Langkah 2 (Eksekusi GPU di Colab via VS Code)**:
   * Script otomatis membuka file `notebook/build_index.ipynb` di editor VS Code Anda.
   * Terminal akan menampilkan pesan instruksi dan masuk ke mode **Watching** (menunggu eksekusi Colab).
   * **Tindakan Anda:**
     1. Pastikan kernel di pojok kanan atas notebook terhubung ke **Google Colab**.
     2. Klik tombol **▶▶ Run All** (atau tekan shortcut Run All).
     3. Tekan **`Ctrl + S`** (Simpan) begitu eksekusi selesai jika editor belum otomatis menyimpan output.
     4. *(Opsional)* Anda dapat menekan **`[ENTER]`** di terminal kapan saja untuk memicu deteksi instan, atau mem-paste langsung URL Cloudflare tunnel jika muncul.
3. **Langkah 3 (Penarikan Vektor Otomatis)**:
   * Begitu Sel 6 selesai berjalan di Colab, script mendeteksi output vektor (baik melalui metadata LZMA Base64 notebook maupun link Cloudflare tunnel).
   * Data vektor otomatis diunduh dan diekstrak ke `data/processed/colab_indexed_vectors.json`.
4. **Langkah 4 (Penyimpanan ke ChromaDB Lokal)**:
   * Script menyimpan seluruh chunk dokumen dan embedding vektor ke ChromaDB lokal (`localhost:8000`).
   * Manifest `data/processed/.index_manifest.json` diperbarui dengan sidik jari SHA-256 dokumen baru.

---

### 3. Opsi dan Parameter CLI

Script `scripts/run_pipeline.py` mendukung berbagai flag fleksibel sesuai kebutuhan:

| Perintah / Flag | Kegunaan |
|---|---|
| `./venv/bin/python scripts/run_pipeline.py` | Menjalankan pipeline standar inkremental (hanya file baru/berubah). |
| `./venv/bin/python scripts/run_pipeline.py --all` | Memproses ulang **seluruh dokumen** dari awal (mengabaikan riwayat manifest). |
| `./venv/bin/python scripts/run_pipeline.py --pull-only` | Melewati tahap sinkronisasi & watcher; langsung mengekstrak vektor dari notebook/file yang sudah ada dan memasukkannya ke ChromaDB. |
| `./venv/bin/python scripts/run_pipeline.py --clean-deleted` | Otomatis menghapus vektor dari ChromaDB jika file aslinya telah dihapus dari folder `data/raw/`. |
| `./venv/bin/python scripts/run_pipeline.py --colab-url <URL>` | Mengunduh hasil vektor langsung dari URL Cloudflare tunnel yang dicetak di Sel 6 Colab. |
| `./venv/bin/python scripts/run_pipeline.py --timeout 3600` | Mengubah batas waktu tunggu eksekusi Colab (default: 1800 detik / 30 menit). |

---

## 📂 Detail Komponen Script

Modul pembangun pipeline di folder `scripts/`:

* **[`scripts/run_pipeline.py`](scripts/run_pipeline.py)**: Orkestrator utama yang menyatukan alur end-to-end dengan UI terminal interaktif.
* **[`scripts/sync_colab_notebook.py`](scripts/sync_colab_notebook.py)**: Engine seleksi dokumen baru vs lama berdasarkan SHA-256 dan injector payload LZMA ke notebook.
* **[`notebook/build_index.ipynb`](notebook/build_index.ipynb)**: Notebook mandiri GPU Colab yang memuat dependensi, tokenizer, chunker, model embedding Nomic, dan exporter vektor.
* **[`scripts/pull_colab_index.py`](scripts/pull_colab_index.py)**: Handler pengunduhan & dekompresi file vektor dari lingkungan Colab ke lokal.
* **[`scripts/update_index.py`](scripts/update_index.py)**: Engine sinkronisasi inkremental ke ChromaDB dengan pemeliharaan integritas manifest.

---

## 🔍 Pengujian & Verifikasi

Setelah pipeline selesai dijalankan, Anda dapat memverifikasi isi database dan menguji semantic search:

### 1. Pengecekan Status Database ChromaDB
Melihat jumlah vektor, dokumen terdaftar, dan metadata klasifikasi di ChromaDB:
```bash
./venv/bin/python scripts/check_database.py
```

### 2. Pengujian Retrieval (Semantic Search)
Menguji pencarian informasi dengan simulasi hak akses RBAC:
```bash
# Pencarian umum (Level INTERNAL/PUBLIC):
./venv/bin/python scripts/test_retrieval.py "Bagaimana prosedur perizinan cuti dinas?"

# Pencarian khusus dengan hak akses SECRET:
./venv/bin/python scripts/test_retrieval.py "Struktur dan kesiapan personel satuan Zeni" --access SECRET
```
