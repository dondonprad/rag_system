# PRODUCT REQUIREMENT DOCUMENT (PRD)
## SISTEM RAG PUSDIKZI (PROJECT EMAS)
**Pusat Pendidikan Zeni TNI AD — Sistem Knowledge Base & Tanya Jawab Berbasis RAG**

---

### Informasi Dokumen
| Atribut | Keterangan |
| :--- | :--- |
| **Nama Proyek** | Project Emas - Modul RAG Terintegrasi PUSDIKZI |
| **Versi Dokumen** | 1.2.0 (Fase PoC Menuju Integrasi LLM) |
| **Branch Terkait** | `llm_itegration` |
| **Status Dokumen** | Active / Living Specification |
| **Target Platform** | Intel Core Ultra 125H (Meteor Lake, Intel Arc iGPU) & Hybrid Cloud-Colab Pipeline |
| **Klasifikasi Keamanan**| Internal Militer & Air-Gapped Ready (Zero Data Leakage) |

---

## 1. Ringkasan Eksekutif & Latar Belakang

Pusat Pendidikan Zeni (PUSDIKZI) TNI AD mengelola volume dokumentasi doktrin militer, petunjuk teknis (Juknis), petunjuk pelaksanaan (Juklak), kurikulum pendidikan perwira/bintara/tamtama, materi zeni tempur/konstruksi/jihandak, dan SOP administrasi dinas yang masif. 

Pencarian manual atas ribuan halaman dokumen ini memakan waktu dan rentan *human error*. **Project Emas - Modul RAG PUSDIKZI** dirancang untuk menghadirkan sistem temu-kembali informasi dan tanya jawab interaktif (*Retrieval-Augmented Generation*) yang:
1. **On-Premise & Air-Gapped Ready**: Dapat beroperasi tanpa koneksi internet publik untuk menjaga kerahasiaan data militer.
2. **Zero Data Leakage**: Dokumen rahasia tidak pernah terekspos ke repositori publik, internet, atau logging telemetri eksternal.
3. **Role-Based Access Control (RBAC)**: Mengatur hak akses kueri data berdasarkan klasifikasi kerahasiaan bertingkat.
4. **Efisiensi Komputasi Fleksibel (Hybrid Compute)**: Mengoptimalkan pemrosesan di laptop lokal (Intel Core Ultra 125H) dengan opsi akselerasi batch indexing di GPU Google Colab melalui pipeline otomasi terisolasi.

---

## 2. Tujuan & Sasaran Produk (Goals & Objectives)

### 2.1 Tujuan Utama (Primary Goals)
- **Akurasi Temu Balik Informasi**: Menghasilkan dokumen rujukan yang relevan hingga tingkat potongan paragraf (*chunk*) dengan skor similaritas kosinus yang tinggi.
- **Transparansi Sumber (Explainability)**: Setiap potongan data wajib menyertakan metadata mandiri (`sumber`, `tipe_dokumen`, `halaman`, `klasifikasi_keamanan`).
- **Pemisahan Hak Akses**: Pengguna dengan hak akses standar (`INTERNAL`) tidak boleh mendapatkan hasil ekstraksi dari dokumen `SECRET`.

### 2.2 Sasaran Teknis (Technical Milestones)
- [x] **Fase 1 (Inisialisasi & Storage)**: Setup Vector Database ChromaDB di container Docker on-premise dengan healthcheck port 8000.
- [x] **Fase 2 (Ingestion & Cleaning)**: Implementasi parsing multi-format (PDF, DOCX, XLSX/XLS, TXT, MD) berbasis pustaka lokal tanpa ketergantungan API pihak ketiga.
- [x] **Fase 3 (Structure-Aware Chunking & Tokenizer)**: Standarisasi potongan 256 token dengan 10% overlap berbasis batasan semantik/paragraf.
- [x] **Fase 4 (Embedding & Hybrid Pipeline)**: Integrasi model embedding `nomic-ai/nomic-embed-text-v1.5` dengan opsi lokal XPU (Intel Arc) serta pipeline otomatisasi inkremental via Google Colab GPU (`scripts/run_pipeline.py`).
- [ ] **Fase 5 (LLM Integration - Sedang Berjalan)**: Menghubungkan hasil retrieval dengan engine LLM lokal (Ollama / IPEX-LLM / GGUF terkuantisasi) untuk sintesis jawaban dan antarmuka web interaktif (Open WebUI).

---

## 3. Arsitektur Sistem (System Architecture)

### 3.1 Diagram Alur Sistem Terkini

```
                                  [DATA SOURCE]
                 Dokumen PUSDIKZI (PDF, DOCX, XLSX, TXT, MD)
                 ├── data/raw/         --> Klasifikasi INTERNAL
                 └── data/raw/SECRET/  --> Klasifikasi SECRET
                                        │
                                        ▼
                         [INGESTION & PARSING SERVICE]
                          FastAPI (app/main.py) / Python
                 - PyMuPDF: Konversi Tabel PDF ke Markdown
                 - python-docx: Ekstraksi teks & tabel terstruktur
                 - openpyxl/xlrd: Konversi baris spreadsheet ke narasi
                                        │
                                        ▼
                        [PROCESSING & SANITIZATION]
                 - Text Cleaner: De-hyphenation, normalisasi spasi/simbol
                 - PII/Sanitization: Masking NIK, NRP/NIP, No HP
                 - Chunking: Structure-Aware 256 tokens (10% overlap)
                 - Metadata: Injeksi {sumber, tipe, halaman, klasifikasi}
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
     [JALUR 1: LOKAL (XPU)]                                [JALUR 2: HYBRID COLAB GPU]
- scripts/build_index.py                              - scripts/sync_colab_notebook.py
- Model: nomic-embed-text-v1.5                          (Kompresi LZMA ke notebook)
- Device: Intel Arc iGPU (batch size 16)              - notebook/build_index.ipynb (Colab GPU)
- Output: Vektor langsung diinjeksi                   - scripts/pull_colab_index.py (ZIP fetch)
                                                      - scripts/update_index.py (Inkremental)
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                            [VECTOR STORE CHROMADB]
                        Container Docker (Port 8000)
                        - Koleksi: pusdikzi_knowledge_base
                        - Metrik: Cosine Distance (HNSW Index)
                        - Penyimpanan Persisten: chroma_data volume
                                        │
                                        ▼
                            [RETRIEVAL ENGINE (RBAC)]
                           scripts/test_retrieval.py
                     - Query Embedding: "search_query: ..."
                     - Metadata Filter: where={"klasifikasi": {"$in": [...]}}
                     - Output: Top-K context relevan + Metadata sitasi
                                        │
                                        ▼  [TAHAP SEDANG BERJALAN: llm_itegration]
                            [LOCAL LLM ORCHESTRATION]
                     - IPEX-LLM / Ollama On-Premise
                     - Grounded Militer Prompting (Anti-Halusinasi)
                     - Open WebUI Dashboard
```

---

## 4. Rincian Spesifikasi Fungsional

### 4.1 Modul Ingestion & Parsing (`src/ingestion/`)
- **Format Didukung**: `.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.txt`, `.md`.
- **Ekstraksi PDF**: Menggunakan PyMuPDF (`fitz`). Seluruh tabel dokumen wajib diekstrak dan ditransformasi menjadi format tabel Markdown terstandarisasi.
- **Ekstraksi DOCX**: Ekstraksi paragraf dan tabel berurutan sesuai alur baca dokumen.
- **Ekstraksi Spreadsheet (XLSX/XLS)**: Mengonversi data tabular menjadi representasi tabel Markdown sekaligus format narasi berbasis entitas baris (*"Pada baris X, Kolom A adalah Y..."*) agar dapat dipahami secara optimal oleh model embedding vektor.
- **Metadata Otomatis**: Setiap halaman/dokumen otomatis dilekatkan metadata:
  ```json
  {
    "sumber": "pedoman_jihandak_2026.pdf",
    "tipe": "pdf",
    "halaman": 12,
    "klasifikasi": "SECRET",
    "hash_dokumen": "e3b0c44298fc1c149afbf4c8996fb924..."
  }
  ```

### 4.2 Modul Pembersihan & Chunking (`src/processing/`)
- **Pembersihan Otomatis (Sanitization & Boilerplate Removal)**:
  - De-hyphenation kata terpotong di akhir baris (`misal- \n nya` $\rightarrow$ `misalnya`).
  - Normalisasi whitespace, newline berulang, dan pembersihan simbol anomali encoding.
  - Penghapusan header/footer non-kontekstual yang berulang.
  - Sanitasi PII (*Personally Identifiable Information*): Masking otomatis terhadap format NIK (16 digit), NRP/NIP militer, nomor kontak, serta penanda kerahasiaan.
- **Strategi Chunking**:
  - **Panjang Chunk**: 256 token (berbasis *subword tokenization* model target).
  - **Overlap**: Rasio 10% (~26 token) untuk menjaga kesinambungan semantik antar-potongan.
  - **Structure-Aware**: Pemotongan mengutamakan batas paragraf (`\n\n`), kalimat (`. `), dan klausa sebelum memotong di tengah kata.

### 4.3 Modul Embedding & Optimasi Hardware (`src/embeddings/`)
- **Model Embedding Master**: `nomic-ai/nomic-embed-text-v1.5`.
  - Prefix Dokumen: `search_document: `
  - Prefix Kueri: `search_query: `
  - Dimensi Vektor: 768 dimensi (Normalisasi L2 aktif).
- **Mitigasi Hardware Lokal (Intel Core Ultra 125H - Meteor Lake)**:
  - Arsitektur iGPU: Intel Arc Graphics (7 Xe-cores, memori terbagi / Shared VRAM max 4.5 GB).
  - Penyesuaian Batch Size: Diturunkan ke `16` (dari default 32/256) untuk mencegah *XPU Out-Of-Memory* (OOM).
  - Konfigurasi Driver: Akselerasi Intel Level Zero / SYCL (`SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1`).
- **Otomasi Hybrid Google Colab GPU (`scripts/run_pipeline.py`)**:
  - Sinkronisasi cerdas melalui pelacakan `.index_manifest.json` (hanya dokumen baru atau yang diubah yang diproses ulang).
  - Snapshot teks terkompresi LZMA diinjeksikan langsung ke dalam [`notebook/build_index.ipynb`](notebook/build_index.ipynb).
  - Watcher otomatis mendeteksi selesainya eksekusi Colab di VS Code.
  - Hasil embedding diekstrak dari arsip `data/zip/` dan diinjeksi ke ChromaDB lokal secara inkremental tanpa membebani GPU/RAM lokal.

### 4.4 Modul Vector Storage & Retrieval (`src/vector_store/`)
- **Database**: ChromaDB (Docker Engine, Port 8000, volume mount persisten `/data`).
- **Koleksi**: `pusdikzi_knowledge_base`.
- **Indeks & Metrik**: HNSW dengan metrik *Cosine Distance*.
- **Role-Based Access Control (RBAC)**:
  - Tingkat klasifikasi: `PUBLIC`, `INTERNAL`, `RESTRICTED`, `CONFIDENTIAL`, `SECRET`.
  - Default folder `data/raw/` $\rightarrow$ `INTERNAL`.
  - Folder khusus `data/raw/SECRET/` $\rightarrow$ `SECRET`.
  - Kueri retrieval menggunakan filter metadata ketat:
    ```python
    where = {"klasifikasi": {"$in": allowed_classifications}}
    ```

### 4.5 Modul API & Layanan Ekstraksi (`app/main.py`)
- FastAPI service on-premise untuk parsing berkas independen:
  - `POST /clean`: Membersihkan teks mentah dari boilerplate dan anomali.
  - `POST /chunk`: Melakukan token-based chunking dan kalkulasi statistik.
  - `POST /extract`: Menerima upload berkas PDF/DOCX/XLSX dan mengekstrak teks, tabel Markdown, serta metadata terstruktur.

---

## 5. Non-Functional Requirements (NFR)

| Kategori | Spesifikasi & Batasan |
| :--- | :--- |
| **Keamanan & Kerahasiaan** | - Direktori `data/` (`raw/`, `raw/SECRET/`, `processed/`, `zip/`, database) terdaftar 100% pada `.gitignore`.<br>- Mode offline Hugging Face diaktifkan (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`).<br>- Telemetri ChromaDB dinonaktifkan (`ANONYMIZED_TELEMETRY=FALSE`). |
| **Efisiensi Memori** | Alokasi RAM container ChromaDB dibatasi maksimal 1024 MB di `docker-compose.yml`. Eksekusi lokal embedding dibatasi batch size 16. |
| **Ketergantungan Eksternal** | Tidak mengandalkan cloud API berbayar (OpenAI/Anthropic). Kode pipeline dibangun mandiri (native python/chromadb/fastapi) tanpa framework wrapper berat yang tidak dibutuhkan. |
| **Integritas Dokumen** | Menggunakan SHA-256 fingerprinting pada manifest indeks (`.index_manifest.json`) untuk mencegah duplikasi pemrosesan dokumen identik. |

---

## 6. Rencana Tahap Selanjutnya: Integrasi LLM (Roadmap Branch `llm_itegration`)

Dengan tuntasnya modul ingestion, chunking, embedding, database ChromaDB, dan pipeline sinkronisasi inkremental, fokus pengembangan pada branch `llm_itegration` mencakup:

1. **Deploy Engine LLM On-Premise**:
   - Pilihan: Ollama lokal atau IPEX-LLM terakselerasi Intel Core Ultra 125H.
   - Target Model: Model instruksi 7B/8B (misal: Llama-3-8B-Instruct, Mistral-7B, atau Qwen2.5-7B) dengan kuantisasi 4-bit (GGUF / INT4).
2. **Context Injection & Prompt Engineering Militer**:
   - Format prompt yang menekankan integritas instruksi: menjawab hanya berdasarkan potongan konteks yang disuplai oleh ChromaDB (*strict grounding*).
   - Penanganan kondisi *no match* ("Informasi tidak ditemukan dalam dokumen doktrin PUSDIKZI").
   - Penyertaan sitasi wajib (Nama Dokumen dan Halaman).
3. **Antarmuka Interaktif Pengguna (GUI)**:
   - Integrasi container Open WebUI atau UI kustom berbasis FastAPI untuk staf dan siswa PUSDIKZI.
   - Manajemen session login dengan penyesuaian hak akses RBAC kueri pengguna.

---

## 7. Struktur Repositori Terkini

```text
rag_pusdikzi/
├── app/
│   ├── __init__.py
│   └── main.py                     # FastAPI Local Data Parsing & Extraction Service
├── config/
│   ├── __init__.py
│   ├── config.yaml                 # Master Configuration (Model, Ingestion, Hardware)
│   └── settings.py                 # Dynamic Configuration Loader (.env fallback)
├── data/                           # [IGNORED BY GIT - ZERO LEAKAGE]
│   ├── raw/                        # Dokumen input (INTERNAL)
│   │   └── SECRET/                 # Dokumen berklasifikasi SECRET
│   ├── processed/                  # Cache ekstraksi teks
│   └── zip/                        # Payload arsip ZIP hasil eksekusi Colab
├── docs/
│   ├── SETUP.md                    # Panduan setup & instalasi on-premise
│   └── PRD_RAG_Pusdikzi.md         # Product Requirement Document (Dokumen ini)
├── notebook/
│   └── build_index.ipynb           # Notebook eksekusi GPU Google Colab via VS Code
├── scripts/
│   ├── build_index.py              # CLI build index lokal (Intel Arc XPU/CPU)
│   ├── run_pipeline.py             # CLI Orchestrator: Pipeline otomatis sinkronisasi ke Colab & lokal
│   ├── sync_colab_notebook.py      # Ekstraksi incremental & kompresi LZMA ke notebook
│   ├── pull_colab_index.py         # Ekstraktor payload ZIP dari Colab
│   ├── update_index.py             # Injeksi inkremental chunk & vektor ke ChromaDB
│   ├── check_database.py           # Utilitas inspeksi isi & koleksi ChromaDB
│   └── test_retrieval.py           # CLI pengujian retrieval vektor dengan RBAC filtering
├── src/
│   ├── embeddings/
│   │   └── embedding_engine.py     # SentenceTransformers engine (Nomic AI v1.5)
│   ├── ingestion/
│   │   ├── loaders.py              # File scanner & loader rekursif
│   │   └── parsers.py              # Parser spesifik (PyMuPDF, docx, openpyxl, xlrd)
│   ├── processing/
│   │   ├── cleaner.py              # Normalisasi teks & sanitasi PII
│   │   └── chunking.py             # Structure-aware tokenizer & chunker (256 tokens)
│   └── vector_store/
│       └── chroma_db.py            # Client ChromaDB (HTTP & Persistent) + Filter RBAC
├── docker-compose.yml              # Deployment ChromaDB Docker on-premise (Port 8000)
├── requirements.txt                # Dependensi Python
├── .env.example                    # Template variabel lingkungan
├── .env                            # Konfigurasi aktif lokal
└── README.md                       # Dokumentasi petunjuk teknis & tutorial pipeline
```

