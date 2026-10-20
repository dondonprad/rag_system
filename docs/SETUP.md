# Panduan Setup & Eksekusi: Project Emas - Modul RAG PUSDIKZI
**Fase: Proof-of-Concept (PoC) Lokal / Air-Gapped (Intel Core Ultra 125H)**

---

## 1. Arsitektur & Prasyarat Sistem
- **Hardware Target**: Intel Core Ultra 125H (Meteor Lake iGPU Arc 7 Xe-cores, RAM terbagi).
- **Docker Engine**: Minimal versi 24+ & Docker Compose v2.
- **Python**: 3.10 - 3.14.

---

## 2. Struktur Modul Tahap Awal (Data Ingestion & Vector Storage)
```text
rag_pusdikzi/
├── config/
│   ├── __init__.py
│   ├── config.yaml          # Parameter master (Chroma, Embedding, mitigasi RAM)
│   └── settings.py          # Loader config otomatis dengan fallback .env
├── data/
│   ├── raw/                 # Tempat dokumen PDF, TXT, MD PUSDIKZI
│   │   ├── SECRET/          # Subdirektori otomatis mendapat RBAC SECRET
│   │   ├── sop_administrasi_pusdikzi.md
│   │   └── profil_pusdikzi_tni_ad.txt
│   └── processed/           # Cache data hasil ekstraksi
├── src/
│   ├── ingestion/           # Loader & parser lokal on-premise
│   ├── processing/          # Pembersih teks (sanitasi PII/rahasia) & chunking
│   ├── embeddings/          # Embedding engine lokal (Nomic Embed Text / ST)
│   └── vector_store/        # Klien ChromaDB Docker dengan RBAC metadata
├── scripts/
│   ├── build_index.py       # Script CLI pemrosesan dokumen ke ChromaDB
│   └── test_retrieval.py    # Script CLI pengujian pencarian vektor & RBAC
├── docker-compose.yml       # Orkestrasi ChromaDB container (port 8000)
├── requirements.txt         # Dependensi Python
├── .env.example             # Contoh variabel lingkungan
└── .env                     # File konfigurasi aktif lokal
```

---

## 3. Langkah-Langkah Menjalankan

### Langkah 1: Jalankan Database Vektor ChromaDB di Docker
Jalankan container ChromaDB yang terisolasi secara on-premise:
```bash
docker compose up -d
```
Pastikan container berjalan dengan memeriksa log / status:
```bash
docker compose ps
curl http://localhost:8000/api/v2/heartbeat
```

### Langkah 2: Persiapkan Lingkungan Virtual Python
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Langkah 3: Tempatkan Dokumen PUSDIKZI
Letakkan file dokumen administratif (`.pdf`, `.txt`, `.md`) ke dalam folder `data/raw/`.
- File di direktori `data/raw/` otomatis berklasifikasi `INTERNAL`.
- File di direktori `data/raw/SECRET/` otomatis berklasifikasi `SECRET`.

### Langkah 4: Jalankan Pembuatan Embedding dan Indeks Vektor
Eksekusi script indexing untuk memproses teks, melakukan embedding, dan menyimpannya ke ChromaDB:
```bash
python scripts/build_index.py
```
Opsi tambahan:
- `--reset`: Mengosongkan koleksi lama sebelum mengindeks ulang dokumen baru.
- `--data-dir /path/lain`: Menentukan folder dokumen alternatif.

### Langkah 5: Uji Pencarian / Retrieval Vektor
Uji kueri pencarian teks langsung dari terminal:
```bash
# Pencarian umum:
python scripts/test_retrieval.py "Berapa lama batas pengarsipan berkas dinas?"

# Pencarian dengan pembatasan RBAC dokumen SECRET:
python scripts/test_retrieval.py "Berapa personel tim Jihandak?" --access SECRET
```

