import os
import shutil
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.settings import settings
from src.ingestion.parsers import DocumentParser, DocumentPage
from src.processing.cleaner import TextCleaner
from src.processing.chunking import TextChunker, DocumentChunk

logger = logging.getLogger("fastapi_parsing_service")

# Inisialisasi FastAPI App
app = FastAPI(
    title="PUSDIKZI RAG - Local Data Parsing & Ingestion Service",
    description=(
        "Layanan ekstraksi dokumen on-premise berkeamanan tinggi (Air-Gapped PoC):\n"
        "- Ekstraksi berbasis library lokal (PyMuPDF, python-docx, openpyxl, xlrd)\n"
        "- Konversi tabel PDF & DOCX ke format Markdown terstruktur\n"
        "- Konversi spreadsheet XLSX/XLS ke tabel Markdown & teks narasi baris entitas\n"
        "- Pembersihan otomatis (Deduplikasi, normalisasi format, de-hyphenation, boilerplate removal)\n"
        "- Chunking structure-aware 256 token dengan 10% overlap\n"
        "- Otomatisasi penyisipan metadata mandiri ({ sumber, tipe, halaman, klasifikasi })"
    ),
    version="1.0.0"
)

# Aktifkan CORS untuk dashboard lokal
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

cleaner = TextCleaner()
chunker = TextChunker()

# --- Schemas ---

class CleanRequest(BaseModel):
    text: str = Field(..., description="Teks mentah yang akan dibersihkan")

class CleanResponse(BaseModel):
    cleaned_text: str
    original_length: int
    cleaned_length: int

class ChunkRequest(BaseModel):
    text: str = Field(..., description="Teks yang akan dipotong")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata dokumen dasar")
    chunk_size: Optional[int] = Field(256, description="Ukuran potongan dalam token (default: 256)")
    chunk_overlap: Optional[int] = Field(26, description="Jumlah overlapping token (~10%)")

class LocalFileParseRequest(BaseModel):
    file_path: str = Field(..., description="Path absolut atau relatif file di server lokal")
    classification: Optional[str] = Field("internal", description="Tingkat kerahasiaan dokumen")
    clean_boilerplate: Optional[bool] = Field(True, description="Apakah menjalankan pembersihan otomatis")

class DocumentChunkResponse(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any]

class DocumentPageResponse(BaseModel):
    content: str
    page_number: int
    metadata: Dict[str, Any]

class PipelineResponse(BaseModel):
    filename: str
    total_pages: int
    total_chunks: int
    classification: str
    chunks: List[DocumentChunkResponse]

# --- Endpoints ---

@app.get("/", tags=["Info"])
def root():
    return {
        "service": "PUSDIKZI RAG Local Data Parsing API",
        "status": "online",
        "version": "1.0.0",
        "environment": "air_gapped_on_premise",
        "documentation": "/docs"
    }

@app.get("/health", tags=["Info"])
def health():
    return {
        "status": "healthy",
        "supported_formats": [".pdf", ".docx", ".xlsx", ".xls", ".doc", ".txt", ".md"],
        "table_extraction_strategy": "Markdown table layout + entity-attribute narrative",
        "chunking": {
            "default_chunk_size": chunker.chunk_size,
            "default_chunk_overlap": chunker.chunk_overlap,
            "unit": "tokens",
            "structure_aware": True
        },
        "security_policy": "Zero Data Leakage / Air-Gapped"
    }

@app.post("/api/v1/clean", response_model=CleanResponse, tags=["Processing & Cleaning"])
def clean_text_endpoint(req: CleanRequest):
    """Pembersihan teks: deduplikasi baris, normalisasi format, de-hyphenation, dan boilerplate removal."""
    cleaned = cleaner.sanitize(req.text)
    return CleanResponse(
        cleaned_text=cleaned,
        original_length=len(req.text),
        cleaned_length=len(cleaned)
    )

@app.post("/api/v1/chunk", response_model=List[DocumentChunkResponse], tags=["Processing & Cleaning"])
def chunk_text_endpoint(req: ChunkRequest):
    """Chunking teks secara structure-aware dengan ukuran 256 token dan 10% overlap."""
    local_chunker = TextChunker(
        chunk_size=req.chunk_size or 256,
        chunk_overlap=req.chunk_overlap or 26
    )
    meta = req.metadata or {}
    meta.setdefault("sumber", "direct_text_input")
    meta.setdefault("tipe", "txt")
    meta.setdefault("halaman", 1)
    meta.setdefault("klasifikasi", "internal")

    dummy_page = DocumentPage(content=req.text, page_number=1, metadata=meta)
    chunks = local_chunker.chunk_document_page(dummy_page)

    return [DocumentChunkResponse(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata) for c in chunks]

def _parse_file_by_path(file_path: Path, classification: str, clean: bool) -> List[DocumentPage]:
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        pages = DocumentParser.parse_pdf(file_path, classification=classification)
    elif ext == ".docx":
        pages = DocumentParser.parse_docx(file_path, classification=classification)
    elif ext == ".xlsx":
        pages = DocumentParser.parse_xlsx(file_path, classification=classification)
    elif ext == ".xls":
        pages = DocumentParser.parse_xls(file_path, classification=classification)
    elif ext == ".doc":
        pages = DocumentParser.parse_doc(file_path, classification=classification)
    elif ext == ".txt":
        pages = DocumentParser.parse_text(file_path, classification=classification)
    elif ext == ".md":
        pages = DocumentParser.parse_markdown(file_path, classification=classification)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Format file '{ext}' belum didukung."
        )

    if clean:
        contents = [p.content for p in pages]
        cleaned_contents = cleaner.clean_document_pages(contents)
        for idx, p in enumerate(pages):
            p.content = cleaned_contents[idx]

    return pages

@app.post("/api/v1/parse-file", response_model=List[DocumentPageResponse], tags=["Local Data Parsing"])
def parse_local_file_endpoint(req: LocalFileParseRequest):
    """Mengekstrak dokumen lokal on-premise berdasarkan path file di server."""
    file_path = Path(req.file_path)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File '{req.file_path}' tidak ditemukan di filesystem server."
        )

    pages = _parse_file_by_path(
        file_path=file_path,
        classification=req.classification or "internal",
        clean=req.clean_boilerplate if req.clean_boilerplate is not None else True
    )

    return [DocumentPageResponse(content=p.content, page_number=p.page_number, metadata=p.metadata) for p in pages]

@app.post("/api/v1/parse", response_model=List[DocumentPageResponse], tags=["Local Data Parsing"])
async def parse_upload_file_endpoint(
    file: UploadFile = File(...),
    classification: str = Form("internal"),
    clean_boilerplate: bool = Form(True)
):
    """Mengekstrak file yang diunggah secara lokal menggunakan PyMuPDF/python-docx/openpyxl."""
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        # Override metadata sumber dengan nama file asli yang diunggah
        pages = _parse_file_by_path(
            file_path=tmp_path,
            classification=classification,
            clean=clean_boilerplate
        )
        for p in pages:
            p.metadata["sumber"] = file.filename
            p.metadata["file_name"] = file.filename

        return [DocumentPageResponse(content=p.content, page_number=p.page_number, metadata=p.metadata) for p in pages]
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

@app.post("/api/v1/pipeline", response_model=PipelineResponse, tags=["Complete Ingestion Pipeline"])
async def pipeline_endpoint(
    file: UploadFile = File(...),
    classification: str = Form("internal"),
    chunk_size: int = Form(256),
    chunk_overlap: int = Form(26)
):
    """Pipeline terintegrasi on-premise lengkap:
    1. Upload & Parsing berbasis library on-premises (PyMuPDF / python-docx / openpyxl)
    2. Data Processing & Cleaning (Deduplikasi, normalisasi format, koreksi teks, boilerplate removal)
    3. Strategi Chunking structure-aware (256 token, 10% overlap)
    4. Penambahan metadata mandiri ({ sumber, tipe, halaman, klasifikasi })
    """
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        pages = _parse_file_by_path(
            file_path=tmp_path,
            classification=classification,
            clean=True
        )
        for p in pages:
            p.metadata["sumber"] = file.filename
            p.metadata["file_name"] = file.filename

        pipeline_chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = pipeline_chunker.chunk_pages(pages)

        return PipelineResponse(
            filename=file.filename,
            total_pages=len(pages),
            total_chunks=len(chunks),
            classification=classification,
            chunks=[DocumentChunkResponse(chunk_id=c.chunk_id, text=c.text, metadata=c.metadata) for c in chunks]
        )
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)

class UpdateIndexRequest(BaseModel):
    data_dir: Optional[str] = Field("./data/raw", description="Folder yang dipindai")
    dry_run: Optional[bool] = Field(False, description="Hanya cek status dokumen lama vs baru tanpa ubah DB")
    clean_deleted: Optional[bool] = Field(True, description="Hapus chunk dari dokumen yang sudah dihapus dari folder")
    force_all: Optional[bool] = Field(False, description="Paksa proses ulang semua dokumen")

@app.post("/api/v1/update-index", tags=["Incremental Index Update"])
def update_index_endpoint(req: UpdateIndexRequest):
    """Pembaruan inkremental database ChromaDB:
    Membedakan dokumen lama vs baru pada folder yang sama dan hanya memproses dokumen baru/berubah.
    """
    from scripts.update_index import (
        MANIFEST_DEFAULT_PATH,
        load_manifest,
        scan_folder_and_classify,
        run_incremental_update
    )

    target_path = Path(req.data_dir)
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Direktori {req.data_dir} tidak ditemukan.")

    manifest = load_manifest(MANIFEST_DEFAULT_PATH)
    supported_exts = settings.get("ingestion", {}).get(
        "supported_extensions", [".pdf", ".txt", ".md", ".docx", ".doc", ".xlsx", ".xls"]
    )

    new_f, mod_f, unch_f, del_f = scan_folder_and_classify(target_path, manifest, supported_exts)

    if req.dry_run:
        return {
            "status": "dry_run",
            "target_dir": str(target_path.resolve()),
            "summary": {
                "total_files": len(new_f) + len(mod_f) + len(unch_f),
                "new": len(new_f),
                "modified": len(mod_f),
                "unchanged": len(unch_f),
                "deleted": len(del_f)
            },
            "new_files": [f["name"] for f in new_f],
            "modified_files": [f["name"] for f in mod_f],
            "unchanged_files": [f["name"] for f in unch_f],
            "deleted_files": del_f
        }

    run_incremental_update(
        data_dir=target_path,
        dry_run=False,
        force_all=req.force_all or False,
        clean_deleted=req.clean_deleted if req.clean_deleted is not None else True,
        output_json=True
    )

    return {
        "status": "success",
        "target_dir": str(target_path.resolve()),
        "processed_new": len(new_f),
        "processed_modified": len(mod_f),
        "skipped_unchanged": len(unch_f),
        "cleaned_deleted": len(del_f) if req.clean_deleted else 0
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8080, reload=False)

