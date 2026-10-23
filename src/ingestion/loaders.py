import logging
from pathlib import Path
from typing import List, Optional
from src.ingestion.parsers import DocumentParser, DocumentPage
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Loader dokumen lokal PUSDIKZI dari direktori lokal atau data/raw/."""

    def __init__(self, raw_dir: Optional[str] = None):
        self.raw_dir = Path(raw_dir or settings.get("ingestion", {}).get("data_raw_dir", "./data/raw"))
        self.supported_extensions = settings.get("ingestion", {}).get(
            "supported_extensions", [".pdf", ".txt", ".md", ".docx", ".doc", ".xlsx", ".xls"]
        )
        self.default_classification = settings.get("security", {}).get(
            "default_classification", "INTERNAL"
        )

    def load_file(self, file_path: Path, classification: Optional[str] = None) -> List[DocumentPage]:
        """Memuat satu file berdasarkan ekstensi dan menyematkan metadata mandiri secara otomatis."""
        ext = file_path.suffix.lower()
        if ext not in self.supported_extensions:
            logger.warning(f"Ekstensi {ext} tidak didukung: {file_path.name}")
            return []

        doc_classification = (classification or self.default_classification).lower()

        try:
            if ext == ".pdf":
                pages = DocumentParser.parse_pdf(file_path, classification=doc_classification)
            elif ext == ".txt":
                pages = DocumentParser.parse_text(file_path, classification=doc_classification)
            elif ext == ".md":
                pages = DocumentParser.parse_markdown(file_path, classification=doc_classification)
            elif ext == ".docx":
                pages = DocumentParser.parse_docx(file_path, classification=doc_classification)
            elif ext == ".doc":
                pages = DocumentParser.parse_doc(file_path, classification=doc_classification)
            elif ext == ".xlsx":
                pages = DocumentParser.parse_xlsx(file_path, classification=doc_classification)
            elif ext == ".xls":
                pages = DocumentParser.parse_xls(file_path, classification=doc_classification)
            else:
                return []

            return pages
        except Exception as e:
            logger.error(f"Gagal memproses file {file_path}: {e}")
            return []

    def load_directory(self, target_dir: Optional[Path] = None) -> List[DocumentPage]:
        """Memindai seluruh dokumen di dalam direktori."""
        dir_to_scan = target_dir or self.raw_dir
        if not dir_to_scan.exists():
            logger.warning(f"Direktori {dir_to_scan} tidak ditemukan, membuat direktori baru.")
            dir_to_scan.mkdir(parents=True, exist_ok=True)
            return []

        all_pages: List[DocumentPage] = []
        files = [p for p in dir_to_scan.rglob("*") if p.is_file() and p.suffix.lower() in self.supported_extensions]
        logger.info(f"Ditemukan {len(files)} file dokumen di {dir_to_scan}")

        # Cek apakah nama direktori itu sendiri adalah nama klasifikasi (misal: ./data/raw/SECRET)
        folder_class = None
        if dir_to_scan.name.upper() in ["PUBLIC", "INTERNAL", "RESTRICTED", "CONFIDENTIAL", "SECRET"]:
            folder_class = dir_to_scan.name.upper()

        for file_path in files:
            # Periksa klasifikasi dari folder utama atau subfolder (misal: SECRET/...)
            classification = folder_class or self.default_classification
            try:
                rel_parts = file_path.relative_to(dir_to_scan).parts
                if len(rel_parts) > 1:
                    potential_class = rel_parts[0].upper()
                    if potential_class in ["PUBLIC", "INTERNAL", "RESTRICTED", "CONFIDENTIAL", "SECRET"]:
                        classification = potential_class
            except ValueError:
                pass

            pages = self.load_file(file_path, classification=classification)
            all_pages.extend(pages)
            logger.info(f"Dimuat {len(pages)} bagian/halaman dari {file_path.name} (Akses: {classification})")

        return all_pages
