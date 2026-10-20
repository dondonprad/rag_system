import os
import re
from pathlib import Path
from typing import List, Dict, Any

class DocumentPage:
    """Representasi satu halaman atau bagian dokumen."""
    def __init__(self, content: str, page_number: int, metadata: Dict[str, Any]):
        self.content = content
        self.page_number = page_number
        self.metadata = metadata

    def __repr__(self):
        return f"<DocumentPage file='{self.metadata.get('file_name')}' page={self.page_number} len={len(self.content)}>"

class DocumentParser:
    """Parser dokumen on-premise untuk PDF, TXT, MD, DOCX, DOC, XLSX, XLS tanpa koneksi cloud (Zero Data Leakage)."""

    @staticmethod
    def parse_pdf(file_path: Path) -> List[DocumentPage]:
        pages: List[DocumentPage] = []
        file_name = file_path.name
        
        # Coba PyMuPDF (fitz) terlebih dahulu
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(file_path)
            for idx, page in enumerate(doc):
                text = page.get_text()
                if text.strip():
                    pages.append(DocumentPage(
                        content=text,
                        page_number=idx + 1,
                        metadata={
                            "source": str(file_path.resolve()),
                            "file_name": file_name,
                            "file_type": "pdf",
                            "page": idx + 1,
                            "total_pages": len(doc)
                        }
                    ))
            doc.close()
            return pages
        except ImportError:
            pass

        # Fallback ke pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            total_pages = len(reader.pages)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append(DocumentPage(
                        content=text,
                        page_number=idx + 1,
                        metadata={
                            "source": str(file_path.resolve()),
                            "file_name": file_name,
                            "file_type": "pdf",
                            "page": idx + 1,
                            "total_pages": total_pages
                        }
                    ))
            return pages
        except ImportError:
            raise ImportError("PyMuPDF (fitz) atau pypdf diperlukan untuk memproses file PDF.")

    @staticmethod
    def parse_text(file_path: Path) -> List[DocumentPage]:
        file_name = file_path.name
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return []

        return [
            DocumentPage(
                content=content,
                page_number=1,
                metadata={
                    "source": str(file_path.resolve()),
                    "file_name": file_name,
                    "file_type": "txt",
                    "page": 1,
                    "total_pages": 1
                }
            )
        ]

    @staticmethod
    def parse_markdown(file_path: Path) -> List[DocumentPage]:
        file_name = file_path.name
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if not content.strip():
            return []

        return [
            DocumentPage(
                content=content,
                page_number=1,
                metadata={
                    "source": str(file_path.resolve()),
                    "file_name": file_name,
                    "file_type": "md",
                    "page": 1,
                    "total_pages": 1
                }
            )
        ]

    @staticmethod
    def parse_docx(file_path: Path) -> List[DocumentPage]:
        """Mengekstrak teks dan tabel dari file Microsoft Word modern (.docx)."""
        import docx
        doc = docx.Document(file_path)
        file_name = file_path.name

        text_lines = []
        # Ambil paragraf
        for p in doc.paragraphs:
            clean_p = p.text.strip()
            if clean_p:
                text_lines.append(clean_p)

        # Ambil tabel
        for t_idx, table in enumerate(doc.tables):
            table_lines = [f"[TABEL {t_idx + 1}]"]
            for row in table.rows:
                row_vals = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                # Hilangkan duplikasi dari sel gabungan
                unique_vals = []
                for val in row_vals:
                    if not unique_vals or val != unique_vals[-1]:
                        unique_vals.append(val)
                if any(unique_vals):
                    table_lines.append(" | ".join(unique_vals))
            if len(table_lines) > 1:
                text_lines.append("\n".join(table_lines))

        full_content = "\n\n".join(text_lines)
        if not full_content.strip():
            return []

        return [
            DocumentPage(
                content=full_content,
                page_number=1,
                metadata={
                    "source": str(file_path.resolve()),
                    "file_name": file_name,
                    "file_type": "docx",
                    "page": 1,
                    "total_pages": 1
                }
            )
        ]

    @staticmethod
    def parse_doc(file_path: Path) -> List[DocumentPage]:
        """Mengekstrak teks dari file Microsoft Word 97-2003 OLE (.doc)."""
        import olefile
        file_name = file_path.name
        
        try:
            ole = olefile.OleFileIO(str(file_path))
            if not ole.exists("WordDocument"):
                return []
            
            stream = ole.openstream("WordDocument")
            data = stream.read()
            ole.close()

            # Ekstrak rangkaian teks yang valid (CP1252 / ASCII)
            raw_segments = re.findall(b"[\x20-\x7e\r\n\t\xa0-\xff]{4,}", data)
            cleaned_segments = []
            for seg in raw_segments:
                decoded = seg.decode("latin1", errors="ignore").strip()
                # Filter noise biner/metadata non-teks
                if len(decoded) >= 4 and not re.match(r"^[\x00-\x1f\x7f]+$", decoded):
                    cleaned_segments.append(decoded)

            full_content = "\n".join(cleaned_segments)
            if not full_content.strip():
                return []

            return [
                DocumentPage(
                    content=full_content,
                    page_number=1,
                    metadata={
                        "source": str(file_path.resolve()),
                        "file_name": file_name,
                        "file_type": "doc",
                        "page": 1,
                        "total_pages": 1
                    }
                )
            ]
        except Exception as e:
            raise RuntimeError(f"Gagal mengekstrak teks dari file .doc ({file_name}): {e}")

    @staticmethod
    def parse_xlsx(file_path: Path) -> List[DocumentPage]:
        """Mengekstrak setiap lembar kerja (sheet) file Microsoft Excel modern (.xlsx)."""
        import openpyxl
        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        file_name = file_path.name
        pages: List[DocumentPage] = []

        total_sheets = len(wb.sheetnames)
        for s_idx, sheet_name in enumerate(wb.sheetnames):
            sheet = wb[sheet_name]
            lines = [f"[SHEET: {sheet_name}]"]

            for row in sheet.iter_rows(values_only=True):
                non_empty = [str(val).strip() for val in row if val is not None and str(val).strip() != ""]
                if non_empty:
                    lines.append(" | ".join(non_empty))

            sheet_text = "\n".join(lines)
            if len(lines) > 1:
                pages.append(DocumentPage(
                    content=sheet_text,
                    page_number=s_idx + 1,
                    metadata={
                        "source": str(file_path.resolve()),
                        "file_name": file_name,
                        "file_type": "xlsx",
                        "sheet_name": sheet_name,
                        "page": s_idx + 1,
                        "total_pages": total_sheets
                    }
                ))

        wb.close()
        return pages

    @staticmethod
    def parse_xls(file_path: Path) -> List[DocumentPage]:
        """Mengekstrak setiap lembar kerja file Microsoft Excel 97-2003 (.xls)."""
        import xlrd
        wb = xlrd.open_workbook(str(file_path), formatting_info=False)
        file_name = file_path.name
        pages: List[DocumentPage] = []

        total_sheets = wb.nsheets
        for s_idx in range(total_sheets):
            sheet = wb.sheet_by_index(s_idx)
            lines = [f"[SHEET: {sheet.name}]"]

            for rx in range(sheet.nrows):
                row_vals = sheet.row_values(rx)
                non_empty = [str(v).strip() for v in row_vals if v is not None and str(v).strip() != ""]
                if non_empty:
                    lines.append(" | ".join(non_empty))

            sheet_text = "\n".join(lines)
            if len(lines) > 1:
                pages.append(DocumentPage(
                    content=sheet_text,
                    page_number=s_idx + 1,
                    metadata={
                        "source": str(file_path.resolve()),
                        "file_name": file_name,
                        "file_type": "xls",
                        "sheet_name": sheet.name,
                        "page": s_idx + 1,
                        "total_pages": total_sheets
                    }
                ))

        return pages
