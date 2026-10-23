import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class DocumentPage:
    """Representasi satu halaman atau bagian dokumen dengan metadata terstruktur."""
    def __init__(self, content: str, page_number: int, metadata: Dict[str, Any]):
        self.content = content
        self.page_number = page_number
        self.metadata = metadata

    def __repr__(self):
        return (
            f"<DocumentPage sumber='{self.metadata.get('sumber')}' "
            f"tipe='{self.metadata.get('tipe')}' hal={self.page_number} len={len(self.content)}>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "page_number": self.page_number,
            "metadata": self.metadata
        }

class DocumentParser:
    """Parser dokumen lokal on-premise PUSDIKZI:
    - PDF: PyMuPDF dengan ekstraksi tabel otomatis ke format Markdown.
    - DOCX: python-docx dengan ekstraksi sekuensial paragraf dan tabel ke format Markdown.
    - XLSX / XLS: openpyxl / xlrd dengan konversi tabel ke format Markdown dan representasi naratif baris.
    - TXT / MD: Ekstraksi teks berbasis stream berkeamanan tinggi.
    - Otomatisasi penyisipan metadata mandiri pada setiap halaman/bagian.
    """

    @staticmethod
    def _create_metadata(
        file_path: Path,
        file_type: str,
        page_number: int,
        total_pages: int,
        classification: str,
        extra: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Menyisipkan metadata mandiri standar secara otomatis."""
        meta = {
            "sumber": file_path.name,
            "tipe": file_type.lower().replace(".", ""),
            "halaman": page_number,
            "klasifikasi": classification.lower(),
            # Atribut pelengkap untuk kompatibilitas sistem & ChromaDB RBAC
            "classification": classification.upper(),
            "total_halaman": total_pages,
            "file_name": file_path.name,
            "file_type": file_type.lower().replace(".", ""),
            "page": page_number,
            "source": str(file_path.resolve()),
            "source_path": str(file_path.resolve()),
        }
        if extra:
            meta.update(extra)
        return meta

    @staticmethod
    def parse_pdf(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Ekstraksi dokumen PDF menggunakan PyMuPDF (fitz).
        Semua tabel diidentifikasi dan diubah ke format Markdown tabel agar
        struktur baris dan kolom dipahami secara optimal oleh model embedding.
        """
        import pymupdf

        pages: List[DocumentPage] = []
        doc = pymupdf.open(str(file_path))
        total_pages = len(doc)

        for idx, page in enumerate(doc):
            page_num = idx + 1
            has_tables = False
            table_count = 0
            page_content = ""

            try:
                # 1. Deteksi tabel pada halaman PDF menggunakan PyMuPDF Layout/Table finder
                tabs = page.find_tables()
                if tabs and tabs.tables:
                    has_tables = True
                    table_count = len(tabs.tables)

                    # Kumpulkan bounding box dan teks Markdown tabel
                    tables_data = []
                    for t in tabs:
                        try:
                            md_table = t.to_markdown().strip()
                            if md_table:
                                tables_data.append((t.bbox, md_table))
                        except Exception as te:
                            logger.warning(f"Gagal konversi tabel PDF ke markdown di {file_path.name} hal {page_num}: {te}")

                    # 2. Ekstrak blok teks di luar area tabel agar tidak duplikat
                    blocks = page.get_text("blocks")
                    ordered_elements = []

                    for b in blocks:
                        # b = (x0, y0, x1, y1, text, block_no, block_type)
                        # block_type 0 = text, 1 = image
                        if len(b) >= 7 and b[6] != 0:
                            continue

                        b_rect = pymupdf.Rect(b[:4])
                        b_text = b[4].strip()
                        if not b_text:
                            continue

                        # Periksa apakah blok teks ini berada di dalam batas tabel
                        inside_table = False
                        for t_bbox, _ in tables_data:
                            t_rect = pymupdf.Rect(t_bbox)
                            intersect = b_rect & t_rect
                            if not intersect.is_empty and (intersect.get_area() > 0.3 * b_rect.get_area()):
                                inside_table = True
                                break

                        if not inside_table:
                            ordered_elements.append((b[1], b_text))

                    # Masukkan representasi Markdown tabel ke urutan visual dokumen
                    for t_bbox, md_table in tables_data:
                        ordered_elements.append((t_bbox[1], md_table))

                    # Urutkan berdasarkan koordinat vertikal (y0) dari atas ke bawah
                    ordered_elements.sort(key=lambda x: x[0])
                    page_content = "\n\n".join(elem[1] for elem in ordered_elements).strip()
                else:
                    # Bila tidak terdeteksi tabel, gunakan ekstraksi teks standar
                    page_content = page.get_text("text").strip()

            except Exception as e:
                logger.warning(f"Fallback ekstraksi teks biasa pada {file_path.name} hal {page_num}: {e}")
                page_content = page.get_text("text").strip()

            if page_content:
                metadata = DocumentParser._create_metadata(
                    file_path=file_path,
                    file_type="pdf",
                    page_number=page_num,
                    total_pages=total_pages,
                    classification=classification,
                    extra={
                        "has_tables": has_tables,
                        "table_count": table_count,
                        "char_length": len(page_content)
                    }
                )
                pages.append(DocumentPage(content=page_content, page_number=page_num, metadata=metadata))

        doc.close()
        return pages

    @staticmethod
    def _table_to_markdown(rows_data: List[List[str]]) -> str:
        """Helper untuk membuat tabel Markdown valid dari list of rows."""
        if not rows_data:
            return ""

        # Hitung jumlah kolom maksimum
        num_cols = max(len(row) for row in rows_data)
        if num_cols == 0:
            return ""

        # Pad setiap baris agar jumlah kolom seragam
        padded_rows = []
        for r in rows_data:
            row_clean = [str(cell).strip().replace("\n", " ").replace("|", "/") for cell in r]
            padded_rows.append(row_clean + [""] * (num_cols - len(row_clean)))

        header_row = padded_rows[0]
        # Pastikan header tidak kosong semua
        if not any(header_row):
            header_row = [f"Kolom_{i + 1}" for i in range(num_cols)]

        header_line = "| " + " | ".join(header_row) + " |"
        sep_line = "| " + " | ".join(["---"] * num_cols) + " |"
        body_lines = ["| " + " | ".join(r) + " |" for r in padded_rows[1:]]

        return "\n".join([header_line, sep_line] + body_lines)

    @staticmethod
    def parse_docx(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Mengekstrak file Microsoft Word (.docx) secara terstruktur.
        Paragraf dan tabel diekstrak secara sekuensial mengikuti struktur dokumen asli.
        Setiap tabel diubah ke format Markdown lengkap.
        """
        import docx
        from docx.oxml.text.paragraph import CT_P
        from docx.oxml.table import CT_Tbl
        from docx.text.paragraph import Paragraph
        from docx.table import Table

        doc = docx.Document(file_path)
        content_parts: List[str] = []
        table_count = 0

        # Iterasi sekuensial melalui elemen body dokumen untuk mempertahankan hierarki asli
        for child in doc.element.body:
            if isinstance(child, CT_P):
                p = Paragraph(child, doc)
                p_text = p.text.strip()
                if p_text:
                    content_parts.append(p_text)

            elif isinstance(child, CT_Tbl):
                t = Table(child, doc)
                table_count += 1
                rows_data = []

                for row in t.rows:
                    row_vals = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                    # Hilangkan duplikasi sel gabungan (merged cells) berurutan
                    deduped_vals = []
                    for v in row_vals:
                        if not deduped_vals or v != deduped_vals[-1]:
                            deduped_vals.append(v)
                    if any(deduped_vals):
                        rows_data.append(deduped_vals)

                if rows_data:
                    md_table = DocumentParser._table_to_markdown(rows_data)
                    if md_table:
                        content_parts.append(md_table)

        full_content = "\n\n".join(content_parts).strip()
        if not full_content:
            return []

        metadata = DocumentParser._create_metadata(
            file_path=file_path,
            file_type="docx",
            page_number=1,
            total_pages=1,
            classification=classification,
            extra={
                "has_tables": table_count > 0,
                "table_count": table_count,
                "char_length": len(full_content)
            }
        )

        return [DocumentPage(content=full_content, page_number=1, metadata=metadata)]

    @staticmethod
    def parse_xlsx(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Mengekstrak lembar kerja spreadsheet Excel modern (.xlsx) ke format terstruktur:
        1. Tabel Markdown untuk merepresentasikan baris dan kolom tabular.
        2. Teks berbasis baris terstruktur: 'Pada baris X, Kolom Nama adalah X dan Nilai adalah Y'
           agar relasi entitas dapat dipahami optimal oleh model embedding vector.
        """
        import openpyxl

        wb = openpyxl.load_workbook(str(file_path), data_only=True, read_only=True)
        pages: List[DocumentPage] = []
        total_sheets = len(wb.sheetnames)

        for s_idx, sheet_name in enumerate(wb.sheetnames):
            sheet = wb[sheet_name]
            raw_rows = []

            for row in sheet.iter_rows(values_only=True):
                # Filter baris yang benar-benar kosong
                if any(val is not None and str(val).strip() != "" for val in row):
                    row_vals = [str(val).strip() if val is not None else "" for val in row]
                    raw_rows.append(row_vals)

            if not raw_rows:
                continue

            # Header kolom diambil dari baris pertama yang valid
            header_row = raw_rows[0]
            col_names = []
            for col_i, h in enumerate(header_row):
                name = h if h else f"Kolom_{col_i + 1}"
                col_names.append(name)

            # 1. Format Tabel Markdown
            md_table = DocumentParser._table_to_markdown(raw_rows)

            # 2. Format Teks Terstruktur Berbasis Baris (Entity-Attribute narrative)
            narrative_lines = []
            for r_idx, row_data in enumerate(raw_rows[1:], start=1):
                pairs = []
                for c_idx, val in enumerate(row_data):
                    if c_idx < len(col_names) and val != "":
                        col_label = col_names[c_idx]
                        pairs.append(f"Kolom {col_label} adalah {val}")
                if pairs:
                    narrative_lines.append(f"Pada baris {r_idx}, " + " dan ".join(pairs) + ".")

            narrative_block = "\n".join(narrative_lines)

            # Gabungkan menjadi konten terstruktur per lembar kerja
            structured_content = (
                f"[LEMBAR KERJA: {sheet_name}]\n\n"
                f"### Struktur Tabel:\n{md_table}\n\n"
                f"### Representasi Terstruktur:\n{narrative_block}"
            ).strip()

            metadata = DocumentParser._create_metadata(
                file_path=file_path,
                file_type="xlsx",
                page_number=s_idx + 1,
                total_pages=total_sheets,
                classification=classification,
                extra={
                    "sheet_name": sheet_name,
                    "row_count": len(raw_rows),
                    "col_count": len(col_names),
                    "char_length": len(structured_content)
                }
            )

            pages.append(DocumentPage(content=structured_content, page_number=s_idx + 1, metadata=metadata))

        wb.close()
        return pages

    @staticmethod
    def parse_xls(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Mengekstrak lembar kerja Excel 97-2003 (.xls) ke format tabel Markdown dan teks naratif baris."""
        import xlrd

        wb = xlrd.open_workbook(str(file_path), formatting_info=False)
        pages: List[DocumentPage] = []
        total_sheets = wb.nsheets

        for s_idx in range(total_sheets):
            sheet = wb.sheet_by_index(s_idx)
            raw_rows = []

            for rx in range(sheet.nrows):
                row_vals = sheet.row_values(rx)
                if any(v is not None and str(v).strip() != "" for v in row_vals):
                    cleaned_vals = [str(v).strip() if v is not None else "" for v in row_vals]
                    raw_rows.append(cleaned_vals)

            if not raw_rows:
                continue

            header_row = raw_rows[0]
            col_names = [h if h else f"Kolom_{i + 1}" for i, h in enumerate(header_row)]

            # 1. Tabel Markdown
            md_table = DocumentParser._table_to_markdown(raw_rows)

            # 2. Narasi baris terstruktur
            narrative_lines = []
            for r_idx, row_data in enumerate(raw_rows[1:], start=1):
                pairs = []
                for c_idx, val in enumerate(row_data):
                    if c_idx < len(col_names) and val != "":
                        col_label = col_names[c_idx]
                        pairs.append(f"Kolom {col_label} adalah {val}")
                if pairs:
                    narrative_lines.append(f"Pada baris {r_idx}, " + " dan ".join(pairs) + ".")

            narrative_block = "\n".join(narrative_lines)

            structured_content = (
                f"[LEMBAR KERJA: {sheet.name}]\n\n"
                f"### Struktur Tabel:\n{md_table}\n\n"
                f"### Representasi Terstruktur:\n{narrative_block}"
            ).strip()

            metadata = DocumentParser._create_metadata(
                file_path=file_path,
                file_type="xls",
                page_number=s_idx + 1,
                total_pages=total_sheets,
                classification=classification,
                extra={
                    "sheet_name": sheet.name,
                    "row_count": len(raw_rows),
                    "col_count": len(col_names),
                    "char_length": len(structured_content)
                }
            )

            pages.append(DocumentPage(content=structured_content, page_number=s_idx + 1, metadata=metadata))

        return pages

    @staticmethod
    def parse_doc(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Mengekstrak teks dari file Microsoft Word 97-2003 OLE (.doc)."""
        import olefile

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
                if len(decoded) >= 4 and not re.match(r"^[\x00-\x1f\x7f]+$", decoded):
                    cleaned_segments.append(decoded)

            full_content = "\n".join(cleaned_segments).strip()
            if not full_content:
                return []

            metadata = DocumentParser._create_metadata(
                file_path=file_path,
                file_type="doc",
                page_number=1,
                total_pages=1,
                classification=classification,
                extra={"char_length": len(full_content)}
            )

            return [DocumentPage(content=full_content, page_number=1, metadata=metadata)]
        except Exception as e:
            raise RuntimeError(f"Gagal mengekstrak teks dari file .doc ({file_path.name}): {e}")

    @staticmethod
    def parse_text(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Mengekstrak file plain text (.txt)."""
        for enc in ["utf-8", "latin1", "cp1252"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        else:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

        if not content.strip():
            return []

        metadata = DocumentParser._create_metadata(
            file_path=file_path,
            file_type="txt",
            page_number=1,
            total_pages=1,
            classification=classification,
            extra={"char_length": len(content)}
        )

        return [DocumentPage(content=content.strip(), page_number=1, metadata=metadata)]

    @staticmethod
    def parse_markdown(file_path: Path, classification: str = "internal") -> List[DocumentPage]:
        """Mengekstrak file Markdown (.md)."""
        pages = DocumentParser.parse_text(file_path, classification=classification)
        for p in pages:
            p.metadata["tipe"] = "md"
            p.metadata["file_type"] = "md"
        return pages
