import hashlib
from typing import List, Dict, Any
from src.ingestion.parsers import DocumentPage
from src.processing.cleaner import TextCleaner
from config.settings import settings

class DocumentChunk:
    """Representasi satu potongan teks hasil chunking yang siap di-embed."""
    def __init__(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return f"<DocumentChunk id='{self.chunk_id}' len={len(self.text)} class='{self.metadata.get('classification')}'>"

class TextChunker:
    """Logika pemotongan (chunking) teks dokumen dengan preservasi konteks dan metadata."""

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        chunk_cfg = settings.get("ingestion", {}).get("chunking", {})
        self.chunk_size = chunk_size or chunk_cfg.get("chunk_size", 500)
        self.chunk_overlap = chunk_overlap or chunk_cfg.get("chunk_overlap", 50)
        self.separators = separators or chunk_cfg.get("separators", ["\n\n", "\n", ". ", " ", ""])
        self.cleaner = TextCleaner()

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursive splitting teks berdasarkan urutan separator."""
        final_chunks: List[str] = []
        separator = separators[-1]
        new_separators = []

        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator != "" else list(text)

        good_splits: List[str] = []
        for s in splits:
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if new_separators:
                    sub_splits = self._split_text(s, new_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.append(s)

        # Gabungkan split kecil dengan batas chunk_size dan chunk_overlap
        current_chunk: List[str] = []
        current_len = 0

        for piece in good_splits:
            piece_len = len(piece) + (len(separator) if current_chunk else 0)
            if current_len + piece_len > self.chunk_size and current_chunk:
                joined = separator.join(current_chunk).strip()
                if joined:
                    final_chunks.append(joined)
                
                # Pertahankan potongan terakhir untuk overlap
                overlap_len = 0
                overlap_chunk: List[str] = []
                for p in reversed(current_chunk):
                    if overlap_len + len(p) <= self.chunk_overlap:
                        overlap_chunk.insert(0, p)
                        overlap_len += len(p)
                    else:
                        break
                current_chunk = overlap_chunk
                current_len = sum(len(x) for x in current_chunk)

            current_chunk.append(piece)
            current_len += len(piece)

        if current_chunk:
            joined = separator.join(current_chunk).strip()
            if joined:
                final_chunks.append(joined)

        return final_chunks

    def chunk_document_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """Membersihkan dan memecah satu halaman dokumen menjadi daftar DocumentChunk."""
        cleaned_text = self.cleaner.sanitize(page.content)
        if not cleaned_text:
            return []

        text_splits = self._split_text(cleaned_text, self.separators)
        chunks: List[DocumentChunk] = []

        for idx, text_part in enumerate(text_splits):
            if not text_part.strip():
                continue

            # Buat hash unik deterministik berdasarkan sumber dan posisi
            source_file = page.metadata.get("file_name", "unknown")
            unique_seed = f"{source_file}_p{page.page_number}_c{idx}_{text_part[:30]}"
            chunk_hash = hashlib.sha256(unique_seed.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"doc_{chunk_hash}"

            # Copy metadata dan tambahkan informasi chunk
            meta = dict(page.metadata)
            meta["chunk_index"] = idx
            meta["char_length"] = len(text_part)

            chunks.append(DocumentChunk(chunk_id=chunk_id, text=text_part, metadata=meta))

        return chunks

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """Memproses sekumpulan DocumentPage menjadi sekumpulan DocumentChunk."""
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_document_page(page))
        return all_chunks

