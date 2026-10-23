import os
import re
import hashlib
import logging
from typing import List, Dict, Any, Optional
from src.ingestion.parsers import DocumentPage
from src.processing.cleaner import TextCleaner
from config.settings import settings

logger = logging.getLogger(__name__)

class DocumentChunk:
    """Representasi satu potongan teks hasil chunking dengan metadata mandiri."""
    def __init__(self, chunk_id: str, text: str, metadata: Dict[str, Any]):
        self.chunk_id = chunk_id
        self.text = text
        self.metadata = metadata

    def __repr__(self):
        return (
            f"<DocumentChunk id='{self.chunk_id}' sumber='{self.metadata.get('sumber')}' "
            f"hal={self.metadata.get('halaman')} tokens={self.metadata.get('token_count')} "
            f"klasifikasi='{self.metadata.get('klasifikasi')}'>"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }

class TokenizerHelper:
    """Helper tokenisasi on-premise:
    Menggunakan tokenizer nomic-embed-text lokal bila tersedia,
    dengan fallback otomatis berbasis regex/subword tokenizer.
    """
    _tokenizer = None

    @classmethod
    def get_tokenizer(cls):
        if cls._tokenizer is None:
            import glob
            # Cari file tokenizer.json lokal di cache HuggingFace
            cached_files = glob.glob(
                os.path.expanduser("~/.cache/huggingface/hub/**/tokenizer.json"),
                recursive=True
            )
            if cached_files:
                try:
                    from tokenizers import Tokenizer
                    cls._tokenizer = Tokenizer.from_file(cached_files[0])
                    logger.info(f"Loaded local tokenizer from: {cached_files[0]}")
                    return cls._tokenizer
                except Exception as e:
                    logger.warning(f"Gagal memuat tokenizer lokal dari file: {e}")

            # Fallback regex-based counter
            cls._tokenizer = False
        return cls._tokenizer

    @classmethod
    def count_tokens(cls, text: str) -> int:
        """Menghitung jumlah token dalam teks."""
        if not text:
            return 0
        tokenizer = cls.get_tokenizer()
        if tokenizer:
            try:
                encoded = tokenizer.encode(text)
                return len(encoded.ids)
            except Exception:
                pass
        # Fallback penghitungan token: rasio standar ~1.15 token per kata/simbol
        words_and_symbols = re.findall(r"\w+|[^\w\s]", text)
        return max(1, int(len(words_and_symbols) * 1.15))

    @classmethod
    def truncate_to_tokens(cls, text: str, max_tokens: int) -> str:
        """Memotong teks agar tidak melebihi max_tokens."""
        tokenizer = cls.get_tokenizer()
        if tokenizer:
            try:
                encoded = tokenizer.encode(text)
                if len(encoded.ids) <= max_tokens:
                    return text
                # Gunakan decoding token
                return tokenizer.decode(encoded.ids[:max_tokens])
            except Exception:
                pass
        words = text.split()
        target_words = int(max_tokens / 1.15)
        return " ".join(words[:target_words])

class TextChunker:
    """Chunker structure-aware:
    - Ukuran chunk default: 256 token.
    - Overlapping: 10% (sekitar 25-26 token).
    - Mempertahankan batas struktur dokumen: Bab, Bagian, Heading, Paragraf, dan Tabel Markdown.
    - Menempelkan metadata mandiri otomatis pada setiap DocumentChunk.
    """

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        overlap_ratio: Optional[float] = None
    ):
        chunk_cfg = settings.get("ingestion", {}).get("chunking", {})
        self.chunk_size = chunk_size or chunk_cfg.get("chunk_size", 256)
        
        # Hitung overlap 10% secara proporsional bila tidak diberikan secara eksplisit
        if chunk_overlap is not None:
            self.chunk_overlap = chunk_overlap
        else:
            ratio = overlap_ratio or chunk_cfg.get("overlap_ratio", 0.10)
            self.chunk_overlap = max(10, int(round(self.chunk_size * ratio)))

        self.cleaner = TextCleaner()
        self.tokenizer = TokenizerHelper()

    def _split_into_structural_units(self, text: str) -> List[Dict[str, Any]]:
        """Memecah teks menjadi unit-unit struktural terkecil (paragraf, judul bab, atau tabel).
        Setiap unit mencatat jenis struktur dan teks aslinya.
        """
        # Pola deteksi Bab/Heading/Pasal
        heading_pattern = re.compile(
            r"^(?:#{1,6}\s+.+|BAB\s+[IVXLCDM\d]+[^\n]*|PASAL\s+\d+[^\n]*|BAGIAN\s+[^\n]+|[0-9A-Z]\.\s+[A-Z\s]{3,})$",
            re.IGNORECASE | re.MULTILINE
        )

        raw_blocks = text.split("\n\n")
        units: List[Dict[str, Any]] = []

        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue

            # 1. Cek apakah ini tabel Markdown
            if block.startswith("|") and "\n|" in block:
                units.append({"type": "table", "text": block})
                continue

            # 2. Cek apakah ini judul bab/heading
            lines = block.splitlines()
            if len(lines) == 1 and heading_pattern.match(lines[0]):
                units.append({"type": "heading", "text": lines[0]})
                continue

            # 3. Paragraf biasa: jika terlalu panjang (> chunk_size token), pecah per kalimat
            block_tokens = self.tokenizer.count_tokens(block)
            if block_tokens <= self.chunk_size:
                units.append({"type": "paragraph", "text": block})
            else:
                # Pecah per kalimat (. ? ! \n)
                sentences = re.split(r"(?<=[.?!;\n])\s+", block)
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    s_tokens = self.tokenizer.count_tokens(s)
                    if s_tokens <= self.chunk_size:
                        units.append({"type": "sentence", "text": s})
                    else:
                        # Fallback jika 1 kalimat sangat panjang, potong berbasis kata/token
                        words = s.split()
                        curr_words = []
                        for w in words:
                            curr_words.append(w)
                            if self.tokenizer.count_tokens(" ".join(curr_words)) >= self.chunk_size:
                                units.append({"type": "sub_sentence", "text": " ".join(curr_words)})
                                curr_words = []
                        if curr_words:
                            units.append({"type": "sub_sentence", "text": " ".join(curr_words)})

        return units

    def _split_markdown_table_if_needed(self, table_text: str) -> List[str]:
        """Jika tabel Markdown lebih besar dari chunk_size, potong per baris data
        dengan tetap menyalin header baris pertama ke setiap potongan tabel.
        """
        table_tokens = self.tokenizer.count_tokens(table_text)
        if table_tokens <= self.chunk_size:
            return [table_text]

        lines = [l for l in table_text.splitlines() if l.strip()]
        if len(lines) < 3:
            return [table_text]

        header = lines[0]
        sep = lines[1]
        data_rows = lines[2:]

        header_prefix = f"{header}\n{sep}\n"
        header_tokens = self.tokenizer.count_tokens(header_prefix)

        table_chunks: List[str] = []
        current_rows: List[str] = []
        current_tokens = header_tokens

        for row in data_rows:
            r_tokens = self.tokenizer.count_tokens(row)
            if current_tokens + r_tokens > self.chunk_size and current_rows:
                table_chunks.append(header_prefix + "\n".join(current_rows))
                current_rows = [row]
                current_tokens = header_tokens + r_tokens
            else:
                current_rows.append(row)
                current_tokens += r_tokens

        if current_rows:
            table_chunks.append(header_prefix + "\n".join(current_rows))

        return table_chunks

    def chunk_document_page(self, page: DocumentPage) -> List[DocumentChunk]:
        """Membersihkan halaman dan memotong teks secara structure-aware
        menjadi kumpulan DocumentChunk 256 token dengan 10% overlap.
        """
        # 1. Bersihkan konten halaman
        cleaned_text = self.cleaner.sanitize(page.content)
        if not cleaned_text:
            return []

        # 2. Pecah ke unit-unit struktural (bab, heading, paragraf, tabel)
        structural_units = self._split_into_structural_units(cleaned_text)
        if not structural_units:
            return []

        chunks: List[DocumentChunk] = []
        current_unit_texts: List[str] = []
        current_token_count = 0
        current_chapter: Optional[str] = None

        def create_chunk(text_parts: List[str], chapter_name: Optional[str], c_idx: int) -> Optional[DocumentChunk]:
            chunk_content = "\n\n".join(text_parts).strip()
            if not chunk_content:
                return None

            t_count = self.tokenizer.count_tokens(chunk_content)
            source_file = page.metadata.get("sumber", page.metadata.get("file_name", "unknown"))
            unique_seed = f"{source_file}_p{page.page_number}_c{c_idx}_{chunk_content[:40]}"
            chunk_hash = hashlib.sha256(unique_seed.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"doc_{chunk_hash}"

            # Duplikasi metadata halaman dan tambahkan atribut chunk mandiri
            meta = dict(page.metadata)
            meta.update({
                "sumber": page.metadata.get("sumber", source_file),
                "tipe": page.metadata.get("tipe", page.metadata.get("file_type", "txt")),
                "halaman": page.page_number,
                "klasifikasi": page.metadata.get("klasifikasi", "internal"),
                "classification": page.metadata.get("classification", page.metadata.get("klasifikasi", "INTERNAL").upper()),
                "chunk_index": c_idx,
                "token_count": t_count,
                "char_length": len(chunk_content),
                "judul_bab": chapter_name or meta.get("judul_bab", "UMUM"),
            })

            return DocumentChunk(chunk_id=chunk_id, text=chunk_content, metadata=meta)

        chunk_counter = 0

        for unit in structural_units:
            u_text = unit["text"]
            u_type = unit["type"]

            if u_type == "heading":
                current_chapter = u_text

            # Tangani tabel besar jika melebihi ukuran chunk
            if u_type == "table":
                sub_tables = self._split_markdown_table_if_needed(u_text)
                for st in sub_tables:
                    st_tokens = self.tokenizer.count_tokens(st)
                    if current_token_count + st_tokens > self.chunk_size and current_unit_texts:
                        c = create_chunk(current_unit_texts, current_chapter, chunk_counter)
                        if c:
                            chunks.append(c)
                            chunk_counter += 1

                        # Hitung overlap dari akhir chunk sebelumnya
                        overlap_units = self._extract_overlap_units(current_unit_texts)
                        current_unit_texts = overlap_units + [st]
                        current_token_count = self.tokenizer.count_tokens("\n\n".join(current_unit_texts))
                    else:
                        current_unit_texts.append(st)
                        current_token_count += st_tokens
                continue

            u_tokens = self.tokenizer.count_tokens(u_text)

            # Jika penambahan unit ini melebihi batas 256 token
            if current_token_count + u_tokens > self.chunk_size and current_unit_texts:
                c = create_chunk(current_unit_texts, current_chapter, chunk_counter)
                if c:
                    chunks.append(c)
                    chunk_counter += 1

                # Hitung overlap struktural (10% ~ 25-26 token)
                overlap_units = self._extract_overlap_units(current_unit_texts)
                current_unit_texts = overlap_units + [u_text]
                current_token_count = self.tokenizer.count_tokens("\n\n".join(current_unit_texts))
            else:
                current_unit_texts.append(u_text)
                current_token_count += u_tokens

        # Sisa unit terakhir
        if current_unit_texts:
            c = create_chunk(current_unit_texts, current_chapter, chunk_counter)
            if c:
                chunks.append(c)

        return chunks

    def _extract_overlap_units(self, text_units: List[str]) -> List[str]:
        """Mengambil unit dari akhir chunk sebelumnya hingga mencapai ~10% (chunk_overlap) token."""
        overlap_units: List[str] = []
        tokens_acc = 0

        for u in reversed(text_units):
            # Jangan memasukkan tabel besar sebagai overlap
            if u.startswith("|") and "\n|" in u:
                continue

            u_tok = self.tokenizer.count_tokens(u)
            if tokens_acc + u_tok <= self.chunk_overlap or not overlap_units:
                overlap_units.insert(0, u)
                tokens_acc += u_tok
            else:
                break

        return overlap_units

    def chunk_pages(self, pages: List[DocumentPage]) -> List[DocumentChunk]:
        """Memproses sekumpulan DocumentPage menjadi sekumpulan DocumentChunk terstruktur."""
        all_chunks: List[DocumentChunk] = []
        for page in pages:
            all_chunks.extend(self.chunk_document_page(page))
        return all_chunks
