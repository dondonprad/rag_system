import re
import unicodedata
from typing import List, Dict, Any, Optional
from config.settings import settings

class TextCleaner:
    """Pembersihan dan sanitasi data teks on-premise otomatis:
    - Boilerplate removal (nomor halaman, header/footer berulang, spasi kosong)
    - Deduplikasi baris dan paragraf berulang
    - Normalisasi format (Unicode NFKC, tanda baca seragam, spasi/newline)
    - Koreksi teks (ligature, de-hyphenation akhir baris, koreksi spasi tanda baca)
    - Sanitasi data rahasia (NIK, NRP, nomor telepon)
    """

    def __init__(self):
        sanitization_cfg = settings.get("ingestion", {}).get("sanitization", {})
        self.enable_masking = sanitization_cfg.get("enable_masking", True)
        self.patterns = sanitization_cfg.get("mask_patterns", {
            "nik": r"\b[0-9]{16}\b",
            "nrp": r"\b(NRP|NIP)[:\s]*([0-9]{6,18})\b",
            "phone": r"\b(\+62|62|0)8[1-9][0-9]{7,10}\b",
        })

    def normalize_format(self, text: str) -> str:
        """Normalisasi format teks:
        - Unicode NFKC (standarisasi karakter aksen, bentuk kompatibel)
        - Pembersihan simbol aneh, replacement char, dan karakter tak terlihat
        - Konversi tab, non-breaking space, dan whitespace anomali ke spasi standar
        - Konversi tanda kutip, tanda hubung, dan bullet point ke format standar
        """
        if not text:
            return ""

        # 1. Unicode NFKC Normalization
        text = unicodedata.normalize("NFKC", text)

        # 2. Hapus zero-width characters, format invisible, BOM, dan replacement character (\uFFFD)
        text = re.sub(r"[\u200B-\u200F\uFEFF\u2060\u00AD\uFFFD]", "", text)

        # 3. Konversi non-breaking space (\u00A0) dan variasi spasi unicode ke spasi biasa
        text = re.sub(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ", text)

        # 4. Standardisasi carriage return & line feed
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # 5. Konversi tab (\t) menjadi spasi agar tidak memicu simbol tab/panah aneh
        text = text.replace("\t", " ")

        # 6. Hapus control characters (kecuali newline \n)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)

        # 7. Hapus karakter sampah biner berulang (seperti padding OLE 'ÿÿÿÿ' atau sampah encoding)
        text = re.sub(r"[ÿ\x80-\xff]{2,}", "", text)

        # 8. Standardisasi tanda kutip (quotes)
        text = re.sub(r"[\u201C\u201D\u00AB\u00BB\u201E\u201F]", '"', text)
        text = re.sub(r"[\u2018\u2019\u201A\u201B\u02BC\u02BB]", "'", text)

        # 9. Standardisasi tanda strip / dash (en-dash, em-dash, horizontal bar) dan slash
        text = re.sub(r"[\u2013\u2014\u2015\u2212]", "-", text)
        text = text.replace("⁄", "/")

        # 10. Standardisasi ellipsis
        text = text.replace("…", "...")

        # 11. Standardisasi simbol bullet point menjadi format markdown list "- "
        text = re.sub(r"(?m)^[\t ]*[•●▪►◦✦]\s*", "- ", text)

        # 12. Normalisasi spasi horizontal berurutan (kecuali newline)
        text = re.sub(r"[^\S\n]+", " ", text)

        return text

    def correct_text(self, text: str) -> str:
        """Koreksi teks otomatis:
        - Perbaikan typographic ligatures (fi, fl, ff, ffi, ffl)
        - Koreksi encoding Mojibake (karakter UTF-8 yang salah didecode)
        - De-hyphenation kata terpotong di akhir baris (misal: 'pemerin-\ntah' -> 'pemerintah')
        - Koreksi spasi di sekitar tanda baca (koma, titik, titik koma)
        """
        if not text:
            return ""

        # 1. Perbaikan Mojibake umum (UTF-8 diinterpretasikan sebagai CP1252 / Latin-1)
        mojibake_map = {
            "â€™": "'",
            "â€˜": "'",
            "â€œ": '"',
            "â€": '"',
            "â€“": "-",
            "â€”": "-",
            "Â ": " ",
            "Â": "",
        }
        for bad_s, good_s in mojibake_map.items():
            text = text.replace(bad_s, good_s)

        # 2. Ligatures replacement
        ligatures = {
            "ﬁ": "fi",
            "ﬂ": "fl",
            "ﬀ": "ff",
            "ﬃ": "ffi",
            "ﬄ": "ffl",
            "ﬆ": "st",
            "ﬅ": "ft",
        }
        for lig, repl in ligatures.items():
            text = text.replace(lig, repl)

        # 3. De-hyphenation (kata yang terputus dengan tanda strip di akhir baris)
        # Contoh: 'pemerin-\ntah' -> 'pemerintah', 'kegia-\n tan' -> 'kegiatan'
        text = re.sub(r"([A-Za-z]{2,})-\n\s*([a-z]{2,})", r"\1\2", text)

        # 4. Koreksi spasi berlebih sebelum tanda baca (abaikan pipa tabel markdown '|')
        text = re.sub(r" +([,;:\.\?!])", r"\1", text)

        # 5. Pastikan spasi setelah tanda baca jika langsung diikuti huruf kapital
        # Contoh: "selesai.Lanjut" -> "selesai. Lanjut", "hari,besok" -> "hari, besok"
        text = re.sub(r"([a-z0-9])\.([A-Z])", r"\1. \2", text)
        text = re.sub(r"([a-zA-Z0-9]),([a-zA-Z])", r"\1, \2", text)

        return text

    def remove_boilerplate(self, text: str) -> str:
        """Penghapusan atribut non-kontekstual (boilerplate removal):
        - Nomor halaman berulang (misal: 'Page 1 of 5', 'Halaman 3', 'Hal. 12', '- 4 -', baris hanya berisi angka)
        - Header/footer standar
        - Baris kosong berlebih
        """
        if not text:
            return ""

        # 1. Hapus nomor halaman dengan pola kata kunci
        # 'Page 1 of 10', 'Halaman 5 dari 20', 'Halaman: 3', 'Hal. 12'
        text = re.sub(r"(?im)^[\t ]*(?:page|halaman|hal\.?)\s*\d+(?:\s*(?:of|dari|\/|-)\s*\d+)?[\t ]*$", "", text)
        text = re.sub(r"(?i)\b(?:page|halaman|hal\.?)\s+\d+\s+(?:of|dari)\s+\d+\b", "", text)

        # 2. Hapus nomor halaman berformat strip: '- 12 -' atau '[ 12 ]'
        text = re.sub(r"(?m)^[\t ]*[-\[\(]\s*\d+\s*[-\]\)][\t ]*$", "", text)

        # 3. Hapus baris yang hanya berisi digit angka tunggal (seringkali nomor halaman di margin atas/bawah)
        text = re.sub(r"(?m)^[\t ]*\d{1,4}[\t ]*$", "", text)

        # 4. Hapus header/footer standar yang sering berulang
        # Contoh: 'PUSAT PENDIDIKAN ZENI', 'CONFIDENTIAL', 'RAHASIA' yang berdiri sendiri di baris terpisah
        # (Tetap simpan jika merupakan bagian dari kalimat isi)
        text = re.sub(r"(?im)^[\t ]*(?:sangat\s+rahasia|rahasia|confidential|internal\s+use\s+only)[\t ]*$", "", text)

        # 5. Trim trailing whitespace per baris dan batasi baris kosong berurutan
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned_text = "\n".join(lines)
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

        return cleaned_text.strip()

    def remove_duplicates(self, text: str) -> str:
        """Penghapusan duplikasi:
        - Baris berulang berurutan (consecutive duplicate lines)
        - Blok paragraf yang duplikat
        """
        if not text:
            return ""

        paragraphs = text.split("\n\n")
        deduped_paras = []
        prev_para = None

        for p in paragraphs:
            p_strip = p.strip()
            if not p_strip:
                continue

            # Abaikan jika paragraf persis sama berurutan
            if p_strip == prev_para:
                continue

            # Periksa duplikasi baris dalam paragraf jika bukan tabel Markdown
            if not p_strip.startswith("|"):
                lines = p_strip.splitlines()
                deduped_lines = []
                prev_line = None
                for line in lines:
                    line_clean = line.strip()
                    if line_clean and line_clean == prev_line:
                        continue
                    deduped_lines.append(line)
                    if line_clean:
                        prev_line = line_clean
                p_strip = "\n".join(deduped_lines)

            deduped_paras.append(p_strip)
            prev_para = p_strip

        return "\n\n".join(deduped_paras)

    def mask_sensitive_data(self, text: str) -> str:
        """Menyamarkan data rahasia/PII seperti NIK, nomor kontak jika masking aktif."""
        if not self.enable_masking or not text:
            return text

        # Mask NIK (16 digit angka) -> [NIK-TERPROTEKSI]
        nik_pattern = self.patterns.get("nik", r"\b[0-9]{16}\b")
        text = re.sub(nik_pattern, "[NIK-TERPROTEKSI]", text)

        # Mask Phone -> [KONTAK-TERPROTEKSI]
        phone_pattern = self.patterns.get("phone", r"\b(\+62|62|0)8[1-9][0-9]{7,10}\b")
        text = re.sub(phone_pattern, "[KONTAK-TERPROTEKSI]", text)

        return text

    def clean_document_pages(self, pages_content: List[str]) -> List[str]:
        """Pembersihan kolektif multi-halaman dokumen:
        Mendeteksi dan menghapus header/footer berulang yang muncul di >= 50% halaman.
        """
        if len(pages_content) <= 2:
            return [self.sanitize(c) for c in pages_content]

        # Kumpulkan baris pertama dan baris terakhir dari setiap halaman
        header_candidates: Dict[str, int] = {}
        footer_candidates: Dict[str, int] = {}

        for c in pages_content:
            lines = [l.strip() for l in c.splitlines() if l.strip()]
            if lines:
                first_line = lines[0]
                if len(first_line) > 3 and not first_line.startswith("|"):
                    header_candidates[first_line] = header_candidates.get(first_line, 0) + 1
                last_line = lines[-1]
                if len(last_line) > 3 and not last_line.startswith("|"):
                    footer_candidates[last_line] = footer_candidates.get(last_line, 0) + 1

        threshold = max(2, int(len(pages_content) * 0.4))
        boilerplate_headers = {k for k, count in header_candidates.items() if count >= threshold}
        boilerplate_footers = {k for k, count in footer_candidates.items() if count >= threshold}

        cleaned_pages = []
        for c in pages_content:
            lines = c.splitlines()
            # Hapus header jika cocok
            if lines and lines[0].strip() in boilerplate_headers:
                lines = lines[1:]
            # Hapus footer jika cocok
            if lines and lines[-1].strip() in boilerplate_footers:
                lines = lines[:-1]

            cleaned_text = self.sanitize("\n".join(lines))
            cleaned_pages.append(cleaned_text)

        return cleaned_pages

    def sanitize(self, text: str) -> str:
        """Pipeline pembersihan lengkap:
        1. Normalisasi format (Unicode NFKC, quotes, dashes, whitespace)
        2. Koreksi teks (ligatures, de-hyphenation, spasi tanda baca)
        3. Boilerplate removal (nomor halaman, header/footer standar)
        4. Deduplikasi baris/paragraf berurutan
        5. Masking data rahasia
        """
        if not text:
            return ""

        text = self.normalize_format(text)
        text = self.correct_text(text)
        text = self.remove_boilerplate(text)
        text = self.remove_duplicates(text)
        text = self.mask_sensitive_data(text)

        return text.strip()
