import re
from typing import Dict, Any
from config.settings import settings

class TextCleaner:
    """Sanitasi dan pembersihan teks sebelum embedding untuk menjamin privasi dan akurasi."""

    def __init__(self):
        sanitization_cfg = settings.get("ingestion", {}).get("sanitization", {})
        self.enable_masking = sanitization_cfg.get("enable_masking", True)
        self.patterns = sanitization_cfg.get("mask_patterns", {
            "nik": r"\b[0-9]{16}\b",
            "nrp": r"\b(NRP|NIP)[:\s]*([0-9]{6,18})\b",
            "phone": r"\b(\+62|62|0)8[1-9][0-9]{7,10}\b",
        })

    def clean_whitespace(self, text: str) -> str:
        """Menghapus karakter kontrol anomali dan menormalkan spasi/baris baru."""
        # Hilangkan zero-width spaces and control chars
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        # Normalkan multiple tabs/spaces menjadi spasi tunggal, tapi pertahankan newline
        text = re.sub(r"[^\S\r\n]+", " ", text)
        # Batasi multiple newline beruntun maksimal 2 baris baru
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def mask_sensitive_data(self, text: str) -> str:
        """Menyamarkan data rahasia/PII seperti NIK, nomor kontak jika masking aktif."""
        if not self.enable_masking:
            return text

        # Mask NIK (16 digit angka) -> [NIK-TERPROTEKSI]
        nik_pattern = self.patterns.get("nik", r"\b[0-9]{16}\b")
        text = re.sub(nik_pattern, "[NIK-TERPROTEKSI]", text)

        # Mask Phone -> [KONTAK-TERPROTEKSI]
        phone_pattern = self.patterns.get("phone", r"\b(\+62|62|0)8[1-9][0-9]{7,10}\b")
        text = re.sub(phone_pattern, "[KONTAK-TERPROTEKSI]", text)

        return text

    def sanitize(self, text: str) -> str:
        """Pipeline pembersihan lengkap."""
        cleaned = self.clean_whitespace(text)
        sanitized = self.mask_sensitive_data(cleaned)
        return sanitized

