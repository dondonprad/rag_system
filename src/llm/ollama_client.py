"""Klien & Orkestrator LLM On-Premise menggunakan Ollama Lokal + Intel IPEX-LLM.

Mendukung akselerasi GPU Intel (Intel Arc Graphics via SYCL / Level-Zero) dengan model
instruksi Qwen 2.5-7B terkuantisasi 4-bit (GGUF / INT4 Q4_K_M).

Aturan Operasional RAG Militer PUSDIKZI:
1. Context Injection & Prompt Engineering Militer dari Vector Store ChromaDB.
2. Strict Grounding: Menjawab HANYA berdasarkan potongan konteks yang disuplai ChromaDB.
3. Penanganan Kondisi No Match: Mengembalikan pesan standar apabila konteks tidak ditemukan
   atau di luar ambang batas relevansi semantik.
4. Penyertaan Sitasi Wajib: Setiap jawaban wajib mencantumkan Nama Dokumen dan Halaman.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import httpx

from config.settings import settings

logger = logging.getLogger(__name__)

# Konfigurasi variabel lingkungan akselerasi Intel IPEX-LLM (SYCL / Level-Zero)
os.environ.setdefault("SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS", "1")
os.environ.setdefault("ZES_ENABLE_SYSMAN", "1")
os.environ.setdefault("ZE_ENABLE_PCI_ID_DEVICE_ORDER", "1")
os.environ.setdefault("ONEAPI_DEVICE_SELECTOR", "level_zero:0")
os.environ.setdefault("OLLAMA_NUM_GPU", "999")

DEFAULT_NO_MATCH_MESSAGE = "Informasi tidak ditemukan dalam dokumen doktrin PUSDIKZI."

MILITARY_SYSTEM_PROMPT = (
    "Anda adalah Asisten AI Doktrin & Administrasi Militer Pusat Pendidikan Zeni (PUSDIKZI) TNI AD.\n"
    "Patuhi 3 aturan ketat berikut dalam Bahasa Indonesia:\n"
    "1. STRICT GROUNDING: Jawab HANYA berdasarkan fakta di dalam <konteks>. Dilarang menambah asumsi di luar <konteks>.\n"
    f"2. NO MATCH: Jika jawaban tidak ada di dalam <konteks>, jawab persis: \"{DEFAULT_NO_MATCH_MESSAGE}\"\n"
    "3. SITASI WAJIB: Cantumkan sitasi sumber di akhir kalimat/poin dengan format [Dokumen: <Nama Dokumen>, Halaman: <Halaman>]. "
    "Jangan menyalin ulang blok <konteks>."
)


class OllamaClient:
    """Wrapper klien HTTP ke server Ollama + IPEX-LLM lokal serta engine sintesis RAG Militer."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        num_ctx: Optional[int] = None,
        max_distance_threshold: Optional[float] = None,
    ):
        llm_cfg = settings.get("llm", {})
        self.base_url = (base_url or os.getenv("OLLAMA_URL") or llm_cfg.get("base_url", "http://localhost:11434")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or llm_cfg.get("model_name", "qwen2.5-7b-int4")
        self.temperature = (
            temperature
            if temperature is not None
            else float(os.getenv("LLM_TEMPERATURE", llm_cfg.get("temperature", 0.1)))
        )
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(os.getenv("LLM_MAX_TOKENS", llm_cfg.get("max_tokens", 256)))
        )
        self.num_ctx = (
            num_ctx
            if num_ctx is not None
            else int(os.getenv("LLM_NUM_CTX", llm_cfg.get("num_ctx", 2048)))
        )
        self.max_distance_threshold = (
            max_distance_threshold
            if max_distance_threshold is not None
            else float(os.getenv("RAG_MAX_DISTANCE", llm_cfg.get("max_distance_threshold", 0.42)))
        )
        self.no_match_message = llm_cfg.get("no_match_message", DEFAULT_NO_MATCH_MESSAGE)

        self.client = httpx.Client(timeout=300.0)
        logger.info(
            f"OllamaClient (IPEX-LLM Intel GPU) siap: url={self.base_url}, model={self.model}, "
            f"quantization=4-bit GGUF/INT4"
        )

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def is_healthy(self) -> bool:
        """Memeriksa apakah container/service Ollama IPEX-LLM aktif dan merespons."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception as e:
            logger.error(f"Gagal terhubung ke Ollama IPEX-LLM pada {self.base_url}: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Mengambil informasi model GGUF/INT4 yang terdaftar di Ollama IPEX-LLM."""
        try:
            resp = self.client.get(f"{self.base_url}/api/tags", timeout=5.0)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            target_info = None
            for m in models:
                if m.get("name", "").startswith(self.model.split(":")[0]):
                    target_info = m
                    break
            if not target_info and models:
                target_info = models[0]
            return {
                "healthy": True,
                "base_url": self.base_url,
                "configured_model": self.model,
                "acceleration": "Intel IPEX-LLM (SYCL / Level-Zero GPU)",
                "available_models": [m.get("name") for m in models],
                "active_model_details": target_info.get("details", {}) if target_info else {},
            }
        except Exception as e:
            return {
                "healthy": False,
                "base_url": self.base_url,
                "configured_model": self.model,
                "error": str(e),
            }

    @staticmethod
    def _clean_response_text(text: str) -> str:
        """Membersihkan blok <think>...</think> atau pengulangan tag konteks jika ada."""
        if not text:
            return ""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"</?think>", "", cleaned).strip()
        cleaned = re.sub(r"</?konteks>", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def generate(
        self,
        prompt: str,
        system: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Menjalankan inferensi teks pada model Qwen 2.5-7B 4-bit GGUF melalui Ollama."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "system": system or MILITARY_SYSTEM_PROMPT,
            "stream": False,
            "think": False,
            "keep_alive": "24h",
            "options": {
                "temperature": self.temperature if temperature is None else temperature,
                "num_predict": self.max_tokens if max_tokens is None else max_tokens,
                "num_ctx": self.num_ctx,
                "top_p": 0.9,
                "stop": ["</konteks>", "<konteks>", "[PERTANYAAN]", "--- KONTEKS"],
            },
        }
        logger.info(f"Menjalankan inferensi LLM {self.model} pada Intel GPU (panjang prompt={len(prompt)} char)...")
        resp = self._post("/api/generate", payload)
        raw_output = resp.get("response", "")
        return self._clean_response_text(raw_output)

    @staticmethod
    def extract_chunk_citation(chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Mengekstrak atribut sitasi standar (Nama Dokumen & Halaman) dari metadata ChromaDB."""
        meta = chunk.get("metadata") or {}
        doc_name = (
            meta.get("sumber")
            or meta.get("file_name")
            or (os.path.basename(str(meta.get("source"))) if meta.get("source") else None)
            or "Dokumen_Tidak_Diketahui"
        )
        raw_page = meta.get("halaman") if meta.get("halaman") is not None else meta.get("page", 1)
        try:
            page_num = int(raw_page)
        except (ValueError, TypeError):
            page_num = 1

        classification = str(
            meta.get("klasifikasi") or meta.get("classification") or "INTERNAL"
        ).upper()
        distance = chunk.get("distance")

        return {
            "document": str(doc_name),
            "page": page_num,
            "classification": classification,
            "distance": round(float(distance), 4) if distance is not None else None,
            "citation_tag": f"[Dokumen: {doc_name}, Halaman: {page_num}]",
        }

    def build_military_rag_prompt(
        self, query: str, chunks: List[Dict[str, Any]]
    ) -> Tuple[str, str, List[Dict[str, Any]]]:
        """Menyusun prompt injeksi konteks dari ChromaDB beserta instruksi strict grounding & sitasi wajib."""
        context_blocks: List[str] = []
        citations: List[Dict[str, Any]] = []
        seen_citations = set()

        for idx, chunk in enumerate(chunks, start=1):
            content = (chunk.get("content") or "").strip()
            if not content:
                continue
            cit = self.extract_chunk_citation(chunk)
            cit_key = (cit["document"], cit["page"])
            if cit_key not in seen_citations:
                seen_citations.add(cit_key)
                citations.append(cit)

            context_blocks.append(
                f"[{idx}] {cit['citation_tag']} (Klasifikasi: {cit['classification']}):\n{content}"
            )

        context_str = "\n\n".join(context_blocks)
        user_prompt = (
            f"<konteks>\n{context_str}\n</konteks>\n\n"
            f"Pertanyaan: {query}\n"
            "Instruksi: Jawab pertanyaan di atas secara langsung dan ringkas HANYA berdasarkan <konteks> di atas "
            "dan sertakan sitasi [Dokumen: <Nama Dokumen>, Halaman: <Halaman>]. "
            f"Jika jawaban tidak ditemukan di dalam <konteks>, tulis persis: \"{self.no_match_message}\"\n"
            "Jawaban:"
        )
        return MILITARY_SYSTEM_PROMPT, user_prompt, citations

    @staticmethod
    def _is_no_match_output(answer_text: str, no_match_message: str) -> bool:
        """Mendeteksi apakah output LLM menyatakan kondisi no-match (informasi tidak ditemukan)."""
        if not answer_text:
            return True
        norm = answer_text.strip().lower()
        triggers = [
            no_match_message.lower().rstrip("."),
            "informasi tidak ditemukan dalam dokumen doktrin pusdikzi",
            "tidak ada data relevan",
            "tidak ditemukan dalam konteks",
            "tidak ditemukan di dalam konteks",
            "tidak disebutkan dalam dokumen",
            "tidak terdapat informasi",
        ]
        return any(t in norm for t in triggers)

    @staticmethod
    def enforce_mandatory_citations(answer_text: str, citations: List[Dict[str, Any]]) -> str:
        """Memastikan jawaban menyertakan sitasi wajib (Nama Dokumen & Halaman)."""
        if not citations:
            return answer_text

        has_inline_citation = bool(
            re.search(
                r"\[(?:Dokumen|Document|Sumber)\s*:\s*[^,\]]+,\s*(?:Halaman|Page|Hal)\s*:\s*\d+\]",
                answer_text,
                re.IGNORECASE,
            )
        )

        normalized_answer = re.sub(
            r"\[Document\s*:\s*([^,\]]+),\s*Page\s*:\s*(\d+)\]",
            r"[Dokumen: \1, Halaman: \2]",
            answer_text,
            flags=re.IGNORECASE,
        )

        if not has_inline_citation:
            primary_tag = citations[0]["citation_tag"]
            normalized_answer = f"{normalized_answer.rstrip()} {primary_tag}"

        if "Sumber Sitasi:" not in normalized_answer and "Referensi Dokumen:" not in normalized_answer:
            citation_lines = [
                f"- {c['citation_tag']} (Klasifikasi: {c['classification']})"
                for c in citations
            ]
            normalized_answer = (
                f"{normalized_answer}\n\n"
                "Sumber Sitasi:\n" + "\n".join(citation_lines)
            )

        return normalized_answer

    def answer_with_context(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        max_distance_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Menjalankan alur penuh Context Injection, Strict Grounding, No-Match Handling, dan Sitasi Wajib."""
        threshold = (
            self.max_distance_threshold
            if max_distance_threshold is None
            else max_distance_threshold
        )

        # 1. Penanganan kondisi No Match jika hasil retrieval kosong
        if not retrieved_chunks:
            logger.info(f"Kondisi No Match: Tidak ada chunk dari ChromaDB untuk kueri '{query}'.")
            return {
                "answer": self.no_match_message,
                "citations": [],
                "retrieved_chunks": [],
                "no_match": True,
                "model": self.model,
            }

        # 2. Filter relevansi berdasarkan Cosine Distance threshold (Strict Grounding)
        relevant_chunks = [
            c for c in retrieved_chunks
            if c.get("distance") is None or float(c["distance"]) <= threshold
        ]

        if not relevant_chunks:
            best_dist = min((float(c["distance"]) for c in retrieved_chunks if c.get("distance") is not None), default=1.0)
            logger.info(
                f"Kondisi No Match: Seluruh chunk melebihi batas cosine distance "
                f"(terbaik={best_dist:.4f} > threshold={threshold:.4f}) untuk kueri '{query}'."
            )
            return {
                "answer": self.no_match_message,
                "citations": [],
                "retrieved_chunks": [],
                "no_match": True,
                "model": self.model,
            }

        # 3. Bangun prompt militer dengan injeksi konteks dan sitasi
        system_prompt, user_prompt, citations = self.build_military_rag_prompt(query, relevant_chunks)

        # 4. Panggil inferensi LLM lokal (Qwen 2.5-7B INT4 via Ollama IPEX-LLM)
        raw_answer = self.generate(prompt=user_prompt, system=system_prompt)

        # 5. Periksa apakah LLM menyatakan informasi tidak ditemukan di dalam konteks
        if self._is_no_match_output(raw_answer, self.no_match_message):
            return {
                "answer": self.no_match_message,
                "citations": [],
                "retrieved_chunks": relevant_chunks,
                "no_match": True,
                "model": self.model,
            }

        # 6. Pastikan sitasi wajib (Nama Dokumen & Halaman) tercantum dalam jawaban final
        final_answer = self.enforce_mandatory_citations(raw_answer, citations)

        return {
            "answer": final_answer,
            "citations": citations,
            "retrieved_chunks": relevant_chunks,
            "no_match": False,
            "model": self.model,
        }
