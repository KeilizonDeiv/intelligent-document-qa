"""Parses uploaded documents (PDF/DOCX/TXT/MD) into overlapping text chunks."""

import hashlib
import os
import re
from datetime import datetime, timezone

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None


class DocumentChunk:
    def __init__(self, text: str, metadata: dict, chunk_id: str | None = None):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id or self._generate_id(text, metadata)

    def _generate_id(self, text: str, metadata: dict) -> str:
        content = f"{text}_{metadata.get('source', '')}_{metadata.get('chunk_num', 0)}"
        return hashlib.md5(content.encode()).hexdigest()

    def to_dict(self) -> dict:
        return {"id": self.chunk_id, "text": self.text, "metadata": self.metadata}


class DocumentProcessor:
    """Extracts text from supported formats and splits it into overlapping chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = [".txt", ".pdf", ".docx", ".md"]

    def process_file(
        self,
        filepath: str,
        session_id: str | None = None,
        source_name: str | None = None,
    ) -> list[DocumentChunk]:
        """Read and chunk a file on disk.

        `source_name` is the human-readable name recorded in each chunk's
        metadata (and later used for listing/deleting documents). It defaults
        to the file's on-disk basename, but callers that save uploads under a
        disk-unique name (e.g. timestamp-prefixed to avoid collisions) should
        pass the original display name explicitly so it matches what the
        client sees.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {ext}")

        if ext == ".pdf":
            text, pages_metadata = self._process_pdf(filepath)
        elif ext == ".docx":
            text, pages_metadata = self._process_docx(filepath)
        else:
            text, pages_metadata = self._process_text(filepath)

        source = source_name or os.path.basename(filepath)
        chunks = self._create_chunks(text, source, pages_metadata, session_id)
        return chunks

    def _process_pdf(self, filepath: str) -> tuple[str, list[dict]]:
        if PdfReader is None:
            raise ImportError("pypdf not installed. Run: pip install pypdf")

        reader = PdfReader(filepath)
        text = ""
        pages_metadata = []

        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            text += page_text + "\n\n"
            pages_metadata.append({"page": page_num, "length": len(page_text)})

        return text, pages_metadata

    def _process_docx(self, filepath: str) -> tuple[str, list[dict]]:
        if DocxDocument is None:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        doc = DocxDocument(filepath)
        text = ""
        pages_metadata = []

        for i, para in enumerate(doc.paragraphs, 1):
            para_text = para.text
            text += para_text + "\n"
            # Every paragraph gets an entry (even blank ones) so _create_chunks
            # can reconstruct raw-text offsets purely from cumulative lengths.
            pages_metadata.append({"paragraph": i, "length": len(para_text)})

        return text, pages_metadata

    def _process_text(self, filepath: str) -> tuple[str, list[dict]]:
        with open(filepath, encoding="utf-8") as f:
            text = f.read()

        lines = text.split("\n")
        pages_metadata = [{"line_count": len(lines), "length": len(text)}]

        return text, pages_metadata

    def _segment_by_metadata(self, text: str, pages_metadata: list[dict]) -> list[tuple[str | None, int, str]]:
        """Split raw text into (label_key, label_value, raw_segment_text) triples.

        pages_metadata entries are produced by _process_pdf ("page", joined
        with "\\n\\n") or _process_docx ("paragraph", joined with "\\n"). Both
        record each segment's pre-clean length, so the original separator
        lengths let us slice the exact raw segment back out without needing
        to change how text is built upstream.
        """
        if not pages_metadata:
            return [(None, 0, text)]

        first = pages_metadata[0]
        if "page" in first:
            label_key, sep_len = "page", 2
        elif "paragraph" in first:
            label_key, sep_len = "paragraph", 1
        else:
            return [(None, 0, text)]

        segments = []
        cursor = 0
        for meta in pages_metadata:
            length = meta.get("length", 0)
            segments.append((label_key, meta[label_key], text[cursor : cursor + length]))
            cursor += length + sep_len

        return segments

    def _label_for_position(self, boundaries: list[tuple[int, int]], pos: int) -> int | None:
        label = None
        for offset, value in boundaries:
            if offset > pos:
                break
            label = value
        return label

    def _create_chunks(
        self,
        text: str,
        source: str,
        pages_metadata: list[dict],
        session_id: str | None = None,
    ) -> list[DocumentChunk]:
        segments = self._segment_by_metadata(text, pages_metadata)

        label_key = None
        cleaned_segments = []
        boundaries: list[tuple[int, int]] = []
        offset = 0

        for key, value, raw_segment in segments:
            cleaned = self._clean_text(raw_segment)
            if not cleaned:
                continue
            label_key = label_key or key
            cleaned_segments.append(cleaned)
            boundaries.append((offset, value))
            offset += len(cleaned) + 2

        text = "\n\n".join(cleaned_segments)

        chunks = []
        start = 0
        chunk_num = 0

        while start < len(text):
            end = start + self.chunk_size

            if end < len(text):
                search_start = max(start, end - 100)
                sentence_end = max(
                    text.rfind(". ", search_start, end),
                    text.rfind("! ", search_start, end),
                    text.rfind("? ", search_start, end),
                    text.rfind("\n\n", search_start, end),
                )

                if sentence_end != -1 and sentence_end > start:
                    end = sentence_end + 1

            chunk_text = text[start:end].strip()

            if chunk_text:
                metadata = {
                    "source": os.path.basename(source),
                    "chunk_num": chunk_num,
                    "start_pos": start,
                    "end_pos": end,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if label_key:
                    label_value = self._label_for_position(boundaries, start)
                    if label_value is not None:
                        metadata[label_key] = label_value
                if session_id:
                    metadata["session_id"] = session_id

                chunks.append(DocumentChunk(chunk_text, metadata))
                chunk_num += 1

            start = end - self.chunk_overlap
            if start >= len(text):
                break

        return chunks

    def _clean_text(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks.

        Paragraph boundaries (blank lines) matter for chunking: _create_chunks
        looks for "\\n\\n" as a preferred sentence/section boundary. Collapsing
        all whitespace to single spaces (the naive approach) destroys that
        signal and makes every chunk boundary fall back to sentence-punctuation
        heuristics even when the source document had clear section structure.
        """
        text = re.sub(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]", "", text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        paragraphs = re.split(r"\n\s*\n", text)
        cleaned_paragraphs = [re.sub(r"\s+", " ", p).strip() for p in paragraphs]
        text = "\n\n".join(p for p in cleaned_paragraphs if p)

        text = text.replace("“", '"').replace("”", '"')
        text = text.replace("‘", "'").replace("’", "'")

        return text.strip()

    def get_document_stats(self, chunks: list[DocumentChunk]) -> dict:
        total_chars = sum(len(chunk.text) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0

        return {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": int(avg_chunk_size),
            "sources": list({chunk.metadata["source"] for chunk in chunks}),
        }
