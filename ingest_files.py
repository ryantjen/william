from __future__ import annotations
import hashlib
import io
from typing import Tuple, List, Dict, Any

import pandas as pd
from pypdf import PdfReader

from ingest import chunk_text

def file_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def ingest_txt(data: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Ingest plain text file from bytes."""
    text = data.decode("utf-8", errors="ignore")
    chunks = chunk_text(text)
    metas = [{"type": "txt_chunk", "name": filename, "chunk": i} for i in range(len(chunks))]
    return chunks, metas

def ingest_csv(data: bytes, filename: str) -> Tuple[List[str], List[Dict[str, Any]], pd.DataFrame]:
    """Ingest CSV file from bytes."""
    bio = io.BytesIO(data)
    df = pd.read_csv(bio)
    csv_text = df.to_csv(index=False)
    chunks = chunk_text(csv_text, max_chars=3000, overlap=200)
    metas = [{"type": "csv_chunk", "name": filename, "chunk": i} for i in range(len(chunks))]
    return chunks, metas, df

def ingest_pdf(data: bytes, filename: str, max_pages: int = 60) -> Tuple[List[str], List[Dict[str, Any]], int]:
    """Ingest text-based PDF file from bytes."""
    pdf_stream = io.BytesIO(data)
    reader = PdfReader(pdf_stream)
    total = min(len(reader.pages), max_pages)

    all_chunks = []
    all_metas = []
    pages_with_text = 0

    for page_idx in range(total):
        page = reader.pages[page_idx]
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        pages_with_text += 1

        chunks = chunk_text(page_text, max_chars=3000, overlap=200)
        for j, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_metas.append({
                "type": "pdf_chunk",
                "name": filename,
                "page": page_idx + 1,
                "chunk": j
            })

    return all_chunks, all_metas, pages_with_text
