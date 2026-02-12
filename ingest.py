# ingest.py
from __future__ import annotations
from typing import List

def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Simple character-based chunker with overlap.
    Good enough to start; improves retrieval vs storing whole docs.
    """
    text = (text or "").strip()
    if not text:
        return []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # If we've reached the end, stop - otherwise we infinite loop
        if end == n:
            break
        
        start = end - overlap  # overlap
        if start < 0:
            start = 0

    return chunks
