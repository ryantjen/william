# ingest.py
from __future__ import annotations
import re
from typing import List


def chunk_text(text: str, max_chars: int = 2000, overlap: int = 200) -> List[str]:
    """
    Smart chunker that prefers natural boundaries:
    1. Section headers (markdown # or numbered sections)
    2. Paragraph boundaries (double newlines)
    3. Sentence boundaries
    4. Falls back to character limit if needed
    """
    text = (text or "").strip()
    if not text:
        return []
    
    # If text is small enough, return as single chunk
    if len(text) <= max_chars:
        return [text]
    
    # First, try to split by sections (headers)
    sections = split_by_sections(text)
    
    # Process each section, splitting further if needed
    chunks = []
    for section in sections:
        if len(section) <= max_chars:
            if section.strip():
                chunks.append(section.strip())
        else:
            # Section too big - split by paragraphs
            sub_chunks = split_by_paragraphs(section, max_chars)
            chunks.extend(sub_chunks)
    
    # Add overlap between chunks for context continuity
    if overlap > 0 and len(chunks) > 1:
        chunks = add_overlap(chunks, overlap)
    
    return [c for c in chunks if c.strip()]


def split_by_sections(text: str) -> List[str]:
    """Split text by section headers (markdown # or numbered sections like '1.' '2.')"""
    # Pattern matches: markdown headers, numbered sections, or roman numerals
    section_pattern = r'(?=\n(?:#{1,6}\s|(?:\d+\.|\([a-z]\)|\([0-9]+\)|[IVX]+\.)\s))'
    
    parts = re.split(section_pattern, text)
    
    # If no sections found, return original text
    if len(parts) <= 1:
        return [text]
    
    return [p.strip() for p in parts if p.strip()]


def split_by_paragraphs(text: str, max_chars: int) -> List[str]:
    """Split text by paragraph boundaries, then by sentences if still too large."""
    # Split by double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
            
        # If adding this paragraph would exceed limit
        if len(current_chunk) + len(para) + 2 > max_chars:
            # Save current chunk if it has content
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too big, split by sentences
            if len(para) > max_chars:
                sentence_chunks = split_by_sentences(para, max_chars)
                chunks.extend(sentence_chunks)
                current_chunk = ""
            else:
                current_chunk = para
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + para
            else:
                current_chunk = para
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def split_by_sentences(text: str, max_chars: int) -> List[str]:
    """Split text by sentence boundaries as last resort."""
    # Split on sentence endings followed by space or newline
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        if len(current_chunk) + len(sentence) + 1 > max_chars:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # If single sentence is too long, force split by characters
            if len(sentence) > max_chars:
                # Split long sentence at word boundaries
                words = sentence.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_chars:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        current_chunk = word
                    else:
                        current_chunk = (current_chunk + " " + word).strip()
            else:
                current_chunk = sentence
        else:
            current_chunk = (current_chunk + " " + sentence).strip()
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Add overlap from previous chunk to current chunk for context."""
    if not chunks or overlap <= 0:
        return chunks
    
    overlapped = [chunks[0]]
    
    for i in range(1, len(chunks)):
        prev_chunk = chunks[i - 1]
        curr_chunk = chunks[i]
        
        # Get last `overlap` characters from previous chunk
        if len(prev_chunk) > overlap:
            overlap_text = prev_chunk[-overlap:]
            # Try to start at a word boundary
            space_idx = overlap_text.find(' ')
            if space_idx > 0:
                overlap_text = overlap_text[space_idx + 1:]
            overlapped.append(f"...{overlap_text}\n\n{curr_chunk}")
        else:
            overlapped.append(curr_chunk)
    
    return overlapped
